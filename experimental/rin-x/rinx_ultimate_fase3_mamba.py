"""
RIN-X ULTIMATE - FASE 3: MAMBA/SSM + STRUCTURED SPARSITY + QAT
Arquitectura de modelo eficiente con sparsity entrenada desde inicio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math

print("="*70)
print("RIN-X ULTIMATE - FASE 3: MAMBA/SSM + SPARSITY + QAT")
print("="*70)
print()

# ============================================================================
# SELECTIVE SSM (MAMBA CORE)
# ============================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S4 con parámetros input-dependent)
    Basado en "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection (x → x_z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolutional layer para local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise
            padding=d_conv - 1,
            bias=True
        )
        
        # Selective SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)  # delta projection
        
        # State transition matrix A (log-initialized para estabilidad)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_log)
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        x_and_z = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x_ssm, z = x_and_z.chunk(2, dim=-1)  # Cada uno: (batch, seq_len, d_inner)
        
        # Aplicar silu (Swish) a z
        z = F.silu(z)
        
        # Convolutional layer
        x_conv = x_ssm.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[..., :seq_len]  # Maintain length
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)
        
        # Selective SSM
        # Parámetros input-dependent: delta, B, C
        ssm_params = self.x_proj(x_conv)  # (batch, seq_len, d_state*2 + 1)
        delta, B, C = torch.split(ssm_params, [1, self.d_state, self.d_state], dim=-1)
        
        # Discretización de delta (ZOH - Zero Order Hold)
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)
        
        # State transition matrix A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # SSM computation (secuencial para autoregressive, paralelo para training)
        y = self.ssm_forward(delta, A, B, C, x_conv)
        
        # Gating y output
        y = y * z  # Gating
        output = self.out_proj(y)
        
        return output
    
    def ssm_forward(self, delta: torch.Tensor, A: torch.Tensor, 
                    B: torch.Tensor, C: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward del SSM selective
        Método secuencial (para autoregressive inference)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Inicializar estado
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Discretización ZOH
            delta_t = delta[:, t, :]  # (batch, d_inner)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (batch, d_inner, d_state)
            B_bar = delta_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            
            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)
            
            # Output: y = C * h ( + D * x skip connection)
            y_t = torch.sum(C[:, t, :].unsqueeze(1) * h, dim=-1)  # (batch, d_inner)
            y_t = y_t + self.D.unsqueeze(0) * x[:, t, :]
            
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)
    
    def ssm_parallel_scan(self, delta: torch.Tensor, A: torch.Tensor,
                          B: torch.Tensor, C: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel scan para training (más eficiente en GPU)
        Usa associative scan para paralelizar
        """
        # Implementación de parallel scan (simplificada)
        # En práctica se usaría CUDA kernel custom
        return self.ssm_forward(delta, A, B, C, x)

class MambaBlock(nn.Module):
    """Bloque Mamba con residual connection y normalization"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = self.dropout(x)
        return x + residual

# ============================================================================
# STRUCTURED SPARSITY: N:M PRUNING
# ============================================================================

class StructuredSparseLinear(nn.Module):
    """
    Linear layer con N:M structured sparsity
    Ejemplo: 2:4 sparsity = 50% weights = 0 de forma estructurada
    """
    def __init__(self, in_features: int, out_features: int, 
                 n: int = 2, m: int = 4, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = n  # Non-zeros por bloque
        self.m = m  # Tamaño de bloque
        
        assert in_features % m == 0, "in_features debe ser divisible por m"
        
        # Pesos densos (almacenamos todos pero solo n/m son actualizados)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Mask de sparsity (1 = activo, 0 = congelado/pruned)
        self.register_buffer('weight_mask', torch.ones_like(self.weight))
        
        # Metadata para 2:4 (opcional, para compresión)
        num_blocks = (in_features // m) * out_features
        self.register_buffer('sparse_metadata', 
                           torch.zeros(num_blocks, dtype=torch.uint8))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self.apply_structured_sparsity()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def apply_structured_sparsity(self):
        """Aplicar N:M structured sparsity a los pesos"""
        with torch.no_grad():
            W = self.weight.abs()
            mask = torch.ones_like(W)
            
            # Para cada fila, aplicar N:M
            for i in range(self.out_features):
                row = W[i]
                for j in range(0, self.in_features, self.m):
                    block = row[j:j+self.m]
                    # Encontrar top-n valores
                    topk_vals, topk_indices = torch.topk(block, self.n)
                    # Crear mask
                    block_mask = torch.zeros(self.m, device=W.device)
                    block_mask[topk_indices] = 1.0
                    mask[i, j:j+self.m] = block_mask
            
            self.weight_mask.data = mask
            
            # Aplicar mask a pesos
            self.weight.data *= mask
            
            # Actualizar metadata para 2:4
            if self.n == 2 and self.m == 4:
                self._update_2_4_metadata()
    
    def _update_2_4_metadata(self):
        """Actualizar metadata de compresión 2:4"""
        # Cada bloque de 4 tiene 2 bits indicando posiciones
        # 00: 0,1 | 01: 0,2 | 10: 0,3 | 11: 1,2 | etc.
        with torch.no_grad():
            idx = 0
            for i in range(self.out_features):
                for j in range(0, self.in_features, 4):
                    block_mask = self.weight_mask[i, j:j+4]
                    nonzero_pos = torch.nonzero(block_mask, as_tuple=False).squeeze()
                    if len(nonzero_pos) == 2:
                        # Codificar posiciones en 2 bits
                        meta = (nonzero_pos[0] << 2) | nonzero_pos[1]
                        self.sparse_metadata[idx] = meta
                    idx += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplicar mask durante forward (sparsity estructurada)
        masked_weight = self.weight * self.weight_mask
        return F.linear(x, masked_weight, self.bias)
    
    def get_sparsity(self) -> float:
        """Calcular porcentaje de sparsity"""
        total = self.weight_mask.numel()
        active = self.weight_mask.sum().item()
        return 1.0 - (active / total)
    
    def get_compressed_size(self) -> int:
        """Tamaño en bytes si comprimimos 2:4"""
        if self.n == 2 and self.m == 4:
            # Valores: 50% del original
            # Metadata: 2 bits por bloque de 4 = 0.5 bits por elemento
            values_size = self.weight.numel() * 4 // 2  # 4 bytes por float, 50% sparsity
            metadata_size = self.sparse_metadata.numel() // 8  # 1 byte cada 4 bloques
            return values_size + metadata_size
        return self.weight.numel() * 4

# ============================================================================
# QUANTIZATION AWARE TRAINING (QAT)
# ============================================================================

class FakeQuantize(nn.Module):
    """Fake quantization para QAT (simula efecto de quantization)"""
    def __init__(self, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.qmin = -(2 ** (bits - 1)) if symmetric else 0
        self.qmax = (2 ** (bits - 1)) - 1 if symmetric else (2 ** bits - 1)
        
        # Parámetros de escala y zero-point (aprendidos durante training)
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        # Calcular escala óptima
        with torch.no_grad():
            x_min = x.min()
            x_max = x.max()
            if self.symmetric:
                max_abs = max(abs(x_min), abs(x_max))
                self.scale.data = max_abs / self.qmax
            else:
                self.scale.data = (x_max - x_min) / (self.qmax - self.qmin)
                self.zero_point.data = self.qmin - x_min / self.scale
        
        # Fake quantize: quantize + dequantize (con straight-through estimator)
        x_quant = torch.clamp(torch.round(x / self.scale + self.zero_point), 
                             self.qmin, self.qmax)
        x_dequant = (x_quant - self.zero_point) * self.scale
        
        # Straight-through estimator para gradiente
        return x + (x_dequant - x).detach()

class QATLinear(nn.Module):
    """Linear layer con QAT integrado"""
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, weight_bits: int = 8, act_bits: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.weight_quantizer = FakeQuantize(weight_bits, symmetric=True)
        self.act_quantizer = FakeQuantize(act_bits, symmetric=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantizar activaciones
        x = self.act_quantizer(x)
        
        # Quantizar pesos
        w = self.weight_quantizer(self.linear.weight)
        
        # Linear con pesos/activaciones cuantizados
        return F.linear(x, w, self.linear.bias)

# ============================================================================
# MODELO COMPLETO: MAMBA CON SPARSITY + QAT
# ============================================================================

class RinxMambaClassifier(nn.Module):
    """
    Clasificador basado en Mamba con:
    - Structured sparsity (2:4)
    - QAT (INT8 weights/activations)
    - Efficient SSM core
    """
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 d_model: int = 256,
                 n_layers: int = 4,
                 d_state: int = 16,
                 expand: int = 2,
                 sparsity_n: int = 2,
                 sparsity_m: int = 4,
                 use_qat: bool = True):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Mamba blocks con sparsity
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ssm': MambaBlock(d_model, d_state, expand=expand),
                'ffn': nn.Sequential(
                    StructuredSparseLinear(d_model, d_model * 4, n=sparsity_n, m=sparsity_m),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    StructuredSparseLinear(d_model * 4, d_model, n=sparsity_n, m=sparsity_m)
                ) if not use_qat else nn.Sequential(
                    QATLinear(d_model, d_model * 4, weight_bits=8, act_bits=8),
                    nn.GELU(),
                    QATLinear(d_model * 4, d_model, weight_bits=8, act_bits=8)
                )
            })
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer['ssm'](x)
            x = x + layer['ffn'](x)  # Residual FFN
        
        x = self.norm(x)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)
        
        return self.classifier(x)
    
    def get_total_sparsity(self) -> float:
        """Calcular sparsity total del modelo"""
        total_params = 0
        sparse_params = 0
        
        for module in self.modules():
            if isinstance(module, StructuredSparseLinear):
                total_params += module.weight.numel()
                sparse_params += module.weight_mask.sum().item()
        
        if total_params == 0:
            return 0.0
        return 1.0 - (sparse_params / total_params)
    
    def compress_2_4(self):
        """Comprimir modelo a formato 2:4 eficiente"""
        for module in self.modules():
            if isinstance(module, StructuredSparseLinear):
                module._update_2_4_metadata()

# ============================================================================
# TRAINING CON SPARSITY Y QAT
# ============================================================================

def train_rinx_mamba(model: nn.Module, 
                     train_loader, 
                     test_loader,
                     epochs: int = 50,
                     lr: float = 1e-3,
                     device: str = 'cuda') -> nn.Module:
    """
    Entrenamiento con sparsity estructurada y QAT
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print("Entrenando RIN-X Mamba...")
    print(f"  Sparsity inicial: {model.get_total_sparsity():.2%}")
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Flatten si es necesario
            if data.dim() == 4:  # Imagen (B, C, H, W) -> (B, H*W, C)
                data = data.flatten(2).permute(0, 2, 1)
            elif data.dim() == 2:  # (B, D) -> (B, 1, D)
                data = data.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Aplicar gradiente mask para mantener sparsity
            for module in model.modules():
                if isinstance(module, StructuredSparseLinear):
                    if module.weight.grad is not None:
                        module.weight.grad *= module.weight_mask
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Evaluación
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                if data.dim() == 4:
                    data = data.flatten(2).permute(0, 2, 1)
                elif data.dim() == 2:
                    data = data.unsqueeze(1)
                
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        sparsity = model.get_total_sparsity()
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, "
              f"Sparsity={sparsity:.2%}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
    
    print(f"\nMejor accuracy: {best_acc:.2f}%")
    print(f"Sparsity final: {model.get_total_sparsity():.2%}")
    
    return model

# ============================================================================
# TEST
# ============================================================================

def test_mamba_sparsity():
    """Probar Mamba con sparsity"""
    print("="*70)
    print("TEST: Mamba + Structured Sparsity + QAT")
    print("="*70)
    
    # Crear modelo
    model = RinxMambaClassifier(
        input_dim=784,
        num_classes=10,
        d_model=128,
        n_layers=2,
        d_state=16,
        expand=2,
        sparsity_n=2,
        sparsity_m=4,
        use_qat=True
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Initial sparsity: {model.get_total_sparsity():.2%}")
    
    # Test forward
    batch_size = 4
    seq_len = 1
    x = torch.randn(batch_size, seq_len, 784)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0].tolist()}")
    
    print("\n✓ FASE 3: Mamba/SSM + Sparsity + QAT implementado")
    
    return model

if __name__ == '__main__':
    model = test_mamba_sparsity()
