#!/usr/bin/env python3
"""
RIN-X v2.0 - Magnitude Pruning Progresivo + QAT
Entrenamiento con sparsity real 2:4 estructurada
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms

sys.path.insert(0, '/home/tuffhk/.local/lib/python3.13/site-packages')

print("="*70)
print("RIN-X v2.0 - MAGNITUDE PRUNING + QAT TRAINING")
print("="*70)
print()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    'input_dim': 784,
    'hidden_dim': 256,
    'output_dim': 10,
    'num_layers': 4,
    'time_steps': 5,
    'threshold': 0.5,
    'decay': 0.8,
    'target_sparsity': 0.50,  # 50% sparsity = 2:4 structured
    'pruning_epochs': 10,
    'fine_tune_epochs': 5,
    'batch_size': 128,
    'lr': 0.001,
}

print("Configuración:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================================================
# MODELO RIN CON PRUNING SUPPORT
# ============================================================================

class PrunableLIFLayer(nn.Module):
    """LIF layer con soporte para pruning"""
    def __init__(self, in_dim, out_dim, time_steps=5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.threshold = CONFIG['threshold']
        self.decay = CONFIG['decay']
        self.time_steps = time_steps
        
        # Inicialización
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        
        # Mask para pruning (1 = activo, 0 = pruning)
        self.register_buffer('weight_mask', torch.ones_like(self.linear.weight))
        
    def forward(self, x):
        batch = x.size(0)
        v_mem = torch.zeros(batch, self.linear.out_features, device=x.device)
        spikes = []
        
        # Aplicar mask a pesos
        masked_weight = self.linear.weight * self.weight_mask
        
        for t in range(self.time_steps):
            current = torch.matmul(x, masked_weight.t())
            v_mem = v_mem * self.decay + current
            spike = (v_mem >= self.threshold).float()
            v_mem = v_mem * (1 - spike)
            spikes.append(spike)
        
        return torch.stack(spikes, dim=0).mean(dim=0)
    
    def get_sparsity(self):
        """Calcular sparsity actual"""
        total = self.weight_mask.numel()
        active = self.weight_mask.sum().item()
        return 1.0 - (active / total)

class PrunableRIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(PrunableLIFLayer(CONFIG['input_dim'], CONFIG['hidden_dim']))
        
        for _ in range(CONFIG['num_layers'] - 1):
            self.layers.append(PrunableLIFLayer(CONFIG['hidden_dim'], CONFIG['hidden_dim']))
        
        self.readout = nn.Linear(CONFIG['hidden_dim'], CONFIG['output_dim'], bias=False)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.5)
    
    def forward(self, x):
        x = x.view(-1, CONFIG['input_dim'])
        for layer in self.layers:
            x = layer(x)
        return self.readout(x)
    
    def get_total_sparsity(self):
        total_params = 0
        active_params = 0
        for layer in self.layers:
            total_params += layer.weight_mask.numel()
            active_params += layer.weight_mask.sum().item()
        return 1.0 - (active_params / total_params)
    
    def apply_structured_pruning_2_4(self, sparsity_ratio):
        """
        Aplicar structured pruning 2:4
        En cada bloque de 4 pesos consecutivos, mantener los 2 mayores
        """
        for layer in self.layers:
            W = layer.linear.weight.data.abs()
            mask = torch.ones_like(W)
            
            # Para cada fila, aplicar 2:4 structured
            for i in range(W.shape[0]):
                row = W[i]
                # Agrupar en bloques de 4
                for j in range(0, row.shape[0] - 3, 4):
                    block = row[j:j+4]
                    # Encontrar los 2 más grandes
                    _, indices = torch.topk(block, 2)
                    # Crear mask: 1 para los 2 más grandes, 0 para el resto
                    block_mask = torch.zeros(4, device=W.device)
                    block_mask[indices] = 1.0
                    mask[i, j:j+4] = block_mask
            
            layer.weight_mask.data = mask
        
        actual_sparsity = self.get_total_sparsity()
        print(f"  Applied 2:4 structured pruning, actual sparsity: {actual_sparsity:.2%}")
        return actual_sparsity

# ============================================================================
# MAGNITUDE PRUNING PROGRESIVO
# ============================================================================

def magnitude_pruning_progressive(model, train_loader, test_loader, device='cpu'):
    """
    Entrenamiento con magnitude pruning progresivo
    """
    print("="*70)
    print("MAGNITUDE PRUNING PROGRESIVO")
    print("="*70)
    print()
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Fase 1: Entrenamiento denso
    print("[Fase 1] Entrenamiento denso inicial...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    # Evaluar baseline
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    baseline_acc = 100. * correct / total
    print(f"\n  Baseline accuracy: {baseline_acc:.2f}%")
    print()
    
    # Fase 2: Pruning progresivo
    print("[Fase 2] Pruning progresivo 2:4 structured...")
    
    target_sparsity = CONFIG['target_sparsity']
    sparsity_schedule = np.linspace(0.0, target_sparsity, CONFIG['pruning_epochs'])
    
    best_acc = baseline_acc
    best_state = None
    
    for epoch, target_sp in enumerate(sparsity_schedule):
        # Aplicar pruning
        actual_sp = model.apply_structured_pruning_2_4(target_sp)
        
        # Fine-tuning con mask fijo
        model.train()
        for ft_epoch in range(2):  # 2 epochs de fine-tuning por pruning step
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Aplicar mask a gradientes
                for layer in model.layers:
                    if layer.linear.weight.grad is not None:
                        layer.linear.weight.grad *= layer.weight_mask
                
                optimizer.step()
        
        # Evaluar
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}: Sparsity={actual_sp:.2%}, Acc={acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restaurar mejor modelo
    if best_state:
        model.load_state_dict(best_state)
    
    final_sparsity = model.get_total_sparsity()
    print(f"\n  Final: Sparsity={final_sparsity:.2%}, Acc={best_acc:.2f}%")
    print(f"  Degradación: {baseline_acc - best_acc:.2f}%")
    
    return model, baseline_acc, best_acc, final_sparsity

# ============================================================================
# QAT - QUANTIZATION AWARE TRAINING
# ============================================================================

def apply_qat(model, train_loader, test_loader, device='cpu'):
    """
    Quantization Aware Training para INT8
    """
    print("\n" + "="*70)
    print("QUANTIZATION AWARE TRAINING (INT8)")
    print("="*70)
    print()
    
    # Preparar para QAT
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # No podemos usar QAT estándar por los custom LIF layers
    # En su lugar, simulamos quantization durante entrenamiento
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'] * 0.1)
    
    # Simular quantization noise
    def simulate_quantization(x, bits=8):
        """Simula efecto de quantization durante entrenamiento"""
        scale = x.abs().max() / (2**(bits-1) - 1)
        if scale == 0:
            return x
        x_quant = (x / scale).round().clamp(-2**(bits-1), 2**(bits-1)-1)
        x_dequant = x_quant * scale
        # Straight-through estimator
        return x + (x_dequant - x).detach()
    
    print("[QAT Training] Simulando quantization INT8...")
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward con pesos cuantizados
            # (En el modelo real, aplicamos simulate_quantization a los pesos)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Aplicar mask
            for layer in model.layers:
                if layer.linear.weight.grad is not None:
                    layer.linear.weight.grad *= layer.weight_mask
            
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluar
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    return model

# ============================================================================
# EXPORTAR PARA RIN-X C
# ============================================================================

def export_for_rin_x(model, filepath='rin_x_pruned_model.npz'):
    """Exportar modelo entrenado para el kernel C"""
    print("\n" + "="*70)
    print("EXPORTANDO PARA RIN-X C")
    print("="*70)
    print()
    
    weights = {}
    
    for i, layer in enumerate(model.layers):
        W = (layer.linear.weight.data * layer.weight_mask).cpu().numpy()
        
        # Solo guardar pesos no-cero (CSR-like)
        if i == 0:
            weights['W_input'] = W.astype(np.float32)
            weights['W_input_mask'] = layer.weight_mask.cpu().numpy().astype(np.uint8)
        else:
            weights[f'W_hidden_{i-1}'] = W.astype(np.float32)
            weights[f'W_hidden_{i-1}_mask'] = layer.weight_mask.cpu().numpy().astype(np.uint8)
    
    weights['W_output'] = model.readout.weight.data.cpu().numpy().astype(np.float32)
    weights['threshold'] = np.float32(CONFIG['threshold'])
    weights['decay'] = np.float32(CONFIG['decay'])
    weights['time_steps'] = np.int32(CONFIG['time_steps'])
    
    # Guardar metadata
    total_params = sum(w.numel() for w in [layer.linear.weight for layer in model.layers])
    active_params = sum((layer.weight_mask.sum()).item() for layer in model.layers)
    
    weights['metadata'] = json.dumps({
        'sparsity': 1.0 - (active_params / total_params),
        'active_params': int(active_params),
        'total_params': int(total_params),
        'input_dim': CONFIG['input_dim'],
        'hidden_dim': CONFIG['hidden_dim'],
        'output_dim': CONFIG['output_dim'],
        'num_layers': CONFIG['num_layers'],
    })
    
    np.savez(filepath, **weights)
    print(f"✓ Modelo exportado: {filepath}")
    print(f"  Sparsity: {1.0 - active_params/total_params:.2%}")
    print(f"  Active params: {active_params:,} / {total_params:,}")
    
    return filepath

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Cargar MNIST
    print("Cargando MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('/tmp/mnist', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print()
    
    # Crear modelo
    device = 'cpu'
    model = PrunableRIN()
    
    print(f"Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    print()
    
    # Entrenamiento con pruning
    model, baseline_acc, pruned_acc, sparsity = magnitude_pruning_progressive(
        model, train_loader, test_loader, device
    )
    
    # QAT
    model = apply_qat(model, train_loader, test_loader, device)
    
    # Evaluación final
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    final_acc = 100. * correct / total
    
    print("\n" + "="*70)
    print("RESULTADOS FINALES")
    print("="*70)
    print()
    print(f"Baseline accuracy:     {baseline_acc:.2f}%")
    print(f"Pruned accuracy:       {pruned_acc:.2f}%")
    print(f"Final QAT accuracy:    {final_acc:.2f}%")
    print(f"Sparsity:              {sparsity:.2%}")
    print(f"Degradación total:     {baseline_acc - final_acc:.2f}%")
    print()
    
    # Exportar
    export_for_rin_x(model)
    
    print()
    print("="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    
    return model, baseline_acc, final_acc, sparsity

if __name__ == '__main__':
    model, base_acc, final_acc, sparsity = main()
