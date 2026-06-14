#!/usr/bin/env python3
"""
THOR GENIUS - Fast Training Pipeline
====================================
Modelo más pequeño para entrenamiento rápido + alta accuracy.
Arquitectura: 784→512→256→10 (~550K params)
Target: >97% accuracy en MNIST en ~15 minutos

Este script genera pesos reales entrenados para validar el C kernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import struct
import os
import sys
import time

print("=" * 70)
print("THOR GENIUS - FAST TRAINING (97% target)")
print("=" * 70)

# ============================================================================
# CONFIGURACIÓN OPTIMIZADA
# ============================================================================

CONFIG = {
    'input_dim': 784,
    'hidden_dims': [512, 256],      # Modelo más pequeño
    'output_dim': 10,
    'epochs': 20,                    # Menos épocas
    'batch_size': 128,               # Batch más pequeño para converger más rápido
    'lr': 0.003,
    'weight_decay': 1e-5,
    'label_smoothing': 0.05,
    'bits': 8,
    'qat_start_epoch': 5,
    'prune_start_epoch': 10,
    'target_sparsity': 0.50,
}

# ============================================================================
# MODELO
# ============================================================================

class QATLinear(nn.Module):
    """Linear con Quantization Aware Training"""
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_mask', torch.ones(out_features, in_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        w_pruned = self.weight * self.weight_mask
        
        if self.training:
            with torch.no_grad():
                w_min, w_max = w_pruned.min(), w_pruned.max()
                self.weight_scale = torch.max(torch.abs(w_min), torch.abs(w_max)) / self.qmax
                self.weight_scale = torch.clamp(self.weight_scale, min=1e-8)
        
        # Fake quantize
        w_quant = torch.clamp(torch.round(w_pruned / self.weight_scale), 
                             self.qmin, self.qmax)
        w_dequant = w_quant * self.weight_scale
        
        output = F.linear(x, w_dequant, self.bias)
        return output
    
    def get_int8_weights(self):
        with torch.no_grad():
            w_pruned = self.weight * self.weight_mask
            w_quant = torch.clamp(torch.round(w_pruned / self.weight_scale), 
                                 self.qmin, self.qmax)
            return w_quant.to(torch.int8), self.weight_scale.item()
    
    def apply_structured_prune_2_4(self):
        with torch.no_grad():
            w_abs = torch.abs(self.weight)
            for i in range(0, self.weight.size(1), 4):
                group_end = min(i + 4, self.weight.size(1))
                group = w_abs[:, i:group_end]
                threshold = torch.kthvalue(group, max(1, group.size(1)-1), dim=1).values.unsqueeze(1)
                self.weight_mask[:, i:group_end] = (group >= threshold).float()
            self.weight.data *= self.weight_mask


class ThorGeniusFast(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dims = [config['input_dim']] + config['hidden_dims'] + [config['output_dim']]
        
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.layers.append(QATLinear(dims[i], dims[i+1], bias=True, bits=config['bits']))
        self.output_layer = nn.Linear(dims[-2], dims[-1], bias=True)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, self.config['input_dim'])
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
    def get_int8_weights_all(self):
        weights_data = {}
        for i, layer in enumerate(self.layers):
            w_int8, scale = layer.get_int8_weights()
            weights_data[f'fc{i+1}'] = {
                'values': w_int8.cpu().numpy().tolist(),
                'scale': scale,
                'shape': list(w_int8.shape),
                'bias': layer.bias.detach().cpu().numpy().tolist() if layer.bias is not None else None,
            }
        weights_data['output'] = {
            'values': self.output_layer.weight.detach().cpu().numpy().tolist(),
            'bias': self.output_layer.bias.detach().cpu().numpy().tolist() if self.output_layer.bias is not None else None,
            'shape': list(self.output_layer.weight.shape),
            'scale': 1.0,
        }
        return weights_data

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('/tmp/mnist', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        total_loss += loss.item()
    return 100. * correct / total, total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total

def train():
    device = 'cpu'
    config = CONFIG
    train_loader, test_loader = get_mnist_loaders(config['batch_size'])
    
    model = ThorGeniusFast(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nArquitectura: {config['input_dim']} → {config['hidden_dims']} → {config['output_dim']}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable:,}")
    print()
    print("Epoch | Train Acc | Test Acc | Sparsity | Status")
    print("-" * 65)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], 
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    best_acc = 0.0
    sparsity = 0.0
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        if epoch == config['qat_start_epoch']:
            print(f"  → QAT ACTIVADO")
        if epoch == config['prune_start_epoch']:
            for layer in model.layers:
                if isinstance(layer, QATLinear):
                    layer.apply_structured_prune_2_4()
            print(f"  → PRUNING 2:4 APLICADO (una vez)")
        
        # Calcular sparsity después de pruning
        if epoch >= config['prune_start_epoch']:
            total_w = 0
            zero_w = 0
            for layer in model.layers:
                total_w += layer.weight_mask.numel()
                zero_w += (layer.weight_mask == 0).sum().item()
            sparsity = 100. * zero_w / total_w
        
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            status = "✓ BEST"
        else:
            status = ""
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'config': config,
        }, 'thor_genius_fast_checkpoint.pt')
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        sparsity_str = f"{sparsity:.1f}%" if epoch >= config['prune_start_epoch'] else "  -  "
        print(f"{epoch:5d} | {train_acc:8.2f}% | {test_acc:7.2f}% | {sparsity_str:>7s} | {status} [{epoch_time:.0f}s]")
    
    return model, best_acc

# ============================================================================
# EXPORTAR A BINARIO
# ============================================================================

def export_to_c_binary(model, filepath='thor_genius_weights.bin'):
    """Exporta pesos a binario para carga desde C"""
    model.eval()
    weights = model.get_int8_weights_all()
    
    with open(filepath, 'wb') as f:
        # Write header
        dims = model.config['hidden_dims'] + [model.config['output_dim']]
        num_hidden = len(model.config['hidden_dims'])
        
        f.write(struct.pack('4s', b'THOR'))  # Magic
        f.write(struct.pack('III', model.config['input_dim'], 
                            dims[-1], num_hidden + 1))
        for d in dims:
            f.write(struct.pack('I', d))
        f.write(struct.pack('I', len(dims)))
        
        # Layer weights
        for i, layer in enumerate(model.layers):
            w_int8, scale = layer.get_int8_weights()
            bias = layer.bias.data.cpu().numpy().astype(np.float32)
            f.write(w_int8.cpu().numpy().tobytes())
            f.write(struct.pack('f', scale))
            f.write(bias.tobytes())
        
        # Output layer (FP32)
        w_out = model.output_layer.weight.data.cpu().numpy().astype(np.float32)
        b_out = model.output_layer.bias.data.cpu().numpy().astype(np.float32)
        f.write(w_out.tobytes())
        f.write(b_out.tobytes())
    
    file_size = os.path.getsize(filepath)
    print(f"\n✅ Pesos exportados: {filepath}")
    print(f"   Tamaño: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    return weights

# ============================================================================
# VALIDACIÓN
# ============================================================================

def validate_quantized(model, test_loader, device='cpu'):
    """Validar accuracy del modelo cuantizado"""
    model.eval()
    correct_fp32 = 0
    correct_int8 = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # FP32
            output_fp32 = model(data)
            _, pred_fp32 = output_fp32.max(1)
            correct_fp32 += pred_fp32.eq(target).sum().item()
            
            # INT8 simulation
            x = data.view(-1, model.config['input_dim'])
            for layer in model.layers:
                w_int8, scale = layer.get_int8_weights()
                w_dequant = w_int8.float() * scale
                x = F.linear(x, w_dequant, layer.bias)
                x = F.relu(x)
            x = model.output_layer(x)
            _, pred_int8 = x.max(1)
            correct_int8 += pred_int8.eq(target).sum().item()
            
            total += target.size(0)
    
    acc_fp32 = 100. * correct_fp32 / total
    acc_int8 = 100. * correct_int8 / total
    acc_drop = acc_fp32 - acc_int8
    
    print(f"\n🔬 VALIDACIÓN DE CUANTIZACIÓN:")
    print(f"   Accuracy FP32: {acc_fp32:.2f}%")
    print(f"   Accuracy INT8: {acc_int8:.2f}%")
    print(f"   Pérdida:       {acc_drop:.2f}%")
    
    return acc_fp32, acc_int8

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("THOR GENIUS - Fast Training\n")
    
    model, accuracy = train()
    
    _, test_loader = get_mnist_loaders(CONFIG['batch_size'])
    acc_fp32, acc_int8 = validate_quantized(model, test_loader)
    
    try:
        export_to_c_binary(model)
    except Exception as e:
        print(f"⚠️ Error export: {e}")
        torch.save(model.state_dict(), 'thor_genius_fast_fallback.pt')
    
    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"   Modelo: 784→{CONFIG['hidden_dims']}→10")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   FP32:   {acc_fp32:.2f}%")
    print(f"   INT8:   {acc_int8:.2f}%")
    if acc_int8 >= 97.0:
        print(f"   ✅ OBJETIVO REVOLUCIONARIO: >97%!")
    elif acc_int8 >= 95.0:
        print(f"   ✅ Buen resultado: >95%")
    else:
        print(f"   ⚠️ Por debajo del objetivo: {acc_int8:.2f}%")
    print(f"\n   Siguiente paso:")
    print(f"   ./bin/thor_v2 thor_genius_weights.bin")
    
    return model

if __name__ == '__main__':
    model = main()
