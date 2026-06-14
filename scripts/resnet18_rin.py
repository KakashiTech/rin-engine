#!/usr/bin/env python3
"""
ResNet-18 para RIN-X
Adaptación de ResNet-18 con LIF neurons y pruning estructurado
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

sys.path.insert(0, '/home/tuffhk/.local/lib/python3.13/site-packages')

print("="*70)
print("RESNET-18 PARA RIN-X - PORT CON LIF NEURONS")
print("="*70)
print()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    'dataset': 'CIFAR10',  # o CIFAR100
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.1,
    'target_sparsity': 0.50,
    'time_steps': 5,
    'threshold': 0.5,
    'decay': 0.8,
}

# ============================================================================
# BASIC BLOCK CON LIF
# ============================================================================

class LIFActivation(nn.Module):
    """LIF Activation Function"""
    def __init__(self, time_steps=5, threshold=0.5, decay=0.8):
        super().__init__()
        self.time_steps = time_steps
        self.threshold = threshold
        self.decay = decay
        
    def forward(self, x):
        # x: (batch, channels, h, w)
        batch, channels, h, w = x.shape
        v_mem = torch.zeros_like(x)
        spikes = []
        
        for t in range(self.time_steps):
            # Simular input current (usamos x como current constante)
            v_mem = v_mem * self.decay + x
            spike = (v_mem >= self.threshold).float()
            v_mem = v_mem * (1 - spike)
            spikes.append(spike)
        
        # Promedio de spikes
        return torch.stack(spikes, dim=0).mean(dim=0)

class PrunableBasicBlock(nn.Module):
    """BasicBlock de ResNet con pruning y LIF"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, prune_ratio=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = LIFActivation(CONFIG['time_steps'], CONFIG['threshold'], CONFIG['decay'])
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        
        # Mask para pruning
        self.register_buffer('conv1_mask', torch.ones_like(self.conv1.weight))
        self.register_buffer('conv2_mask', torch.ones_like(self.conv2.weight))
        self.prune_ratio = prune_ratio
        
    def forward(self, x):
        # Aplicar pruning
        self.conv1.weight.data *= self.conv1_mask
        self.conv2.weight.data *= self.conv2_mask
        
        out = self.bn1(self.conv1(x))
        out = self.lif1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out
    
    def apply_structured_pruning(self):
        """Aplicar pruning 2:4 estructurado a convoluciones"""
        with torch.no_grad():
            # Pruning conv1
            W = self.conv1.weight.data.abs()
            mask = self._structured_2_4_mask(W)
            self.conv1_mask.data = mask
            
            # Pruning conv2
            W = self.conv2.weight.data.abs()
            mask = self._structured_2_4_mask(W)
            self.conv2_mask.data = mask
    
    def _structured_2_4_mask(self, W):
        """Crear mask 2:4 estructurado"""
        mask = torch.ones_like(W)
        # Aplanar y aplicar 2:4
        W_flat = W.view(-1)
        mask_flat = mask.view(-1)
        
        for i in range(0, W_flat.shape[0] - 3, 4):
            block = W_flat[i:i+4]
            _, indices = torch.topk(block, 2)
            block_mask = torch.zeros(4, device=W.device)
            block_mask[indices] = 1.0
            mask_flat[i:i+4] = block_mask
        
        return mask_flat.view_as(W)
    
    def get_sparsity(self):
        total = self.conv1_mask.numel() + self.conv2_mask.numel()
        active = self.conv1_mask.sum() + self.conv2_mask.sum()
        return 1.0 - (active / total)

# ============================================================================
# RESNET-18 RIN
# ============================================================================

class ResNet18RIN(nn.Module):
    def __init__(self, num_classes=10, prune_ratio=0.5):
        super().__init__()
        self.in_planes = 64
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = LIFActivation(CONFIG['time_steps'], CONFIG['threshold'], CONFIG['decay'])
        
        # Layers
        self.layer1 = self._make_layer(64, 2, stride=1, prune_ratio=prune_ratio)
        self.layer2 = self._make_layer(128, 2, stride=2, prune_ratio=prune_ratio)
        self.layer3 = self._make_layer(256, 2, stride=2, prune_ratio=prune_ratio)
        self.layer4 = self._make_layer(512, 2, stride=2, prune_ratio=prune_ratio)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * PrunableBasicBlock.expansion, num_classes)
    
    def _make_layer(self, planes, num_blocks, stride, prune_ratio):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PrunableBasicBlock(self.in_planes, planes, stride, prune_ratio))
            self.in_planes = planes * PrunableBasicBlock.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        
        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def apply_pruning(self):
        """Aplicar pruning a todas las capas"""
        for module in self.modules():
            if isinstance(module, PrunableBasicBlock):
                module.apply_structured_pruning()
    
    def get_total_sparsity(self):
        total = 0
        active = 0
        for module in self.modules():
            if isinstance(module, PrunableBasicBlock):
                total += module.conv1_mask.numel() + module.conv2_mask.numel()
                active += module.conv1_mask.sum() + module.conv2_mask.sum()
        return 1.0 - (active / total) if total > 0 else 0.0

# ============================================================================
# ENTRENAMIENTO CON PRUNING
# ============================================================================

def train_resnet_rin(model, train_loader, test_loader, device='cpu'):
    """Entrenar ResNet-18 RIN con pruning progresivo"""
    print("="*70)
    print("ENTRENAMIENTO RESNET-18 RIN")
    print("="*70)
    print()
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['lr'], 
                               momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    best_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Aplicar mask a gradientes
            for module in model.modules():
                if isinstance(module, PrunableBasicBlock):
                    if module.conv1.weight.grad is not None:
                        module.conv1.weight.grad *= module.conv1_mask
                    if module.conv2.weight.grad is not None:
                        module.conv2.weight.grad *= module.conv2_mask
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # Evaluar
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().item()
        
        acc = 100. * correct_test / total_test
        if acc > best_acc:
            best_acc = acc
        
        sparsity = model.get_total_sparsity()
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"Train Acc={100.*correct/total:.2f}%, "
              f"Test Acc={acc:.2f}%, "
              f"Sparsity={sparsity:.2%}, "
              f"LR={scheduler.get_last_lr()[0]:.4f}")
        
        # Aplicar pruning gradual
        if epoch == 20:
            print("  Aplicando pruning 25%...")
            CONFIG['target_sparsity'] = 0.25
            model.apply_pruning()
        elif epoch == 35:
            print("  Aplicando pruning 50%...")
            CONFIG['target_sparsity'] = 0.50
            model.apply_pruning()
        
        scheduler.step()
    
    print(f"\nBest accuracy: {best_acc:.2f}%")
    return model, best_acc

# ============================================================================
# EXPORTAR PARA C
# ============================================================================

def export_resnet_for_c(model, filepath='resnet18_rin.npz'):
    """Exportar ResNet-18 RIN para kernel C"""
    print("\nExportando ResNet-18 RIN...")
    
    weights = {}
    
    # Stem
    weights['stem_conv'] = model.conv1.weight.data.cpu().numpy()
    weights['stem_bn'] = model.bn1.weight.data.cpu().numpy()
    weights['stem_bn_bias'] = model.bn1.bias.data.cpu().numpy()
    
    # Layers
    layer_idx = 0
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            prefix = f'block_{layer_idx}'
            weights[f'{prefix}_conv1'] = block.conv1.weight.data.cpu().numpy()
            weights[f'{prefix}_conv1_mask'] = block.conv1_mask.cpu().numpy()
            weights[f'{prefix}_bn1'] = block.bn1.weight.data.cpu().numpy()
            weights[f'{prefix}_bn1_bias'] = block.bn1.bias.data.cpu().numpy()
            
            weights[f'{prefix}_conv2'] = block.conv2.weight.data.cpu().numpy()
            weights[f'{prefix}_conv2_mask'] = block.conv2_mask.cpu().numpy()
            weights[f'{prefix}_bn2'] = block.bn2.weight.data.cpu().numpy()
            weights[f'{prefix}_bn2_bias'] = block.bn2.bias.data.cpu().numpy()
            
            if len(block.shortcut) > 0:
                weights[f'{prefix}_shortcut'] = block.shortcut[0].weight.data.cpu().numpy()
            
            layer_idx += 1
    
    # FC
    weights['fc'] = model.fc.weight.data.cpu().numpy()
    weights['fc_bias'] = model.fc.bias.data.cpu().numpy() if model.fc.bias is not None else np.zeros(model.fc.out_features)
    
    # Metadata
    weights['metadata'] = json.dumps({
        'sparsity': model.get_total_sparsity(),
        'num_classes': model.fc.out_features,
        'time_steps': CONFIG['time_steps'],
        'threshold': CONFIG['threshold'],
        'decay': CONFIG['decay'],
    })
    
    np.savez(filepath, **weights)
    print(f"✓ Exportado: {filepath}")
    print(f"  Sparsity: {model.get_total_sparsity():.2%}")
    
    return filepath

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Cargar CIFAR-10
    print("Cargando CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG['batch_size'], 
                                              shuffle=True, num_workers=2)
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    print(f"  Train: {len(trainset)} samples")
    print(f"  Test: {len(testset)} samples")
    print()
    
    # Crear modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = ResNet18RIN(num_classes=10, prune_ratio=CONFIG['target_sparsity'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")
    print()
    
    # Entrenar
    model, best_acc = train_resnet_rin(model, trainloader, testloader, device)
    
    # Exportar
    export_resnet_for_c(model)
    
    print("\n" + "="*70)
    print("RESNET-18 RIN COMPLETADO")
    print("="*70)
    
    return model, best_acc

if __name__ == '__main__':
    model, acc = main()
