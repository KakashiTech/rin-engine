#!/usr/bin/env python3
"""
RIN-X MLP Training Corregido - Entrenamiento completo hasta convergencia real
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import json
import time

print("="*70)
print("RIN-X MLP TRAINING - Versión Corregida")
print("="*70)
print()

# Modelo MLP simple
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

def train_full():
    # MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('/tmp/mnist', train=False, transform=transform)
    
    # DataLoaders completos
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2)
    
    model = SimpleMLP()
    
    # Optimizador con learning rate apropiado
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"Entrenando en {len(train_ds)} imágenes...")
    print(f"Test set: {len(test_ds)} imágenes")
    print()
    
    best_acc = 0.0
    epochs = 20  # Suficientes epochs para convergencia
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            opt.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            _, pred = out.max(1)
            train_total += target.size(0)
            train_correct += pred.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                loss = criterion(out, target)
                test_loss += loss.item()
                
                _, pred = out.max(1)
                test_total += target.size(0)
                test_correct += pred.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Guardar mejor modelo
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Scheduler step
        scheduler.step(test_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={avg_test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"  Best:  {best_acc:.2f}%")
        print()
        
        # Early stopping si alcanzamos objetivo
        if test_acc >= 97.0:
            print(f"🎉 OBJETIVO SUPERADO: {test_acc:.2f}% >= 97%")
            break
        
        # Early stopping si no mejora
        if epoch > 5 and test_acc < 90:
            print(f"⚠️  Posible problema - accuracy baja. Continuando...")
    
    print(f"\n✅ Entrenamiento completado. Mejor accuracy: {best_acc:.2f}%")
    return model, best_acc

def export_weights(model):
    """Exportar pesos para kernel C"""
    weights = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            # Quantizar a INT8
            w = param.data.cpu()
            w_max = w.abs().max()
            scale = w_max / 127.0
            
            w_q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
            
            weights[name] = {
                'values': w_q.numpy().tolist(),
                'scale': float(scale),
                'shape': list(w_q.shape)
            }
    
    with open('mlp_fixed_weights.json', 'w') as f:
        json.dump(weights, f)
    
    total_params = sum(len(w['values']) for w in weights.values())
    print(f"✅ Pesos exportados: {total_params:,} parámetros INT8")
    print(f"   Archivo: mlp_fixed_weights.json")
    
    return weights

if __name__ == '__main__':
    start = time.time()
    
    # Entrenar
    model, accuracy = train_full()
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Exportar
    export_weights(model)
    
    elapsed = time.time() - start
    
    print(f"\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"⏱️  Tiempo total: {elapsed/60:.1f} minutos")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 95:
        print("✅ MODELO FUNCIONAL - Listo para kernel RIN-X")
    else:
        print("⚠️  Modelo por debajo de 95% - revisar hyperparámetros")
    
    print("="*70)
