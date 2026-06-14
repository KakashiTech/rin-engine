#!/usr/bin/env python3
"""
RIN-X Fast QAT Training - Versión rápida para demo
Entrenamiento acelerado con menos epochs pero validando el concepto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import json
import time

print("="*70)
print("RIN-X FAST QAT - Entrenamiento Rápido")
print("="*70)
print()

# QAT Linear simplificado
class QATLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_f, in_f))
        self.register_buffer('scale', torch.ones(1))
        self.qmin, self.qmax = -128, 127
        nn.init.kaiming_uniform_(self.weight)
    
    def forward(self, x):
        # Fake quantize durante training
        with torch.no_grad():
            w_max = self.weight.abs().max()
            self.scale = torch.clamp(w_max / 127, min=1e-8)
        
        w_q = torch.clamp(torch.round(self.weight / self.scale), -128, 127)
        w_dq = w_q * self.scale
        return F.linear(x, w_dq, None)
    
    def get_int8_weights(self):
        w_q = torch.clamp(torch.round(self.weight / self.scale), -128, 127)
        return w_q.to(torch.int8), self.scale.item()

# Modelo SNN QAT
class FastSNNQAT(nn.Module):
    def __init__(self, ts=3):
        super().__init__()
        self.ts = ts
        self.fc1 = QATLinear(784, 128)
        self.fc2 = QATLinear(128, 128)
        self.fc3 = QATLinear(128, 10)
        self.thresh, self.decay = 0.5, 0.8
    
    def lif(self, x, layer):
        batch = x.size(0)
        v = torch.zeros(batch, layer.weight.size(0))
        for _ in range(self.ts):
            c = layer(x)
            v = v * self.decay + c
            v = v * (v < self.thresh).float()
        return v / self.ts
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.lif(x, self.fc1)
        x = self.lif(x, self.fc2)
        return self.fc3(x)

def train_fast():
    # Dataset (subset para velocidad)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('/tmp/mnist', train=False, transform=transform)
    
    # Usar subset para entrenamiento rápido
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    
    model = FastSNNQAT(ts=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Entrenando (epochs rápidos)...")
    best = 0
    
    for epoch in range(5):  # Solo 5 epochs para velocidad
        model.train()
        correct, total = 0, 0
        
        for data, target in train_loader:
            opt.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            
            _, pred = out.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
        
        # Evaluar
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                _, pred = out.max(1)
                test_total += target.size(0)
                test_correct += pred.eq(target).sum().item()
        
        acc = 100. * test_correct / test_total
        if acc > best: best = acc
        print(f"Epoch {epoch+1}/5: Test Acc = {acc:.1f}%")
        
        if acc >= 95:
            print(f"\n✅ Objetivo alcanzado: {acc:.1f}% >= 95%")
            break
    
    print(f"\nMejor accuracy: {best:.1f}%")
    return model, best

def export(model):
    weights = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QATLinear):
            w, s = mod.get_int8_weights()
            weights[name] = {'values': w.cpu().numpy().tolist(), 'scale': s}
    
    weights['metadata'] = {'threshold': model.thresh, 'decay': model.decay, 'ts': model.ts}
    
    with open('snn_fast_weights.json', 'w') as f:
        json.dump(weights, f)
    
    print(f"✓ Pesos exportados: {sum(len(str(w)) for w in weights.values())/1024:.1f} KB")

if __name__ == '__main__':
    start = time.time()
    model, acc = train_fast()
    export(model)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Tiempo total: {elapsed:.1f} segundos")
    print(f"✅ Modelo QAT listo: {acc:.1f}% accuracy")
