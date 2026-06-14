#!/usr/bin/env python3
"""
Tiny RIN Training - Entrenar modelo pequeño en PC local
MNIST con modelo de 64 dims, 2 capas LIF
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json

sys.path.insert(0, '/home/tuffhk/.local/lib/python3.13/site-packages')

print("="*70)
print("TINY RIN TRAINING - MNIST en CPU")
print("="*70)

# Configuración pequeña para CPU
INPUT_DIM = 28*28  # MNIST
HIDDEN_DIM = 64    # Pequeño para CPU rápido
OUTPUT_DIM = 10    # 10 clases MNIST
NUM_LAYERS = 2
EPOCHS = 3         # Pocas épocas para rapidez
BATCH_SIZE = 64

print(f"Config: {INPUT_DIM} → {HIDDEN_DIM} → {HIDDEN_DIM} → {OUTPUT_DIM}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
print()

# Cargar MNIST
print("[1/4] Cargando MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('/tmp/mnist', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"  Train: {len(train_dataset)} samples")
print(f"  Test: {len(test_dataset)} samples")
print()

# Modelo LIF para MNIST
class TinyLIFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, time_steps=5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.threshold = 0.5
        self.decay = 0.8
        self.time_steps = time_steps
        
        # Inicialización
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        # x: (batch, in_dim)
        batch_size = x.size(0)
        v_mem = torch.zeros(batch_size, self.linear.out_features)
        spikes = []
        
        for t in range(self.time_steps):
            current = self.linear(x)
            v_mem = v_mem * self.decay + current
            spike = (v_mem >= self.threshold).float()
            v_mem = v_mem * (1 - spike)
            spikes.append(spike)
        
        # Promedio de spikes a lo largo del tiempo
        return torch.stack(spikes, dim=0).mean(dim=0)

class TinyRIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TinyLIFLayer(INPUT_DIM, HIDDEN_DIM)
        self.layer2 = TinyLIFLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.readout = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        
    def forward(self, x):
        x = x.view(-1, INPUT_DIM)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.readout(x)
        return x

model = TinyRIN()
print("[2/4] Entrenando...")
print(f"  Parámetros: {sum(p.numel() for p in model.parameters())}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.1f}%")

# Evaluar
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

test_acc = 100. * correct / total
print(f"\n  Test Accuracy: {test_acc:.2f}%")
print()

# Guardar pesos para RIN-X
print("[3/4] Exportando pesos a RIN-X...")

weights = {
    'layer1_w': model.layer1.linear.weight.detach().numpy().astype(np.float32),
    'layer2_w': model.layer2.linear.weight.detach().numpy().astype(np.float32),
    'readout_w': model.readout.weight.detach().numpy().astype(np.float32),
    'threshold': 0.5,
    'decay': 0.8,
    'time_steps': 5,
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'output_dim': OUTPUT_DIM,
    'test_accuracy': test_acc
}

# Guardar como JSON
weights_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in weights.items()}
with open('tiny_rin_weights.json', 'w') as f:
    json.dump(weights_dict, f)

# Guardar como binario NPZ
np.savez('tiny_rin_weights.npz',
         layer1_w=weights['layer1_w'],
         layer2_w=weights['layer2_w'],
         readout_w=weights['readout_w'])

print("  Guardado: tiny_rin_weights.json (texto)")
print("  Guardado: tiny_rin_weights.npz (binario)")
print()

# Guardar algunos samples para validación
print("[4/4] Guardando samples de validación...")
test_samples = []
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        if i >= 2: break  # Solo 2000 samples
        output = model(data)
        pred = output.argmax(dim=1)
        for j in range(min(10, len(data))):  # 10 samples por batch
            test_samples.append({
                'input': data[j].numpy().tolist(),
                'target': int(target[j]),
                'prediction': int(pred[j]),
                'output_logits': output[j].numpy().tolist()
            })

with open('tiny_rin_test_samples.json', 'w') as f:
    json.dump(test_samples[:100], f)  # 100 samples

print(f"  Guardados: {len(test_samples[:100])} samples en tiny_rin_test_samples.json")
print()

print("="*70)
print("ENTRENAMIENTO COMPLETADO")
print("="*70)
print(f"Test Accuracy: {test_acc:.2f}%")
print()
print("Archivos generados:")
print("  - tiny_rin_weights.json (pesos entrenados)")
print("  - tiny_rin_test_samples.json (samples de validación)")
print()
print("Siguiente paso: Compilar RIN-X con estos pesos y comparar salidas")
print("="*70)
