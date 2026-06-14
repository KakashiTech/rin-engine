#!/usr/bin/env python3
"""
Exportar pesos con BatchNorm fusionado para kernel C
W' = W * γ/σ,  b' = β - μ*γ/σ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys

sys.path.insert(0, '/home/tuffhk/Work/THOR')

print("="*70)
print("EXPORTAR PESOS CON BATCHNORM FUSIONADO")
print("="*70)
print()

# Cargar modelo entrenado
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10, bias=True)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# Cargar
model = SimpleMLP()
try:
    model.load_state_dict(torch.load('best_model.pth'))
    print("✓ Modelo cargado desde best_model.pth")
except:
    print("⚠️ No se pudo cargar best_model.pth")
    exit(1)

model.eval()

print("\nFUSIONANDO BATCHNORM...")
print()

# Función para fusionar BN con Linear
def fuse_bn_with_linear(linear, bn):
    """
    Fusión: W' = W * γ/σ, b' = b + (β - μ*γ/σ)
    """
    # Extraer parámetros
    weight = linear.weight.data.clone()
    bias = linear.bias.data.clone() if linear.bias is not None else torch.zeros(linear.out_features)
    
    # Parámetros BN
    gamma = bn.weight.data.clone()  # γ
    beta = bn.bias.data.clone()     # β
    mu = bn.running_mean.data.clone()  # μ
    var = bn.running_var.data.clone()  # σ²
    eps = bn.eps
    
    # σ = sqrt(var + eps)
    sigma = torch.sqrt(var + eps)
    
    # W' = W * γ/σ (cada fila de pesos se escala)
    # weight shape: [out_features, in_features]
    # gamma/sigma shape: [out_features]
    scale = gamma / sigma
    fused_weight = weight * scale.unsqueeze(1)
    
    # b' = b + (β - μ*γ/σ)
    fused_bias = bias + (beta - mu * scale)
    
    return fused_weight, fused_bias

# Fusionar capas
w1_fused, b1_fused = fuse_bn_with_linear(model.fc1, model.bn1)
w2_fused, b2_fused = fuse_bn_with_linear(model.fc2, model.bn2)

# Capa 3 no tiene BN
w3 = model.fc3.weight.data.clone()
b3 = model.fc3.bias.data.clone() if model.fc3.bias is not None else torch.zeros(10)

print("  Capa 1: W shape", w1_fused.shape, "| B shape", b1_fused.shape)
print("  Capa 2: W shape", w2_fused.shape, "| B shape", b2_fused.shape)
print("  Capa 3: W shape", w3.shape, "| B shape", b3.shape)

# Quantizar a INT8
def quantize_tensor(t):
    """Quantizar tensor a INT8"""
    t_max = t.abs().max()
    scale = t_max / 127.0 if t_max > 0 else 1.0
    t_q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
    return t_q, float(scale)

w1_q, s1 = quantize_tensor(w1_fused)
w2_q, s2 = quantize_tensor(w2_fused)
w3_q, s3 = quantize_tensor(w3)

# Bias se mantiene en FP32 (escalado)
b1_scaled = b1_fused / s1
b2_scaled = b2_fused / s2
b3_scaled = b3 / s3

print(f"\n  Scales de quantización:")
print(f"    Capa 1: {s1:.6f}")
print(f"    Capa 2: {s2:.6f}")
print(f"    Capa 3: {s3:.6f}")

# Exportar a JSON
weights_data = {
    'w1': {
        'values': w1_q.cpu().numpy().tolist(),
        'scale': s1,
        'bias': b1_scaled.cpu().numpy().tolist(),
        'shape': list(w1_q.shape)
    },
    'w2': {
        'values': w2_q.cpu().numpy().tolist(),
        'scale': s2,
        'bias': b2_scaled.cpu().numpy().tolist(),
        'shape': list(w2_q.shape)
    },
    'w3': {
        'values': w3_q.cpu().numpy().tolist(),
        'scale': s3,
        'bias': b3_scaled.cpu().numpy().tolist(),
        'shape': list(w3_q.shape)
    },
    'metadata': {
        'architecture': '784->256->256->10',
        'batchnorm_fused': True,
        'quantization': 'INT8 post-training'
    }
}

with open('mlp_fused_weights.json', 'w') as f:
    json.dump(weights_data, f)

print(f"\n✅ Pesos exportados: mlp_fused_weights.json")

# Validar que el modelo fusionado da igual resultado
print("\nVALIDANDO FUSIÓN...")
model_fused = SimpleMLP()
model_fused.fc1.weight.data = w1_fused
model_fused.fc1.bias.data = b1_fused
model_fused.fc2.weight.data = w2_fused
model_fused.fc2.bias.data = b2_fused
model_fused.fc3.weight.data = w3
model_fused.fc3.bias.data = b3
# Desactivar BN (eval mode pasa datos sin cambios)
model_fused.bn1.eval()
model_fused.bn2.eval()
model_fused.fc1.eval()
model_fused.fc2.eval()
model_fused.fc3.eval()

dummy = torch.randn(1, 784)

with torch.no_grad():
    out_original = model(dummy)
    out_fused = model_fused(dummy)

diff = (out_original - out_fused).abs().max().item()
print(f"  Diferencia máxima: {diff:.6f}")

if diff < 1e-5:
    print("  ✅ Fusión correcta (diferencia < 1e-5)")
else:
    print(f"  ⚠️ Diferencia detectada: {diff}")

print("\n" + "="*70)
print("Listo para usar en kernel C con BatchNorm fusionado")
print("="*70)
