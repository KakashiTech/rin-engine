#!/usr/bin/env python3
"""
PyTorch Baseline Real para Comparación RIN

Implementa el mismo modelo (8 capas, dim 512) en PyTorch para comparación
justa de consumo energético.

Uso con medición de energía:
    sudo perf stat -d python3 pytorch_baseline.py
    
O con RAPL:
    # Leer energía antes
    python3 pytorch_baseline.py
    # Leer energía después
"""

import torch
import torch.nn as nn
import time
import sys

# Configuración idéntica a RIN
MODEL_DIM = 512
NUM_LAYERS = 8
NUM_TOKENS = 100

class DenseLIFLayer(nn.Module):
    """
    Simula capa LIF pero con operaciones densas PyTorch
    (esto es lo que hace PyTorch por defecto - no optimizado)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Pesos densos (no sparsity)
        self.weights = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.threshold = 0.3
        self.decay = 0.75
        self.v_mem = None
        
    def forward(self, x):
        if self.v_mem is None:
            self.v_mem = torch.zeros(x.shape[0], self.dim)
        
        # Multiplicación densa (FP32)
        current = torch.matmul(x, self.weights.t())
        
        # LIF dynamics con FP32
        self.v_mem = self.v_mem * self.decay + current
        
        # Spike generation
        spikes = (self.v_mem >= self.threshold).float()
        self.v_mem = self.v_mem * (1 - spikes)  # Reset donde hay spike
        
        return spikes

class PyTorchBaselineModel(nn.Module):
    """Modelo baseline equivalente al RIN"""
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLIFLayer(dim) for _ in range(num_layers)
        ])
        self.embedding = nn.Parameter(torch.randn(1, dim))
        
    def forward(self, num_tokens):
        results = []
        for t in range(num_tokens):
            # Generar embedding para este token
            x = self.embedding + torch.randn(1, MODEL_DIM) * 0.1
            
            # Forward through layers
            for layer in self.layers:
                x = layer(x)
            
            results.append(x)
        
        return torch.stack(results)

def benchmark():
    print(f"PyTorch Baseline - {NUM_LAYERS} layers, dim={MODEL_DIM}, {NUM_TOKENS} tokens")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print()
    
    # Crear modelo
    model = PyTorchBaselineModel(MODEL_DIM, NUM_LAYERS)
    model.eval()
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            model(10)
    
    # Benchmark real
    print(f"Running benchmark ({NUM_TOKENS} tokens)...")
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        output = model(NUM_TOKENS)
    
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.3f} sec")
    print(f"  Tokens/sec: {NUM_TOKENS/elapsed:.2f}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    
    return elapsed

if __name__ == "__main__":
    # Permitir cambiar número de tokens
    num_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_TOKENS
    NUM_TOKENS = num_tokens
    
    benchmark()
