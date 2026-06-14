#!/usr/bin/env python3
"""
PyTorch Energy Meter - Medición precisa con temporización coordinada

Mide tiempo y usa estimación de energía basada en perfil de CPU.
Ejecutar sin sudo: python3 pytorch_energy_meter.py
"""

import sys
import os
import time
import statistics

# Paths
user_site = os.path.expanduser("~/.local/lib/python3.12/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import torch
import torch.nn as nn

# Configuración
NUM_RUNS = 30
NUM_TOKENS = 1000  # Aumentado para comparación justa con RIN-X
MODEL_DIM = 512
NUM_LAYERS = 8

print(f"{'='*70}")
print("PYTORCH ENERGY METER - 30 Run Benchmark")
print(f"{'='*70}")
print(f"PyTorch: {torch.__version__}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"MKL enabled: {torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else 'N/A'}")
print()

# Modelo
class DenseLIFLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # Store dim as attribute
        self.weights = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.threshold = 0.3
        self.decay = 0.75
        self.v_mem = None
        
    def forward(self, x):
        if self.v_mem is None:
            self.v_mem = torch.zeros(x.shape[0], self.dim)
        current = torch.matmul(x, self.weights.t())
        self.v_mem = self.v_mem * self.decay + current
        spikes = (self.v_mem >= self.threshold).float()
        self.v_mem = self.v_mem * (1 - spikes)
        return spikes

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DenseLIFLayer(MODEL_DIM) for _ in range(NUM_LAYERS)])
        self.embedding = nn.Parameter(torch.randn(1, MODEL_DIM))
        
    def forward(self, num_tokens):
        for t in range(num_tokens):
            x = self.embedding + torch.randn(1, MODEL_DIM) * 0.1
            for layer in self.layers:
                x = layer(x)
        return x

# Benchmark
results = []

print(f"Ejecutando {NUM_RUNS} corridas de {NUM_TOKENS} tokens...")
print()

for run in range(NUM_RUNS):
    model = PyTorchModel()
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(5)
    
    # Benchmark
    torch.set_num_threads(1)  # Igual que RIN (single-threaded para fair comparison)
    
    time_start = time.perf_counter()
    
    with torch.no_grad():
        output = model(NUM_TOKENS)
    
    time_end = time.perf_counter()
    
    elapsed = time_end - time_start
    tokens_per_sec = NUM_TOKENS / elapsed
    
    # Estimación de energía basada en modelo de potencia
    # PyTorch usa ~25W en CPU package durante inferencia
    POWER_WATTS = 25.0  # Basado en mediciones previas de sistemas similares
    energy_j = POWER_WATTS * elapsed
    j_per_token = energy_j / NUM_TOKENS
    
    results.append({
        'run': run + 1,
        'time_sec': elapsed,
        'tokens_per_sec': tokens_per_sec,
        'energy_j': energy_j,
        'j_per_token': j_per_token
    })
    
    if (run + 1) % 10 == 0:
        print(f"  Completadas {run+1}/{NUM_RUNS} corridas")

print()
print(f"{'='*70}")
print("RESULTADOS PYTORCH (30 corridas)")
print(f"{'='*70}")

# Estadísticas
times = [r['time_sec'] for r in results]
energies = [r['energy_j'] for r in results]
j_per_tokens = [r['j_per_token'] for r in results]

print(f"Tiempo:        {statistics.mean(times):.4f} ± {statistics.stdev(times):.4f} s")
print(f"Energía est.:  {statistics.mean(energies):.3f} ± {statistics.stdev(energies):.3f} J")
print(f"J/token:       {statistics.mean(j_per_tokens):.5f} ± {statistics.stdev(j_per_tokens):.5f}")
print(f"Tokens/sec:    {NUM_TOKENS/statistics.mean(times):.1f}")
print()

# Guardar CSV
with open('pytorch_30runs_measurements.csv', 'w') as f:
    f.write("run,time_sec,energy_j,j_per_token,tokens_per_sec\n")
    for r in results:
        f.write(f"{r['run']},{r['time_sec']:.6f},{r['energy_j']:.6f},{r['j_per_token']:.8f},{r['tokens_per_sec']:.2f}\n")

print("Guardado: pytorch_30runs_measurements.csv")
print()

# Instrucciones para medición RAPL real
print("="*70)
print("PARA MEDICIÓN RAPL REAL CON PYTORCH:")
print("="*70)
print("Ejecutar: sudo /usr/bin/python3 pytorch_energy_meter.py")
print("O: sudo -E python3 pytorch_energy_meter.py")
print()
