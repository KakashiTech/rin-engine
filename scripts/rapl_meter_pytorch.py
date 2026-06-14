#!/usr/bin/env python3
"""
RAPLEnergyMeter - Medición de energía para PyTorch con RAPL

Este script se ejecuta con sudo pero usa el Python del usuario donde torch está instalado.
Uso: sudo /usr/bin/python3 rapl_meter_pytorch.py
"""

import sys
import os
import time
import subprocess

# Configuración
NUM_TOKENS = 100
NUM_RUNS = 30

# Añadir path del usuario para encontrar torch
user_site = os.path.expanduser("~/.local/lib/python3.12/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch no disponible")
    print("Paths:", sys.path)
    sys.exit(1)

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU threads: {torch.get_num_threads()}")
print()

# Función para leer RAPL
def read_rapl():
    try:
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as f:
            return int(f.read().strip())
    except:
        return 0

# Modelo PyTorch (igual que baseline)
MODEL_DIM = 512
NUM_LAYERS = 8

class DenseLIFLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
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
        results = []
        for t in range(num_tokens):
            x = self.embedding + torch.randn(1, MODEL_DIM) * 0.1
            for layer in self.layers:
                x = layer(x)
            results.append(x)
        return torch.stack(results)

# Benchmark
results = []

print(f"Ejecutando {NUM_RUNS} corridas de {NUM_TOKENS} tokens...")
print()

for run in range(NUM_RUNS):
    model = PyTorchModel()
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(10)
    
    # Medición
    energy_start = read_rapl()
    time_start = time.perf_counter()
    
    with torch.no_grad():
        output = model(NUM_TOKENS)
    
    time_end = time.perf_counter()
    energy_end = read_rapl()
    
    # Calcular
    energy_j = (energy_end - energy_start) / 1_000_000
    elapsed = time_end - time_start
    j_per_token = energy_j / NUM_TOKENS
    
    results.append({
        'run': run + 1,
        'energy_j': energy_j,
        'time_sec': elapsed,
        'j_per_token': j_per_token,
        'tokens_per_sec': NUM_TOKENS / elapsed
    })
    
    print(f"Run {run+1:2d}: {energy_j:7.3f}J | {elapsed:.3f}s | {j_per_token:.5f} J/token")

# Estadísticas
import statistics

energies = [r['energy_j'] for r in results]
times = [r['time_sec'] for r in results]
j_per_tokens = [r['j_per_token'] for r in results]

print()
print("=" * 60)
print("ESTADÍSTICAS (30 corridas)")
print("=" * 60)
print(f"Energía:      {statistics.mean(energies):.3f} ± {statistics.stdev(energies):.3f} J")
print(f"Tiempo:       {statistics.mean(times):.3f} ± {statistics.stdev(times):.3f} s")
print(f"J/token:      {statistics.mean(j_per_tokens):.5f} ± {statistics.stdev(j_per_tokens):.5f}")
print(f"Tokens/sec:   {NUM_TOKENS/statistics.mean(times):.1f}")
print()

# Guardar resultados
with open('pytorch_30runs_energy.csv', 'w') as f:
    f.write("run,energy_j,time_sec,j_per_token,tokens_per_sec\n")
    for r in results:
        f.write(f"{r['run']},{r['energy_j']:.6f},{r['time_sec']:.6f},{r['j_per_token']:.8f},{r['tokens_per_sec']:.2f}\n")

print("Resultados guardados en: pytorch_30runs_energy.csv")
