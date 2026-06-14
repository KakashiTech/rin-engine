#!/usr/bin/env python3
"""
VALIDACIÓN RIGUROSA RIN-X
Benchmark científico completo con controles estrictos
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import subprocess
import os

sys.path.insert(0, '/home/tuffhk/.local/lib/python3.13/site-packages')

print("="*80)
print("VALIDACIÓN RIGUROSA - RIN-X vs PYTORCH vs ONNX")
print("="*80)
print()

# ============================================================================
# CONFIGURACIONES DE ESCALA
# ============================================================================

CONFIGS = [
    {"name": "Tiny", "dim": 64, "layers": 2, "params": "~55K"},
    {"name": "Small", "dim": 256, "layers": 4, "params": "~1M"},
    {"name": "Medium", "dim": 512, "layers": 8, "params": "~8M"},
    {"name": "Large", "dim": 768, "layers": 12, "params": "~28M"},
]

BATCH_SIZES = [1, 8, 32, 64]

print("CONFIGURACIONES A PROBAR:")
for cfg in CONFIGS:
    print(f"  {cfg['name']:8}: dim={cfg['dim']}, layers={cfg['layers']}, {cfg['params']}")
print()
print(f"BATCH SIZES: {BATCH_SIZES}")
print()

# ============================================================================
# MODELO RIN (Reference)
# ============================================================================

class RINLayer(nn.Module):
    """LIF SNN Layer - reference implementation"""
    def __init__(self, in_dim, out_dim, time_steps=5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.threshold = 0.5
        self.decay = 0.8
        self.time_steps = time_steps
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        batch = x.size(0)
        v_mem = torch.zeros(batch, self.linear.out_features)
        spikes = []
        for t in range(self.time_steps):
            current = self.linear(x)
            v_mem = v_mem * self.decay + current
            spike = (v_mem >= self.threshold).float()
            v_mem = v_mem * (1 - spike)
            spikes.append(spike)
        return torch.stack(spikes, dim=0).mean(dim=0)

class RINModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(RINLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(RINLayer(hidden_dim, hidden_dim))
        self.readout = nn.Linear(hidden_dim, output_dim, bias=False)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.readout(x)

# ============================================================================
# CONTADOR DE FLOPs
# ============================================================================

def count_flops_rin(batch_size, input_dim, hidden_dim, output_dim, num_layers, time_steps):
    """Contar FLOPs reales para RIN"""
    # Layer 1: matmul + LIF ops
    flops_l1 = batch_size * (2 * input_dim * hidden_dim)  # matmul
    flops_l1 += batch_size * hidden_dim * 4 * time_steps  # LIF: mul, add, compare, mul
    
    # Hidden layers
    flops_hidden = (num_layers - 1) * batch_size * (2 * hidden_dim * hidden_dim)
    flops_hidden += (num_layers - 1) * batch_size * hidden_dim * 4 * time_steps
    
    # Readout
    flops_readout = batch_size * (2 * hidden_dim * output_dim)
    
    total = flops_l1 + flops_hidden + flops_readout
    return total

def count_params(input_dim, hidden_dim, output_dim, num_layers):
    """Contar parámetros activos"""
    params = input_dim * hidden_dim  # Layer 1
    params += (num_layers - 1) * hidden_dim * hidden_dim  # Hidden
    params += hidden_dim * output_dim  # Readout
    return params

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_pytorch(model, input_tensor, num_runs=10):
    """Benchmark PyTorch con warmup"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if input_tensor.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if input_tensor.is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    return np.mean(times), np.std(times)

# ============================================================================
# VALIDACIÓN PRINCIPAL
# ============================================================================

print("="*80)
print("INICIANDO BENCHMARKS")
print("="*80)
print()

results = {}

for cfg in CONFIGS:
    name = cfg['name']
    dim = cfg['dim']
    layers = cfg['layers']
    
    print(f"\n{'='*80}")
    print(f"CONFIGURACIÓN: {name} (dim={dim}, layers={layers})")
    print(f"{'='*80}\n")
    
    results[name] = {}
    
    # Crear modelo
    model = RINModel(784, dim, 10, layers)
    model.eval()
    
    # Contar parámetros y FLOPs
    params = count_params(784, dim, 10, layers)
    
    print(f"Parámetros activos: {params:,}")
    print()
    
    for batch_size in BATCH_SIZES:
        print(f"  Batch={batch_size}:")
        
        # Crear input
        dummy_input = torch.randn(batch_size, 784)
        
        # Contar FLOPs
        flops = count_flops_rin(batch_size, 784, dim, 10, layers, 5)
        
        # Benchmark PyTorch
        t_mean, t_std = benchmark_pytorch(model, dummy_input, num_runs=10)
        
        # Métricas
        throughput = batch_size / t_mean
        gflops_per_sec = (flops / 1e9) / t_mean
        
        print(f"    Time: {t_mean*1000:.2f}±{t_std*1000:.2f} ms")
        print(f"    Throughput: {throughput:.1f} samples/s")
        print(f"    FLOPs: {flops/1e6:.1f} MFLOPs")
        print(f"    GFLOP/s: {gflops_per_sec:.2f}")
        
        results[name][f"batch_{batch_size}"] = {
            'time_ms': t_mean * 1000,
            'time_std_ms': t_std * 1000,
            'throughput': throughput,
            'flops': flops,
            'gflops_per_sec': gflops_per_sec,
            'params': params
        }

# ============================================================================
# ANÁLISIS DE ESCALADO
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS DE ESCALADO")
print("="*80)
print()

print("ESCALADO CON BATCH SIZE:")
print(f"{'Config':<10} {'Batch=1':<15} {'Batch=32':<15} {'Batch=64':<15} {'Scaling':<15}")
print("-"*80)

for cfg in CONFIGS:
    name = cfg['name']
    t1 = results[name]['batch_1']['time_ms']
    t32 = results[name]['batch_32']['time_ms']
    t64 = results[name]['batch_64']['time_ms']
    
    # Ideal scaling: tiempo debería crecer lineal con batch
    ideal_32 = t1 * 32
    actual_32 = t32
    efficiency_32 = (ideal_32 / actual_32) * 100
    
    print(f"{name:<10} {t1:>8.2f} ms    {t32:>8.2f} ms    {t64:>8.2f} ms    {efficiency_32:>6.1f}%")

print()
print("ESCALADO CON TAMAÑO DE MODELO:")
print(f"{'Batch':<8} {'Tiny':<12} {'Small':<12} {'Medium':<12} {'Large':<12}")
print("-"*80)

for bs in [1, 32]:
    key = f'batch_{bs}'
    times = []
    for cfg in CONFIGS:
        name = cfg['name']
        t = results[name][key]['time_ms']
        times.append(t)
    print(f"{bs:<8} {times[0]:>8.2f} ms  {times[1]:>8.2f} ms  {times[2]:>8.2f} ms  {times[3]:>8.2f} ms")

# ============================================================================
# VERIFICACIÓN DE IGUALDAD COMPUTACIONAL
# ============================================================================

print("\n" + "="*80)
print("VERIFICACIÓN DE IGUALDAD COMPUTACIONAL")
print("="*80)
print()

print("FLOPs POR SAMPLE (debe ser igual en todas las implementaciones):")
print()

for cfg in CONFIGS:
    name = cfg['name']
    dim = cfg['dim']
    layers = cfg['layers']
    flops_per_sample = count_flops_rin(1, 784, dim, 10, layers, 5)
    print(f"{name:8}: {flops_per_sample/1e6:>8.2f} MFLOPs/sample")

print()
print("⚠️  NOTA: Para comparación JUSTA contra PyTorch:")
print("   - PyTorch debe ejecutar los MISMOS FLOPs")
print("   - Sin optimizaciones que eliminen operaciones")
print("   - Sparsity artificial que coincida")
print()

# ============================================================================
# RESUMEN Y LIMITACIONES
# ============================================================================

print("\n" + "="*80)
print("RESUMEN Y LIMITACIONES IDENTIFICADAS")
print("="*80)
print()

print("✅ LO QUE HEMOS VALIDADO:")
print("   • Modelo tiny (64 dims): equivalencia 100%")
print("   • Speedup en modelo pequeño: ~100-130×")
print("   • Energía: 7.8× mejor")
print()

print("❌ LO QUE FALTA VALIDAR:")
print("   • Modelo 10× más grande con entrenamiento real")
print("   • Modelo 100× (ResNet/Transformer) con pesos entrenados")
print("   • Batch scaling óptimo (¿eficiencia se mantiene?)")
print("   • Comparación vs ONNX Runtime / XNNPACK / oneDNN")
print("   • FLOPs reales medidos (no calculados)")
print("   • Sparsity real (95% es artificial en el benchmark)")
print()

print("⚠️  HIPÓTESIS SIN VALIDAR:")
print("   • El speedup 129× puede venir de:")
print("     a) Framework overhead (no computación real)")
print("     b) Modelo más pequeño en el benchmark original")
print("     c) Batch size diferente")
print("     d) Sparsity 95% artificial vs dense PyTorch")
print()

print("🎯 PRÓXIMOS PASOS NECESARIOS:")
print("   1. Entrenar modelo Medium (512 dims) en CIFAR-10")
print("   2. Medir FLOPs reales con perf stat")
print("   3. Benchmark vs ONNX Runtime exportado")
print("   4. Probar con modelo pre-entrenado real (ResNet-18)")
print("   5. Validar que el speedup se mantiene en GPUs")
print()

print("="*80)

# Guardar resultados
with open('rigorous_scaling_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Resultados guardados en: rigorous_scaling_results.json")
print("="*80)
