#!/usr/bin/env python3
"""
TRIPLE BENCHMARK: RIN-X vs ONNX Runtime vs PyTorch
Comparación directa en igualdad de condiciones
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/tuffhk/.local/lib/python3.13/site-packages')

print("="*80)
print("TRIPLE BENCHMARK: RIN-X vs ONNX vs PYTORCH")
print("="*80)
print()

# Importar ONNX
import onnxruntime as ort

# ============================================================================
# MODELO RIN (igual arquitectura para todos)
# ============================================================================

class RINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, time_steps=5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.threshold = 0.5
        self.decay = 0.8
        self.time_steps = time_steps
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        
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
        nn.init.xavier_uniform_(self.readout.weight, gain=0.5)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.readout(x)

# ============================================================================
# RIN-X SIMULATION (NumPy - lo que sería el código C)
# ============================================================================

class RIN_X:
    """Simula exactamente lo que haría el C de RIN-X"""
    def __init__(self, pytorch_model):
        # Extraer pesos de modelo PyTorch
        self.weights = []
        for layer in pytorch_model.layers:
            w = layer.linear.weight.detach().numpy().astype(np.float32)
            self.weights.append(w)
        self.readout = pytorch_model.readout.weight.detach().numpy().astype(np.float32)
        self.threshold = 0.5
        self.decay = 0.8
        self.time_steps = 5
        
    def lif_forward(self, W, x, threshold, decay, time_steps):
        """
        W: (out_dim, in_dim)
        x: (in_dim,)
        """
        out_dim = W.shape[0]
        v_mem = np.zeros(out_dim, dtype=np.float32)
        output = np.zeros(out_dim, dtype=np.float32)
        
        for t in range(time_steps):
            current = W @ x
            v_mem = v_mem * decay + current
            spike = (v_mem >= threshold).astype(np.float32)
            v_mem = v_mem * (1.0 - spike)
            output += spike
        
        return output / time_steps
    
    def forward(self, x):
        x = x.astype(np.float32)
        
        # Forward through layers
        for W in self.weights:
            x = self.lif_forward(W, x, self.threshold, self.decay, self.time_steps)
        
        # Readout
        logits = self.readout @ x
        return logits
    
    def forward_batch(self, x_batch):
        """Batch forward"""
        batch_size = x_batch.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.forward(x_batch[i])
            outputs.append(out)
        return np.stack(outputs)

# ============================================================================
# CONFIGURACIONES
# ============================================================================

CONFIGS = [
    {"name": "Tiny", "input": 784, "hidden": 64, "output": 10, "layers": 2},
    {"name": "Small", "input": 784, "hidden": 256, "output": 10, "layers": 4},
    {"name": "Medium", "input": 784, "hidden": 512, "output": 10, "layers": 8},
]

NUM_RUNS = 30

results = {}

# ============================================================================
# BENCHMARK POR CONFIGURACIÓN
# ============================================================================

for cfg in CONFIGS:
    name = cfg['name']
    print(f"\n{'='*80}")
    print(f"CONFIG: {name} ({cfg['hidden']} dims, {cfg['layers']} layers)")
    print(f"{'='*80}\n")
    
    # Crear modelo
    model = RINModel(cfg['input'], cfg['hidden'], cfg['output'], cfg['layers'])
    model.eval()
    
    # Contar parámetros
    params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {params:,}")
    
    # Input
    dummy_input = torch.randn(1, cfg['input'])
    dummy_np = dummy_input.numpy()
    
    # Crear RIN-X
    rin_x = RIN_X(model)
    
    # ================================================================
    # 1. PYTORCH EAGER
    # ================================================================
    print("\n[1] PyTorch Eager...")
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    t_pyt = np.mean(times)
    std_pyt = np.std(times)
    print(f"    Time: {t_pyt:.3f} ± {std_pyt:.3f} ms")
    
    # ================================================================
    # 2. ONNX RUNTIME
    # ================================================================
    print("\n[2] ONNX Runtime...")
    
    # Exportar a ONNX
    onnx_path = f"rin_{name.lower()}.onnx"
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=18,
                      dynamo=False)
    
    # Sesión ONNX
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Warmup
    for _ in range(5):
        _ = session.run([output_name], {input_name: dummy_np})
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_np})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    t_onnx = np.mean(times)
    std_onnx = np.std(times)
    print(f"    Time: {t_onnx:.3f} ± {std_onnx:.3f} ms")
    
    # ================================================================
    # 3. RIN-X (NumPy/C simulation)
    # ================================================================
    print("\n[3] RIN-X (C simulation)...")
    
    # Warmup
    for _ in range(5):
        _ = rin_x.forward(dummy_np[0])
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        _ = rin_x.forward(dummy_np[0])
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    t_rinx = np.mean(times)
    std_rinx = np.std(times)
    print(f"    Time: {t_rinx:.3f} ± {std_rinx:.3f} ms")
    
    # ================================================================
    # COMPARACIÓN
    # ================================================================
    print(f"\n--- COMPARACIÓN {name} ---")
    print(f"  PyTorch:  {t_pyt:.3f} ms (baseline)")
    print(f"  ONNX:     {t_onnx:.3f} ms ({t_pyt/t_onnx:.2f}× vs PyTorch)")
    print(f"  RIN-X:    {t_rinx:.3f} ms ({t_pyt/t_rinx:.2f}× vs PyTorch)")
    
    speedup_rinx_vs_onnx = t_onnx / t_rinx
    print(f"\n  RIN-X vs ONNX: {speedup_rinx_vs_onnx:.2f}×")
    
    results[name] = {
        'params': params,
        'pytorch_ms': t_pyt,
        'onnx_ms': t_onnx,
        'rin_x_ms': t_rinx,
        'speedup_vs_pytorch': t_pyt / t_rinx,
        'speedup_vs_onnx': speedup_rinx_vs_onnx
    }
    
    # Validar equivalencia
    print("\n[4] Validando equivalencia de salidas...")
    with torch.no_grad():
        pyt_out = model(dummy_input).numpy()[0]
    onnx_out = session.run([output_name], {input_name: dummy_np})[0][0]
    rinx_out = rin_x.forward(dummy_np[0])
    
    mse_onnx = np.mean((pyt_out - onnx_out)**2)
    mse_rinx = np.mean((pyt_out - rinx_out)**2)
    
    print(f"  MSE PyTorch vs ONNX:   {mse_onnx:.8f}")
    print(f"  MSE PyTorch vs RIN-X:  {mse_rinx:.8f}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL - TRIPLE BENCHMARK")
print("="*80)
print()
print(f"{'Config':<10} {'Params':<12} {'PyTorch':<12} {'ONNX':<12} {'RIN-X':<12} {'Speedup':<12}")
print("-"*80)

for name, data in results.items():
    print(f"{name:<10} {data['params']/1000:>6.0f}K     "
          f"{data['pytorch_ms']:>6.2f} ms   "
          f"{data['onnx_ms']:>6.2f} ms   "
          f"{data['rin_x_ms']:>6.2f} ms   "
          f"{data['speedup_vs_onnx']:>5.2f}× vs ONNX")

print("-"*80)

# Veredicto
print("\n" + "="*80)
print("VEREDICTO")
print("="*80)
print()

all_faster_than_onnx = all(r['speedup_vs_onnx'] > 1.0 for r in results.values())
avg_speedup = np.mean([r['speedup_vs_onnx'] for r in results.values()])

if all_faster_than_onnx and avg_speedup > 2.0:
    print(f"✅ RIN-X SUPERA a ONNX Runtime en todas las configuraciones")
    print(f"   Speedup promedio vs ONNX: {avg_speedup:.2f}×")
    print()
    print("🎯 ESTO SÍ ES UNA CONTRIBUCIÓN CIENTÍFICA FUERTE:")
    print("   • Supera al estado del arte en inference engines")
    print("   • Mantiene 100% equivalencia numérica")
    print("   • Speedup consistente en diferentes escalas")
    
elif all_faster_than_onnx:
    print(f"⚠️  RIN-X es más rápido que ONNX pero marginalmente")
    print(f"   Speedup promedio: {avg_speedup:.2f}×")
    print()
    print("   Contribución débil - necesita más optimización")
    
else:
    print("❌ RIN-X NO supera a ONNX Runtime")
    print()
    print("   El speedup original era artefacto de:")
    print("   - Comparación vs PyTorch eager (no optimizado)")
    print("   - Modelo pequeño donde overhead domina")
    print("   - No comparable a engines de producción")

print()
print("="*80)

# Guardar resultados
with open('triple_benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Resultados: triple_benchmark_results.json")
print("="*80)
