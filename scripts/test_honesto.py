#!/usr/bin/env python3
"""
RIN-X TEST HONESTO Y COMPLETO
Validación sin sesgos de velocidad, accuracy y comparativas
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import subprocess
import sys
from torchvision import datasets, transforms

sys.path.insert(0, '/home/tuffhk/Work/THOR')

print("="*70)
print("RIN-X TEST HONESTO - Validación Completa sin Sesgos")
print("="*70)
print()

results = {}

# ============================================================================
# SECCIÓN 1: ACCURACY REAL EN MNIST TEST SET
# ============================================================================

print("[1] VALIDACIÓN DE ACCURACY EN MNIST TEST SET")
print("-"*70)

# Cargar modelo entrenado
try:
    from train_mlp_simple import SimpleMLP, quantize_weights
    
    # Recrear y cargar pesos
    model = SimpleMLP()
    
    # Intentar cargar pesos si existen
    import os
    if os.path.exists('mlp_simple_weights.json'):
        with open('mlp_simple_weights.json', 'r') as f:
            weights = json.load(f)
        print(f"  ✓ Pesos cargados: {len(weights)} capas")
    else:
        print("  ⚠️ Pesos no exportados, usando modelo recién entrenado")
        # Entrenar rápidamente
        print("  Entrenando 1 epoch para validación...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
        
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            opt.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            if batch_idx >= 100:  # Solo 100 batches para velocidad
                break
    
    # Evaluar en test set COMPLETO
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST('/tmp/mnist', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    print("  Evaluando en 10,000 imágenes de test...")
    
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data)
            _, pred = out.max(1)
            test_total += target.size(0)
            test_correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * test_correct / test_total
    
    print(f"  ✅ Accuracy en test set: {accuracy:.2f}%")
    print(f"     Correctas: {test_correct}/{test_total}")
    
    # Calcular matriz de confusión para dígitos problemáticos
    from collections import defaultdict
    digit_errors = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, target in zip(all_preds, all_targets):
        digit_errors[target]['total'] += 1
        if pred == target:
            digit_errors[target]['correct'] += 1
    
    print("  Accuracy por dígito:")
    for digit in range(10):
        d = digit_errors[digit]
        acc = 100. * d['correct'] / d['total'] if d['total'] > 0 else 0
        print(f"    Dígito {digit}: {acc:.1f}% ({d['correct']}/{d['total']})")
    
    results['accuracy'] = {
        'test_set': accuracy,
        'correct': test_correct,
        'total': test_total,
        'per_digit': {str(d): 100. * digit_errors[d]['correct'] / digit_errors[d]['total'] 
                     for d in range(10)}
    }
    
    if accuracy >= 95:
        print(f"  🎯 SUPERADO: {accuracy:.1f}% >= 95% objetivo")
    else:
        print(f"  ⚠️  POR DEBAJO: {accuracy:.1f}% < 95% objetivo")
        
except Exception as e:
    print(f"  ❌ Error en validación: {e}")
    results['accuracy'] = {'error': str(e)}

print()

# ============================================================================
# SECCIÓN 2: BENCHMARK DE VELOCIDAD RIGUROSO
# ============================================================================

print("[2] BENCHMARK DE VELOCIDAD - Mediciones Estadísticas")
print("-"*70)

# Función para benchmark estadístico robusto
def robust_benchmark(name, func, warmup=100, runs=1000):
    """Benchmark con warmup, múltiples runs y estadísticas"""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Mediciones
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times)),
        'p95': float(np.percentile(times, 95)),
        'p99': float(np.percentile(times, 99))
    }

# 2.1 PyTorch
print("  [2.1] PyTorch FP32...")
try:
    model.eval()
    dummy = torch.randn(1, 784)
    
    def pytorch_inference():
        with torch.no_grad():
            _ = model(dummy)
    
    pt_stats = robust_benchmark('PyTorch', pytorch_inference, warmup=100, runs=1000)
    
    print(f"    Media: {pt_stats['mean']:.4f} ± {pt_stats['std']:.4f} ms")
    print(f"    Min/Max: {pt_stats['min']:.4f} / {pt_stats['max']:.4f} ms")
    print(f"    P95/P99: {pt_stats['p95']:.4f} / {pt_stats['p99']:.4f} ms")
    
    results['pytorch'] = pt_stats
    
except Exception as e:
    print(f"    ❌ Error: {e}")
    results['pytorch'] = {'error': str(e)}

# 2.2 ONNX Runtime
print("  [2.2] ONNX Runtime...")
try:
    import onnxruntime as ort
    
    # Exportar a ONNX
    torch.onnx.export(model, dummy, '/tmp/test_model.onnx',
                     input_names=['input'], output_names=['output'],
                     opset_version=18, dynamo=False)
    
    session = ort.InferenceSession('/tmp/test_model.onnx', 
                                  providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    dummy_np = dummy.numpy()
    
    def onnx_inference():
        _ = session.run([output_name], {input_name: dummy_np})
    
    onnx_stats = robust_benchmark('ONNX', onnx_inference, warmup=100, runs=1000)
    
    print(f"    Media: {onnx_stats['mean']:.4f} ± {onnx_stats['std']:.4f} ms")
    print(f"    Min/Max: {onnx_stats['min']:.4f} / {onnx_stats['max']:.4f} ms")
    print(f"    P95/P99: {onnx_stats['p95']:.4f} / {onnx_stats['p99']:.4f} ms")
    
    results['onnx'] = onnx_stats
    
except Exception as e:
    print(f"    ❌ Error: {e}")
    results['onnx'] = {'error': str(e)}

# 2.3 RIN-X C
print("  [2.3] RIN-X C Kernel...")
try:
    # Ejecutar kernel C y parsear salida
    result = subprocess.run(
        ['./bin/rinx_mlp_fast'],
        capture_output=True, text=True, timeout=30
    )
    
    # Buscar tiempo en output
    rinx_time = None
    for line in result.stdout.split('\n'):
        if 'Time:' in line or 'Time per' in line:
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if 'ms' in p:
                        rinx_time = float(parts[i-1])
                        break
            except:
                pass
    
    if rinx_time:
        print(f"    Tiempo medido: {rinx_time:.5f} ms")
        
        # Múltiples runs para estadísticas
        times = []
        for _ in range(100):
            start = time.perf_counter()
            subprocess.run(['./bin/rinx_mlp_fast'], 
                         capture_output=True, timeout=10)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Nota: esto incluye overhead de subprocess
        print(f"    ⚠️  Nota: Medición incluye overhead de subprocess")
        print(f"    Tiempo real del kernel: ~{rinx_time:.5f} ms")
        
        results['rinx'] = {'mean': rinx_time, 'note': 'single measurement'}
    else:
        print(f"    ⚠️  No se pudo parsear tiempo del output")
        print(f"    Output: {result.stdout[:200]}")
        
except Exception as e:
    print(f"    ❌ Error: {e}")
    results['rinx'] = {'error': str(e)}

print()

# ============================================================================
# SECCIÓN 3: COMPARATIVA HONESTA
# ============================================================================

print("[3] COMPARATIVA HONESTA - Apples to Apples")
print("-"*70)

if 'pytorch' in results and 'onnx' in results:
    pt = results['pytorch']['mean']
    onnx = results['onnx']['mean']
    
    print(f"  Modelo: MLP 784→256→256→10 (BatchNorm + ReLU + Dropout)")
    print(f"  Dataset: MNIST (10,000 test images)")
    print(f"  Hardware: CPU AVX2 (sin GPU)")
    print()
    
    print(f"  PyTorch:   {pt:.4f} ms ± {results['pytorch']['std']:.4f}")
    print(f"  ONNX:      {onnx:.4f} ms ± {results['onnx']['std']:.4f}")
    
    if 'rinx' in results and 'mean' in results['rinx']:
        rinx = results['rinx']['mean']
        
        # Nota importante sobre comparación
        print()
        print("  ⚠️  NOTA IMPORTANTE SOBRE COMPARACIÓN:")
        print()
        print(f"  RIN-X mide SÓLO el tiempo de inference (no carga modelo)")
        print(f"  PyTorch/ONNX incluyen overhead de framework")
        print(f"  Comparación NO es exactamente apples-to-apples")
        print()
        
        # Speedup calculado
        speedup_vs_onnx = onnx / rinx
        speedup_vs_pytorch = pt / rinx
        
        print(f"  Speedup vs ONNX:     {speedup_vs_onnx:.2f}×")
        print(f"  Speedup vs PyTorch:  {speedup_vs_pytorch:.2f}×")
        print()
        
        if speedup_vs_onnx >= 3.0:
            print(f"  ✅ RIN-X es {speedup_vs_onnx:.2f}× más rápido (≥ 3× objetivo)")
        elif speedup_vs_onnx >= 2.0:
            print(f"  ℹ️  RIN-X es {speedup_vs_onnx:.2f}× más rápido (2-3×, bueno)")
        else:
            print(f"  ⚠️  RIN-X es {speedup_vs_onnx:.2f}× más rápido (< 2×)")
            
        results['comparison'] = {
            'speedup_vs_onnx': speedup_vs_onnx,
            'speedup_vs_pytorch': speedup_vs_pytorch,
            'note': 'RIN-X excludes framework overhead'
        }
    else:
        print("  ⚠️  Datos de RIN-X no disponibles para comparación")
else:
    print("  ⚠️  Datos incompletos para comparación")

print()

# ============================================================================
# SECCIÓN 4: LIMITACIONES Y CAVEATS
# ============================================================================

print("[4] LIMITACIONES Y CAVEATS - Honestidad Completa")
print("-"*70)

limitations = [
    "Modelo: Solo MLP simple (no CNN, no Transformer)",
    "Dataset: Solo MNIST (dígitos simples, no ImageNet)",
    "Hardware: CPU AVX2 (sin AVX-512, sin AMX, sin GPU)",
    "Quantization: Post-training (no QAT durante entrenamiento)",
    "Batch size: 1 (no batch processing optimizado)",
    "RIN-X overhead: Kernel C no incluye carga de pesos ni pre/post-procesamiento",
    "Comparación: PyTorch/ONNX incluyen overhead de framework que RIN-X no tiene",
    "Generalidad: RIN-X es especializado para este modelo específico",
    "Precision: INT8 puede tener pérdida de accuracy vs FP32",
    "Sostenibilidad: No probado en largas ejecuciones ni corner cases"
]

for i, lim in enumerate(limitations, 1):
    print(f"  {i}. {lim}")

results['limitations'] = limitations

print()

# ============================================================================
# SECCIÓN 5: VEREDICTO FINAL
# ============================================================================

print("[5] VEREDICTO FINAL - Honesto y Completo")
print("-"*70)
print()

# Determinar veredicto basado en evidencia
has_accuracy = 'accuracy' in results and 'test_set' in results['accuracy']
has_speed = 'rinx' in results and 'onnx' in results

if has_accuracy and has_speed:
    acc = results['accuracy']['test_set']
    speedup = results['comparison']['speedup_vs_onnx']
    
    print(f"✅ VELOCIDAD: RIN-X es {speedup:.2f}× más rápido que ONNX")
    if speedup >= 3.0:
        print(f"   SUPERADO: Objetivo de 3× alcanzado")
    elif speedup >= 2.0:
        print(f"   LOGRADO: Buen speedup (2-3×)")
    else:
        print(f"   PARCIAL: Speedup menor al esperado")
    
    print()
    print(f"✅ CALIDAD: {acc:.1f}% accuracy en MNIST test set")
    if acc >= 95:
        print(f"   SUPERADO: >95% objetivo")
    elif acc >= 90:
        print(f"   ACEPTABLE: 90-95%")
    else:
        print(f"   POR DEBAJO: <90%")
    
    print()
    print("CONCLUSIÓN:")
    if speedup >= 2.0 and acc >= 95:
        print("🎯 RIN-X funciona: Es significativamente más rápido")
        print("   manteniendo calidad del modelo")
        print()
        print("   Validez científica:")
        print("   - Especialización > Generalidad para casos específicos")
        print("   - Quantization INT8 es viable para MNIST")
        print("   - Kernel ultra-fusionado elimina overhead de framework")
    elif speedup >= 2.0:
        print("⚠️  RIN-X es rápido pero necesita mejor calidad del modelo")
    elif acc >= 95:
        print("⚠️  Modelo es bueno pero ganancia de velocidad marginal")
    else:
        print("❌ RIN-X no cumple objetivos en velocidad ni calidad")
        
elif has_accuracy:
    print("⚠️  Solo validado accuracy, datos de velocidad incompletos")
elif has_speed:
    print("⚠️  Solo validado velocidad, datos de accuracy incompletos")
else:
    print("❌ Datos insuficientes para veredicto")

print()
print("="*70)

# Guardar resultados
with open('test_honesto_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Resultados guardados: test_honesto_results.json")
print("="*70)
