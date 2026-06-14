"""
RIN-X ULTIMATE - FASE 4: GLOBAL MEMORY PLANNING + AUTO-TUNING
TinyEngine-style memory scheduling + TVM-style auto-schedule
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import heapq
from collections import defaultdict

print("="*70)
print("RIN-X ULTIMATE - FASE 4: MEMORY PLANNING + AUTO-TUNING")
print("="*70)
print()

# ============================================================================
# GLOBAL MEMORY PLANNING (TinyEngine-style)
# ============================================================================

@dataclass
class MemoryBlock:
    """Bloque de memoria en el pool"""
    offset: int
    size: int
    is_free: bool = True
    tensor_name: Optional[str] = None
    
    def __lt__(self, other):
        return self.offset < other.offset

class GlobalMemoryPlanner:
    """
    Planificador global de memoria estilo TinyEngine
    No layer-wise optimization, sino graph-level planning
    """
    
    def __init__(self, total_memory: int = 256 * 1024 * 1024):  # 256MB default
        self.total_memory = total_memory
        self.blocks: List[MemoryBlock] = [MemoryBlock(0, total_memory, True)]
        self.allocations: Dict[str, MemoryBlock] = {}
        
        # Tensor lifetimes (cuándo se crea y destruye cada tensor)
        self.tensor_lifetimes: Dict[str, Tuple[int, int]] = {}  # (birth, death)
        self.tensor_sizes: Dict[str, int] = {}
        
    def analyze_tensor_lifetimes(self, graph_ops: List[Tuple[str, List[str], List[str]]]):
        """
        Analizar cuándo nace y muere cada tensor
        graph_ops: [(op_name, inputs, outputs), ...]
        """
        # birth: primera vez que se produce
        # death: última vez que se consume
        
        birth = {}
        death = {}
        
        for step, (op_name, inputs, outputs) in enumerate(graph_ops):
            # Inputs se usan en este step
            for inp in inputs:
                if inp not in birth:
                    continue  # Constante o parámetro
                death[inp] = max(death.get(inp, 0), step)
            
            # Outputs nacen en este step
            for out in outputs:
                if out not in birth:
                    birth[out] = step
                # Si ya existía (in-place), actualizar death
                death[out] = step
        
        self.tensor_lifetimes = {k: (birth[k], death[k]) for k in birth}
        
    def plan_memory_in_place(self) -> Dict[str, int]:
        """
        Planificación in-place: reutilizar memoria de tensors muertos
        Similar a TinyEngine
        """
        # Ordenar tensors por tiempo de nacimiento
        sorted_tensors = sorted(self.tensor_lifetimes.items(), 
                               key=lambda x: x[1][0])
        
        allocations = {}
        free_blocks = []  # (offset, size, death_time)
        
        for tensor_name, (birth, death) in sorted_tensors:
            size = self.tensor_sizes.get(tensor_name, 0)
            if size == 0:
                continue
            
            # Alinear a 64 bytes
            aligned_size = (size + 63) // 64 * 64
            
            # Buscar bloque reutilizable (in-place optimization)
            best_block = None
            best_idx = -1
            
            for idx, (offset, block_size, block_death) in enumerate(free_blocks):
                # El bloque debe estar muerto antes de que nazca el nuevo tensor
                if block_death <= birth and block_size >= aligned_size:
                    if best_block is None or block_size < best_block[1]:
                        best_block = (offset, block_size, block_death)
                        best_idx = idx
            
            if best_block is not None:
                # Reutilizar bloque
                offset, block_size, _ = best_block
                allocations[tensor_name] = offset
                
                # Actualizar bloque libre (puede sobrar espacio)
                remaining = block_size - aligned_size
                if remaining > 0:
                    free_blocks[best_idx] = (offset + aligned_size, remaining, death)
                else:
                    free_blocks.pop(best_idx)
            else:
                # Allocar nuevo espacio al final
                if free_blocks:
                    last_offset = max(offset + size for offset, size, _ in free_blocks)
                else:
                    last_offset = 0
                    for alloc in allocations.values():
                        last_offset = max(last_offset, alloc + aligned_size)
                
                allocations[tensor_name] = last_offset
            
            # Registrar este bloque como liberado cuando muera el tensor
            free_blocks.append((allocations[tensor_name], aligned_size, death))
            free_blocks.sort()  # Mantener ordenados por offset
        
        return allocations
    
    def calculate_peak_memory(self, allocations: Dict[str, int]) -> int:
        """Calcular memoria pico usada"""
        max_addr = 0
        for tensor, offset in allocations.items():
            size = self.tensor_sizes.get(tensor, 0)
            max_addr = max(max_addr, offset + size)
        return max_addr
    
    def optimize_for_reuse(self, graph_ops: List) -> Dict[str, int]:
        """
        Optimización completa de reutilización
        """
        self.analyze_tensor_lifetimes(graph_ops)
        allocations = self.plan_memory_in_place()
        peak = self.calculate_peak_memory(allocations)
        
        print(f"[MemoryPlanner] Peak memory: {peak / 1024 / 1024:.2f} MB")
        print(f"[MemoryPlanner] Tensors allocated: {len(allocations)}")
        
        return allocations

# ============================================================================
# AUTO-TUNING SYSTEM (TVM-style Meta-Schedule)
# ============================================================================

@dataclass
class TileConfig:
    """Configuración de tiling para una operación"""
    M_tile: int
    N_tile: int  
    K_tile: int
    M_warp: int
    N_warp: int
    unroll_factor: int
    
    def __hash__(self):
        return hash((self.M_tile, self.N_tile, self.K_tile, 
                    self.M_warp, self.N_warp, self.unroll_factor))

class PerformanceCache:
    """Cache de performances medidos"""
    def __init__(self):
        self.cache: Dict[str, float] = {}
    
    def get_key(self, op_type: str, M: int, N: int, K: int, config: TileConfig) -> str:
        return f"{op_type}_{M}_{N}_{K}_{config.M_tile}_{config.N_tile}_{config.K_tile}"
    
    def lookup(self, op_type: str, M: int, N: int, K: int, config: TileConfig) -> Optional[float]:
        key = self.get_key(op_type, M, N, K, config)
        return self.cache.get(key)
    
    def store(self, op_type: str, M: int, N: int, K: int, config: TileConfig, time_ms: float):
        key = self.get_key(op_type, M, N, K, config)
        self.cache[key] = time_ms

class AutoTuner:
    """
    Sistema de auto-tuning estilo TVM
    Busca la mejor configuración de tiles por operación
    """
    
    def __init__(self):
        self.cache = PerformanceCache()
        self.sample_count = 0
        self.max_samples = 100  # Límite para evitar explosión combinatorial
        
    def generate_candidates(self, M: int, N: int, K: int) -> List[TileConfig]:
        """Generar candidatos de configuración de tiling"""
        candidates = []
        
        # Espacio de búsqueda basado en cache sizes típicos
        M_tiles = [32, 64, 128, 256] if M >= 256 else [32, 64, 128]
        N_tiles = [32, 64, 128, 256] if N >= 256 else [32, 64, 128]
        K_tiles = [64, 128, 256, 512] if K >= 512 else [64, 128, 256]
        
        warps = [4, 8, 16]
        unrolls = [1, 2, 4]
        
        for Mt in M_tiles:
            if Mt > M:
                continue
            for Nt in N_tiles:
                if Nt > N:
                    continue
                for Kt in K_tiles:
                    if Kt > K:
                        continue
                    for Mw in warps:
                        if Mw > Mt:
                            continue
                        for Nw in warps:
                            if Nw > Nt:
                                continue
                            for unroll in unrolls:
                                candidates.append(TileConfig(
                                    M_tile=Mt, N_tile=Nt, K_tile=Kt,
                                    M_warp=Mw, N_warp=Nw, unroll_factor=unroll
                                ))
        
        return candidates
    
    def measure_config(self, config: TileConfig, M: int, N: int, K: int) -> float:
        """
        Medir performance de una configuración
        En implementación real, esto ejecutaría el kernel
        Aquí usamos un modelo analítico
        """
        # Modelo analítico de performance
        # Asume: time = compute_time + memory_time
        
        # Compute: M*N*K FLOPs
        flops = 2.0 * M * N * K
        
        # Estimación de throughput según tile sizes
        # Tiles más grandes = mejor locality pero más register pressure
        
        compute_efficiency = min(1.0, 
            (config.M_tile * config.N_tile * config.K_tile) / (128 * 128 * 256))
        
        # Memory traffic
        # A: M*K, B: K*N, C: M*N
        memory_bytes = (M * K + K * N + M * N) * 4
        
        # Cache hit rate estimada
        l2_cache_size = 256 * 1024  # 256KB
        working_set = (config.M_tile * config.K_tile + 
                      config.K_tile * config.N_tile +
                      config.M_tile * config.N_tile) * 4
        
        if working_set < l2_cache_size:
            cache_efficiency = 0.95  # 95% L2 hit
        elif working_set < 4 * l2_cache_size:
            cache_efficiency = 0.75
        else:
            cache_efficiency = 0.50
        
        # Memory bandwidth (estimado)
        memory_bw = 50e9  # 50 GB/s
        memory_time = (memory_bytes * (1 - cache_efficiency)) / memory_bw
        
        # Compute throughput (estimado)
        peak_gflops = 500  # 500 GFLOP/s
        compute_time = flops / (peak_gflops * 1e9 * compute_efficiency)
        
        total_time = max(compute_time, memory_time) * 1000  # ms
        
        return total_time
    
    def tune_gemm(self, M: int, N: int, K: int) -> TileConfig:
        """
        Encontrar mejor configuración para GEMM(M,N,K)
        """
        print(f"[AutoTuner] Tuning GEMM({M},{N},{K})...")
        
        candidates = self.generate_candidates(M, N, K)
        
        # Filtrar por cache
        candidates_to_measure = []
        for config in candidates:
            cached = self.cache.lookup("gemm", M, N, K, config)
            if cached is None:
                candidates_to_measure.append(config)
        
        # Limitar número de mediciones
        if len(candidates_to_measure) > self.max_samples:
            import random
            candidates_to_measure = random.sample(candidates_to_measure, self.max_samples)
        
        print(f"  Candidates: {len(candidates)}, to measure: {len(candidates_to_measure)}")
        
        # Medir candidatos
        best_config = None
        best_time = float('inf')
        
        for config in candidates_to_measure:
            time_ms = self.measure_config(config, M, N, K)
            self.cache.store("gemm", M, N, K, config, time_ms)
            
            if time_ms < best_time:
                best_time = time_ms
                best_config = config
        
        # También revisar cacheados
        for config in candidates:
            cached = self.cache.lookup("gemm", M, N, K, config)
            if cached is not None and cached < best_time:
                best_time = cached
                best_config = config
        
        print(f"  Best config: M{best_config.M_tile},N{best_config.N_tile},K{best_config.K_tile}")
        print(f"  Estimated time: {best_time:.3f} ms")
        
        return best_config
    
    def tune_conv2d(self, batch: int, in_c: int, out_c: int, 
                   h: int, w: int, k: int) -> TileConfig:
        """Tune convolución 2D"""
        # Similar a GEMM pero con consideraciones de stride/padding
        return self.tune_gemm(h * w, out_c, in_c * k * k)

# ============================================================================
# RUNTIME ADAPTIVO CON HARDWARE DETECTION
# ============================================================================

class HardwareInfo:
    """Información del hardware detectado"""
    def __init__(self):
        self.has_avx512 = False
        self.has_avx2 = False
        self.has_fma = False
        self.has_amx = False  # Advanced Matrix Extensions
        self.has_vnni = False  # Vector Neural Network Instructions
        
        self.l1_cache_size = 32 * 1024
        self.l2_cache_size = 256 * 1024
        self.l3_cache_size = 12 * 1024 * 1024
        
        self.num_cores = 1
        self.num_threads = 1
        
    def detect(self):
        """Detectar capacidades de CPU"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            
            flags = info.get('flags', [])
            self.has_avx512 = 'avx512f' in flags
            self.has_avx2 = 'avx2' in flags
            self.has_fma = 'fma' in flags
            self.has_amx = 'amx_tile' in flags
            self.has_vnni = 'avx512_vnni' in flags or 'avx_vnni' in flags
            
            self.num_cores = info.get('count', 1)
            
        except ImportError:
            print("[HardwareInfo] cpuinfo no disponible, usando defaults")
    
    def get_optimal_tile_config(self) -> Dict[str, int]:
        """Obtener configuración óptima según hardware"""
        if self.has_amx:
            # Sapphire Rapids+
            return {'M': 256, 'N': 256, 'K': 512}
        elif self.has_avx512:
            # AVX-512 but no AMX
            return {'M': 128, 'N': 128, 'K': 256}
        else:
            # AVX2
            return {'M': 64, 'N': 64, 'K': 128}
    
    def __str__(self):
        return (f"CPU: {self.num_cores} cores\n"
                f"AVX-512: {self.has_avx512}\n"
                f"AMX: {self.has_amx}\n"
                f"VNNI: {self.has_vnni}\n"
                f"L2 Cache: {self.l2_cache_size/1024:.0f}KB")

class AdaptiveRuntime:
    """
    Runtime que adapta la ejecución según hardware y workload
    """
    def __init__(self):
        self.hardware = HardwareInfo()
        self.hardware.detect()
        
        self.tuner = AutoTuner()
        self.memory_planner = GlobalMemoryPlanner()
        
        # Configuraciones cacheadas por workload
        self.workload_configs: Dict[str, Dict] = {}
        
    def get_kernel_config(self, op_type: str, *dims) -> Dict:
        """Obtener configuración óptima para una operación"""
        cache_key = f"{op_type}_{'_'.join(map(str, dims))}"
        
        if cache_key not in self.workload_configs:
            if op_type == "gemm":
                M, N, K = dims
                tile_config = self.tuner.tune_gemm(M, N, K)
                self.workload_configs[cache_key] = {
                    'tile': tile_config,
                    'use_avx512': self.hardware.has_avx512,
                    'num_threads': min(self.hardware.num_cores, 8)
                }
            elif op_type == "conv2d":
                # Similar
                self.workload_configs[cache_key] = self.hardware.get_optimal_tile_config()
        
        return self.workload_configs[cache_key]
    
    def schedule_workload(self, graph_ops: List, tensor_sizes: Dict[str, int]) -> Tuple[Dict, Dict]:
        """
        Schedule completo: memory + kernel configs
        """
        print("="*70)
        print("ADAPTIVE RUNTIME: Scheduling Workload")
        print("="*70)
        print(f"Hardware:\n{self.hardware}")
        
        # Memory planning
        self.memory_planner.tensor_sizes = tensor_sizes
        memory_plan = self.memory_planner.optimize_for_reuse(graph_ops)
        
        # Kernel tuning para cada operación
        kernel_configs = {}
        for op_name, inputs, outputs in graph_ops:
            # Extraer dims de inputs/outputs si es GEMM
            if 'matmul' in op_name.lower() or 'gemm' in op_name.lower():
                # Asumir shapes estandar para demo
                config = self.get_kernel_config('gemm', 256, 256, 256)
                kernel_configs[op_name] = config
        
        print(f"\n[AdaptiveRuntime] Scheduled {len(graph_ops)} operations")
        print(f"[AdaptiveRuntime] Peak memory: {self.memory_planner.calculate_peak_memory(memory_plan) / 1024:.0f} KB")
        
        return memory_plan, kernel_configs

# ============================================================================
# TEST
# ============================================================================

def test_memory_planner():
    """Probar planificador de memoria"""
    print("="*70)
    print("TEST: Global Memory Planner")
    print("="*70)
    
    planner = GlobalMemoryPlanner(total_memory=10 * 1024 * 1024)  # 10MB
    
    # Simular grafo: MLP simple
    # Layer 1: Linear(784, 256) → ReLU
    # Layer 2: Linear(256, 128) → ReLU
    # Layer 3: Linear(128, 10)
    
    graph_ops = [
        ('matmul_1', ['input', 'W1'], ['mm1_out']),
        ('relu_1', ['mm1_out'], ['act1']),
        ('matmul_2', ['act1', 'W2'], ['mm2_out']),
        ('relu_2', ['mm2_out'], ['act2']),
        ('matmul_3', ['act2', 'W3'], ['output']),
    ]
    
    # Tamaños de tensores (batch=32)
    batch = 32
    planner.tensor_sizes = {
        'input': batch * 784 * 4,
        'W1': 784 * 256 * 4,
        'mm1_out': batch * 256 * 4,
        'act1': batch * 256 * 4,
        'W2': 256 * 128 * 4,
        'mm2_out': batch * 128 * 4,
        'act2': batch * 128 * 4,
        'W3': 128 * 10 * 4,
        'output': batch * 10 * 4,
    }
    
    allocations = planner.optimize_for_reuse(graph_ops)
    
    print("\nAllocations:")
    for tensor, offset in sorted(allocations.items(), key=lambda x: x[1]):
        size = planner.tensor_sizes.get(tensor, 0)
        print(f"  {tensor}: offset={offset}, size={size} bytes")
    
    peak = planner.calculate_peak_memory(allocations)
    print(f"\nPeak memory: {peak / 1024:.1f} KB")
    print(f"Without reuse: {sum(planner.tensor_sizes.values()) / 1024:.1f} KB")
    print(f"Savings: {(1 - peak / sum(planner.tensor_sizes.values())) * 100:.1f}%")

def test_auto_tuner():
    """Probar auto-tuner"""
    print("\n" + "="*70)
    print("TEST: Auto-Tuner")
    print("="*70)
    
    tuner = AutoTuner()
    
    # Tunear GEMM de diferentes tamaños
    configs = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    for M, N, K in configs:
        best = tuner.tune_gemm(M, N, K)
        print()

def test_adaptive_runtime():
    """Probar runtime adaptativo"""
    print("\n" + "="*70)
    print("TEST: Adaptive Runtime")
    print("="*70)
    
    runtime = AdaptiveRuntime()
    
    # Simular workload
    graph_ops = [
        ('matmul_1', ['input', 'W1'], ['mm1_out']),
        ('relu_1', ['mm1_out'], ['act1']),
        ('matmul_2', ['act1', 'W2'], ['mm2_out']),
    ]
    
    tensor_sizes = {
        'input': 32 * 784 * 4,
        'W1': 784 * 256 * 4,
        'mm1_out': 32 * 256 * 4,
        'act1': 32 * 256 * 4,
        'W2': 256 * 128 * 4,
        'mm2_out': 32 * 128 * 4,
    }
    
    memory_plan, kernel_configs = runtime.schedule_workload(graph_ops, tensor_sizes)
    
    print("\n✓ FASE 4: Memory Planning + Auto-tuning + Adaptive Runtime implementado")

if __name__ == '__main__':
    test_memory_planner()
    test_auto_tuner()
    test_adaptive_runtime()
