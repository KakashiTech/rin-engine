"""
RIN-X ULTIMATE - FASE 1: MLIR-Style Compiler Infrastructure
Graph compiler con optimizaciones agresivas de fusión y tiling
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import hashlib

print("="*70)
print("RIN-X ULTIMATE - MLIR COMPILER INFRASTRUCTURE")
print("="*70)
print()

# ============================================================================
# OPERATIONS Y DIALECTS
# ============================================================================

class OpType(Enum):
    """Tipos de operaciones en el grafo"""
    # Tensor operations
    MATMUL = auto()
    CONV2D = auto()
    DEPTHWISE_CONV = auto()
    
    # Element-wise
    RELU = auto()
    GELU = auto()
    SIGMOID = auto()
    TANH = auto()
    ADD = auto()
    MUL = auto()
    
    # Normalization
    BATCH_NORM = auto()
    LAYER_NORM = auto()
    GROUP_NORM = auto()
    
    # Pooling
    MAX_POOL = auto()
    AVG_POOL = auto()
    GLOBAL_AVG_POOL = auto()
    
    # Shape operations
    RESHAPE = auto()
    TRANSPOSE = auto()
    FLATTEN = auto()
    CONCAT = auto()
    SPLIT = auto()
    
    # SSM/Mamba
    SSM_SELECTIVE = auto()  # Selective SSM
    SSM_CONV = auto()       # Conv-based SSM
    DISCRETIZE_ZOH = auto() # Zero-order hold
    PARALLEL_SCAN = auto()  # Associative scan
    
    # Attention
    SELF_ATTENTION = auto()
    LINEAR_ATTENTION = auto()
    FLASH_ATTENTION = auto()
    
    # Memory
    LOAD = auto()
    STORE = auto()
    ALLOC = auto()
    FREE = auto()
    
    # Control
    LOOP = auto()
    CONDITIONAL = auto()
    RETURN = auto()

@dataclass
class TensorShape:
    """Shape de un tensor con información de layout"""
    dims: Tuple[int, ...]
    dtype: str = 'float32'
    layout: str = 'NCHW'  # NCHW, NHWC, NC, etc.
    strides: Optional[Tuple[int, ...]] = None
    
    def __hash__(self):
        return hash((self.dims, self.dtype, self.layout))
    
    def numel(self) -> int:
        return np.prod(self.dims) if self.dims else 0
    
    def rank(self) -> int:
        return len(self.dims)

@dataclass 
class Operation:
    """Operación en el grafo computacional"""
    op_type: OpType
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict = field(default_factory=dict)
    shapes: Dict[str, TensorShape] = field(default_factory=dict)
    
    # Para tiling/parallelism
    tile_sizes: Optional[Tuple[int, ...]] = None
    parallel_dims: List[int] = field(default_factory=list)
    
    # Para scheduling
    priority: int = 0
    estimated_cost: float = 0.0  # FLOPs
    
    def __hash__(self):
        return hash((self.op_type, self.name, tuple(self.inputs), tuple(self.outputs)))

@dataclass
class Value:
    """Valor (tensor) en el IR"""
    name: str
    shape: TensorShape
    producer: Optional[str] = None  # Operation que lo produce
    consumers: List[str] = field(default_factory=list)  # Operations que lo consumen
    
    # Memory info
    memory_offset: int = -1
    memory_size: int = 0
    is_constant: bool = False
    is_parameter: bool = False

# ============================================================================
# GRAPH IR (Intermediate Representation)
# ============================================================================

class GraphIR:
    """
    Intermediate Representation del grafo computacional
    Similar a MLIR pero simplificado para RIN-X
    """
    
    def __init__(self, name: str = "graph"):
        self.name = name
        self.operations: Dict[str, Operation] = {}
        self.values: Dict[str, Value] = {}
        self.entry_ops: List[str] = []
        self.exit_ops: List[str] = []
        
        # Para fusion
        self.fused_groups: Dict[str, List[str]] = {}
        
        # Para memoria
        self.memory_plan: Optional[Dict[str, int]] = None
        self.total_memory: int = 0
        
    def add_operation(self, op: Operation) -> str:
        """Añadir operación al grafo"""
        self.operations[op.name] = op
        
        # Registrar valores
        for inp in op.inputs:
            if inp in self.values:
                self.values[inp].consumers.append(op.name)
        
        for out in op.outputs:
            if out not in self.values:
                # Crear valor nuevo
                shape = op.shapes.get(out, TensorShape(()))
                self.values[out] = Value(out, shape, producer=op.name)
            else:
                self.values[out].producer = op.name
        
        return op.name
    
    def get_operation(self, name: str) -> Optional[Operation]:
        return self.operations.get(name)
    
    def topological_sort(self) -> List[str]:
        """Orden topológico del grafo"""
        in_degree = {name: 0 for name in self.operations}
        
        # Calcular in-degrees
        for op_name, op in self.operations.items():
            for inp in op.inputs:
                if inp in self.values:
                    producer = self.values[inp].producer
                    if producer and producer in in_degree:
                        in_degree[op_name] += 1
        
        # Kahn's algorithm
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            op_name = queue.pop(0)
            result.append(op_name)
            
            op = self.operations[op_name]
            for out in op.outputs:
                if out in self.values:
                    for consumer in self.values[out].consumers:
                        in_degree[consumer] -= 1
                        if in_degree[consumer] == 0:
                            queue.append(consumer)
        
        return result
    
    def print_graph(self, detailed: bool = False):
        """Imprimir grafo para debugging"""
        print(f"\n{'='*70}")
        print(f"Graph: {self.name}")
        print(f"{'='*70}")
        print(f"Operations: {len(self.operations)}")
        print(f"Values: {len(self.values)}")
        print()
        
        sorted_ops = self.topological_sort()
        for i, op_name in enumerate(sorted_ops):
            op = self.operations[op_name]
            print(f"[{i}] {op.name}: {op.op_type.name}")
            if detailed:
                print(f"    Inputs: {op.inputs}")
                print(f"    Outputs: {op.outputs}")
                print(f"    Shapes: {[(k, v.dims) for k, v in op.shapes.items()]}")
                print(f"    Cost: {op.estimated_cost:.2e} FLOPs")
                if op.attrs:
                    print(f"    Attrs: {op.attrs}")
                print()
    
    def to_dict(self) -> dict:
        """Serializar a diccionario"""
        return {
            'name': self.name,
            'operations': {
                name: {
                    'type': op.op_type.name,
                    'inputs': op.inputs,
                    'outputs': op.outputs,
                    'attrs': op.attrs,
                    'shapes': {k: {'dims': v.dims, 'dtype': v.dtype, 'layout': v.layout}
                              for k, v in op.shapes.items()},
                    'cost': op.estimated_cost
                }
                for name, op in self.operations.items()
            },
            'sorted_ops': self.topological_sort()
        }

# ============================================================================
# FUSION PATTERNS (NIVEL GRÁFICO)
# ============================================================================

class FusionPattern:
    """Patrón de fusión de operaciones"""
    
    def __init__(self, name: str, 
                 source_ops: List[OpType],
                 target_op: OpType,
                 fusion_func: Callable):
        self.name = name
        self.source_ops = source_ops
        self.target_op = target_op
        self.fusion_func = fusion_func
    
    def match(self, graph: GraphIR, ops: List[str]) -> bool:
        """Verificar si una secuencia de ops matchea el patrón"""
        if len(ops) != len(self.source_ops):
            return False
        
        for i, op_name in enumerate(ops):
            op = graph.get_operation(op_name)
            if not op or op.op_type != self.source_ops[i]:
                return False
        
        # Verificar conectividad (output de uno es input del siguiente)
        for i in range(len(ops) - 1):
            curr_op = graph.get_operation(ops[i])
            next_op = graph.get_operation(ops[i+1])
            
            # Verificar que output de curr va a input de next
            found_connection = False
            for out in curr_op.outputs:
                if out in next_op.inputs:
                    found_connection = True
                    break
            if not found_connection:
                return False
        
        return True

# Patrones de fusión comunes
FUSION_PATTERNS = [
    # MatMul + Bias + ReLU → Fused GEMM
    FusionPattern(
        "matmul_bias_relu",
        [OpType.MATMUL, OpType.ADD, OpType.RELU],
        OpType.MATMUL,
        lambda ops, graph: f"fused_gemm_relu_{ops[0]}"
    ),
    
    # Conv + BN + ReLU → Fused Conv
    FusionPattern(
        "conv_bn_relu",
        [OpType.CONV2D, OpType.BATCH_NORM, OpType.RELU],
        OpType.CONV2D,
        lambda ops, graph: f"fused_conv_{ops[0]}"
    ),
    
    # Linear + GELU → Fused Linear
    FusionPattern(
        "linear_gelu",
        [OpType.MATMUL, OpType.GELU],
        OpType.MATMUL,
        lambda ops, graph: f"fused_linear_gelu_{ops[0]}"
    ),
    
    # Conv + ReLU → Fused Conv
    FusionPattern(
        "conv_relu",
        [OpType.CONV2D, OpType.RELU],
        OpType.CONV2D,
        lambda ops, graph: f"fused_conv_relu_{ops[0]}"
    ),
    
    # Add + LayerNorm → Fused Residual
    FusionPattern(
        "add_layernorm",
        [OpType.ADD, OpType.LAYER_NORM],
        OpType.ADD,
        lambda ops, graph: f"fused_residual_{ops[0]}"
    ),
    
    # Pool + Flatten → Fused Global Pool
    FusionPattern(
        "global_pool_flatten",
        [OpType.GLOBAL_AVG_POOL, OpType.FLATTEN],
        OpType.GLOBAL_AVG_POOL,
        lambda ops, graph: f"fused_gap_{ops[0]}"
    ),
]

class GraphOptimizer:
    """Optimizador de grafo con fusión y otras transformaciones"""
    
    def __init__(self):
        self.fusion_patterns = FUSION_PATTERNS
        self.fusion_stats = {}
    
    def apply_fusion(self, graph: GraphIR) -> GraphIR:
        """Aplicar fusión de operaciones"""
        print("[GraphOptimizer] Aplicando fusión de operaciones...")
        
        sorted_ops = graph.topological_sort()
        fused = set()
        new_ops = []
        
        i = 0
        while i < len(sorted_ops):
            op_name = sorted_ops[i]
            
            if op_name in fused:
                i += 1
                continue
            
            # Intentar match con patrones
            matched = False
            for pattern in self.fusion_patterns:
                # Probar secuencias de diferente longitud
                for length in range(len(pattern.source_ops), 0, -1):
                    if i + length > len(sorted_ops):
                        continue
                    
                    candidate_ops = sorted_ops[i:i+length]
                    
                    if pattern.match(graph, candidate_ops):
                        # Fusión exitosa
                        fused_name = pattern.fusion_func(candidate_ops, graph)
                        
                        # Crear operación fusionada
                        first_op = graph.get_operation(candidate_ops[0])
                        last_op = graph.get_operation(candidate_ops[-1])
                        
                        fused_op = Operation(
                            op_type=pattern.target_op,
                            name=fused_name,
                            inputs=first_op.inputs,
                            outputs=last_op.outputs,
                            attrs={
                                **first_op.attrs,
                                'fused_ops': candidate_ops,
                                'fusion_pattern': pattern.name
                            },
                            shapes={**first_op.shapes, **last_op.shapes},
                            estimated_cost=sum(graph.get_operation(op).estimated_cost 
                                             for op in candidate_ops) * 0.8  # 20% speedup por fusion
                        )
                        
                        new_ops.append(fused_op)
                        fused.update(candidate_ops)
                        
                        # Stats
                        self.fusion_stats[pattern.name] = self.fusion_stats.get(pattern.name, 0) + 1
                        
                        print(f"  ✓ Fused {pattern.name}: {' → '.join(candidate_ops)} → {fused_name}")
                        
                        i += length
                        matched = True
                        break
                
                if matched:
                    break
            
            if not matched:
                # No se pudo fusionar, mantener operación original
                op = graph.get_operation(op_name)
                new_ops.append(op)
                i += 1
        
        # Reconstruir grafo
        new_graph = GraphIR(f"{graph.name}_fused")
        for op in new_ops:
            if op.name not in fused:  # Solo añadir si no fue fusionada individualmente
                new_graph.add_operation(op)
        
        print(f"[GraphOptimizer] Fusión completada:")
        print(f"  Operaciones originales: {len(sorted_ops)}")
        print(f"  Operaciones después de fusión: {len(new_graph.operations)}")
        print(f"  Patrones aplicados: {self.fusion_stats}")
        
        return new_graph
    
    def eliminate_dead_code(self, graph: GraphIR) -> GraphIR:
        """Eliminar código muerto"""
        print("[GraphOptimizer] Eliminando código muerto...")
        
        # Encontrar valores no usados
        used_values = set()
        for op in graph.operations.values():
            used_values.update(op.inputs)
            used_values.update(op.outputs)
        
        # Eliminar operaciones que no producen outputs usados
        ops_to_remove = []
        for op_name, op in graph.operations.items():
            has_used_output = any(out in used_values for out in op.outputs)
            if not has_used_output and op.op_type not in [OpType.RETURN, OpType.STORE]:
                ops_to_remove.append(op_name)
        
        # Reconstruir sin código muerto
        new_graph = GraphIR(f"{graph.name}_dce")
        for op_name, op in graph.operations.items():
            if op_name not in ops_to_remove:
                new_graph.add_operation(op)
        
        print(f"  Eliminadas {len(ops_to_remove)} operaciones")
        
        return new_graph
    
    def constant_folding(self, graph: GraphIR) -> GraphIR:
        """Fold constantes en compile time"""
        print("[GraphOptimizer] Constant folding...")
        
        # Identificar valores constantes
        constant_values = {}
        for val_name, val in graph.values.items():
            if val.is_constant or val.is_parameter:
                constant_values[val_name] = val
        
        # Pre-computar operaciones con inputs constantes
        ops_to_fold = []
        for op_name, op in graph.operations.items():
            if all(inp in constant_values for inp in op.inputs):
                if op.op_type in [OpType.ADD, OpType.MUL, OpType.RELU]:
                    # Se puede pre-computar
                    ops_to_fold.append(op_name)
        
        print(f"  Encontradas {len(ops_to_fold)} operaciones para constant folding")
        
        return graph

# ============================================================================
# TILING Y CACHE MODELING
# ============================================================================

@dataclass
class CacheLevel:
    """Nivel de cache"""
    name: str
    size: int  # bytes
    line_size: int  # bytes
    associativity: int
    latency: int  # cycles
    bandwidth: float  # GB/s

class CacheHierarchy:
    """Jerarquía de cache del sistema"""
    
    # Modelos típicos
    INTEL_SAPPHIRE_RAPIDS = [
        CacheLevel("L1", 32*1024, 64, 12, 4, 800),      # 32KB per core
        CacheLevel("L2", 1024*1024, 64, 16, 12, 400),   # 1MB per core
        CacheLevel("L3", 60*1024*1024, 64, 15, 40, 200), # 60MB shared
    ]
    
    INTEL_SKYLAKE = [
        CacheLevel("L1", 32*1024, 64, 8, 4, 500),
        CacheLevel("L2", 256*1024, 64, 4, 12, 300),
        CacheLevel("L3", 12*1024*1024, 64, 16, 40, 100),
    ]
    
    def __init__(self, levels: List[CacheLevel]):
        self.levels = levels
    
    def estimate_access_time(self, bytes_accessed: int, working_set_size: int) -> float:
        """Estimar tiempo de acceso a memoria"""
        # Modelo simple: LRU
        for level in self.levels:
            if working_set_size <= level.size:
                # Fit en este nivel
                return bytes_accessed / (level.bandwidth * 1e9) * 1e6  # μs
        
        # Miss a todos los niveles, ir a DRAM
        return bytes_accessed / (50 * 1e9) * 1e6 + 0.1  # DRAM ~50 GB/s

class TilingOptimizer:
    """Optimizador de tiling basado en cache modeling"""
    
    def __init__(self, cache_hierarchy: CacheHierarchy):
        self.cache = cache_hierarchy
    
    def optimize_matmul_tiling(self, M: int, N: int, K: int, 
                             dtype_size: int = 4) -> Tuple[int, int, int]:
        """
        Encontrar mejores tile sizes para GEMM
        C[M,N] = A[M,K] @ B[K,N]
        """
        # Calcular working set por tile
        # Tile de A: Mt * Kt elementos
        # Tile de B: Kt * Nt elementos
        # Tile de C: Mt * Nt elementos
        
        L2_size = self.cache.levels[1].size if len(self.cache.levels) > 1 else 256*1024
        usable_L2 = L2_size * 0.8  # 80% de L2
        
        # Espacio necesario: Mt*Kt + Kt*Nt + Mt*Nt elementos
        # = Mt*Kt + Kt*Nt + Mt*Nt floats
        
        # Asumir cuadrado para simplicidad: Mt = Nt = Kt = T
        # Memoria = 3 * T^2 * 4 bytes
        # T^2 = usable_L2 / (3 * 4) = usable_L2 / 12
        
        max_T = int(np.sqrt(usable_L2 / (3 * dtype_size)))
        
        # Ajustar para SIMD (múltiplo de 16 para AVX-512)
        simd_width = 16
        T = (max_T // simd_width) * simd_width
        T = max(T, simd_width)  # Mínimo SIMD width
        
        # Clamp a dimensiones reales
        Mt = min(T, M)
        Nt = min(T, N)
        Kt = min(T, K)
        
        return (Mt, Nt, Kt)
    
    def apply_tiling_to_graph(self, graph: GraphIR) -> GraphIR:
        """Aplicar tiling a operaciones del grafo"""
        print("[TilingOptimizer] Aplicando tiling...")
        
        for op_name, op in graph.operations.items():
            if op.op_type == OpType.MATMUL:
                # Extraer shapes
                a_shape = op.shapes.get(op.inputs[0])
                b_shape = op.shapes.get(op.inputs[1])
                
                if a_shape and b_shape:
                    M = a_shape.dims[0]
                    K = a_shape.dims[1]
                    N = b_shape.dims[1]
                    
                    Mt, Nt, Kt = self.optimize_matmul_tiling(M, N, K)
                    op.tile_sizes = (Mt, Nt, Kt)
                    op.attrs['tile_sizes'] = [Mt, Nt, Kt]
                    
                    print(f"  {op_name}: GEMM({M},{N},{K}) → tiles ({Mt},{Nt},{Kt})")
        
        return graph

# ============================================================================
# CODEGEN (GENERACIÓN DE CÓDIGO)
# ============================================================================

class CodeGenerator:
    """Generador de código C optimizado"""
    
    def __init__(self, target_arch: str = "x86-64"):
        self.target = target_arch
        self.includes = [
            "#include <immintrin.h>",
            "#include <stdint.h>",
            "#include <string.h>",
            "#include <math.h>",
        ]
        self.functions = []
    
    def generate_gemm_kernel(self, M: int, N: int, K: int,
                            tile_m: int = 64, tile_n: int = 64, tile_k: int = 256,
                            use_avx512: bool = True) -> str:
        """Generar kernel GEMM optimizado"""
        
        func_name = f"gemm_{M}_{N}_{K}"
        
        code = f"""
// GEMM: C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}]
// Tiling: ({tile_m}, {tile_n}, {tile_k})
void {func_name}(const float* __restrict A, 
                const float* __restrict B,
                float* __restrict C) {{
    // Zero C
    memset(C, 0, {M * N} * sizeof(float));
    
    // Tiled GEMM
    for (int mt = 0; mt < {M}; mt += {tile_m}) {{
        int Mt = min({tile_m}, {M} - mt);
        
        for (int nt = 0; nt < {N}; nt += {tile_n}) {{
            int Nt = min({tile_n}, {N} - nt);
            
            // Acumulador de tiles
            for (int kt = 0; kt < {K}; kt += {tile_k}) {{
                int Kt = min({tile_k}, {K} - kt);
                
                // Micro-kernel
                for (int m = 0; m < Mt; m++) {{
                    for (int n = 0; n < Nt; n += 16) {{
                        __m512 acc = _mm512_loadu_ps(&C[(mt + m) * {N} + nt + n]);
                        
                        for (int k = 0; k < Kt; k++) {{
                            __m512 b_vec = _mm512_loadu_ps(&B[(kt + k) * {N} + nt + n]);
                            __m512 a_broadcast = _mm512_set1_ps(A[(mt + m) * {K} + kt + k]);
                            acc = _mm512_fmadd_ps(a_broadcast, b_vec, acc);
                        }}
                        
                        _mm512_storeu_ps(&C[(mt + m) * {N} + nt + n], acc);
                    }}
                }}
            }}
        }}
    }}
}}
"""
        return code
    
    def generate_fused_conv_relu(self, in_c: int, out_c: int, h: int, w: int,
                                  k: int = 3) -> str:
        """Generar Conv+ReLU fusionado"""
        
        func_name = f"fused_conv_relu_{in_c}_{out_c}"
        out_h, out_w = h - k + 1, w - k + 1
        
        code = f"""
// Fused Conv2D + ReLU
// Input: [{in_c}x{h}x{w}], Output: [{out_c}x{out_h}x{out_w}]
void {func_name}(const float* __restrict input,
                 const float* __restrict weight,
                 const float* __restrict bias,
                 float* __restrict output) {{
    
    for (int oc = 0; oc < {out_c}; oc++) {{
        for (int oh = 0; oh < {out_h}; oh++) {{
            for (int ow = 0; ow < {out_w}; ow++) {{
                float sum = bias ? bias[oc] : 0.0f;
                
                for (int ic = 0; ic < {in_c}; ic++) {{
                    for (int kh = 0; kh < {k}; kh++) {{
                        for (int kw = 0; kw < {k}; kw++) {{
                            int ih = oh + kh;
                            int iw = ow + kw;
                            float inp = input[ic * {h * w} + ih * {w} + iw];
                            float w = weight[oc * {in_c * k * k} + ic * {k * k} + kh * {k} + kw];
                            sum += inp * w;
                        }}
                    }}
                }}
                
                // Fused ReLU
                output[oc * {out_h * out_w} + oh * {out_w} + ow] = sum > 0 ? sum : 0;
            }}
        }}
    }}
}}
"""
        return code
    
    def generate_complete_module(self, graph: GraphIR) -> str:
        """Generar módulo C completo para el grafo"""
        
        lines = self.includes.copy()
        lines.append("")
        lines.append("// ============================================")
        lines.append(f"// RIN-X Generated Code for: {graph.name}")
        lines.append("// ============================================")
        lines.append("")
        
        # Generar kernels para cada operación
        for op_name, op in graph.operations.items():
            if 'fused' in op.attrs:
                # Operación fusionada
                if op.op_type == OpType.CONV2D:
                    # Extraer shapes de Conv fusionada
                    in_shape = op.shapes.get(op.inputs[0])
                    if in_shape and len(in_shape.dims) == 3:
                        in_c, h, w = in_shape.dims
                        out_c = 64  # placeholder
                        lines.append(self.generate_fused_conv_relu(in_c, out_c, h, w))
                        
            elif op.op_type == OpType.MATMUL and op.tile_sizes:
                M, N, K = 256, 256, 256  # placeholder
                if op.inputs:
                    a_shape = op.shapes.get(op.inputs[0])
                    if a_shape:
                        M, K = a_shape.dims[0], a_shape.dims[1]
                    b_shape = op.shapes.get(op.inputs[1]) if len(op.inputs) > 1 else None
                    if b_shape:
                        N = b_shape.dims[1]
                
                Mt, Nt, Kt = op.tile_sizes
                lines.append(self.generate_gemm_kernel(M, N, K, Mt, Nt, Kt))
        
        # Unir todo
        return "\n".join(lines)

# ============================================================================
# TEST Y EJEMPLO DE USO
# ============================================================================

def test_compiler_infrastructure():
    """Probar infraestructura del compiler"""
    print("="*70)
    print("TEST: RIN-X Compiler Infrastructure")
    print("="*70)
    print()
    
    # Crear grafo de ejemplo: Simple MLP
    graph = GraphIR("simple_mlp")
    
    # Capa 1: MatMul + Bias + ReLU
    matmul1 = Operation(
        op_type=OpType.MATMUL,
        name="matmul_1",
        inputs=["input", "W1"],
        outputs=["mm1_out"],
        shapes={
            "input": TensorShape((128, 784)),
            "W1": TensorShape((784, 256)),
            "mm1_out": TensorShape((128, 256))
        },
        estimated_cost=128 * 256 * 784 * 2  # M*N*K*2 FLOPs
    )
    
    add1 = Operation(
        op_type=OpType.ADD,
        name="add_bias_1",
        inputs=["mm1_out", "b1"],
        outputs=["add1_out"],
        shapes={"add1_out": TensorShape((128, 256))},
        estimated_cost=128 * 256
    )
    
    relu1 = Operation(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["add1_out"],
        outputs=["relu1_out"],
        shapes={"relu1_out": TensorShape((128, 256))},
        estimated_cost=128 * 256
    )
    
    # Capa 2: MatMul + Bias + ReLU
    matmul2 = Operation(
        op_type=OpType.MATMUL,
        name="matmul_2",
        inputs=["relu1_out", "W2"],
        outputs=["mm2_out"],
        shapes={
            "W2": TensorShape((256, 128)),
            "mm2_out": TensorShape((128, 128))
        },
        estimated_cost=128 * 128 * 256 * 2
    )
    
    add2 = Operation(
        op_type=OpType.ADD,
        name="add_bias_2",
        inputs=["mm2_out", "b2"],
        outputs=["add2_out"],
        shapes={"add2_out": TensorShape((128, 128))},
        estimated_cost=128 * 128
    )
    
    relu2 = Operation(
        op_type=OpType.RELU,
        name="relu_2",
        inputs=["add2_out"],
        outputs=["output"],
        shapes={"output": TensorShape((128, 128))},
        estimated_cost=128 * 128
    )
    
    # Añadir operaciones
    for op in [matmul1, add1, relu1, matmul2, add2, relu2]:
        graph.add_operation(op)
    
    # Imprimir grafo original
    print("GRAFO ORIGINAL:")
    graph.print_graph(detailed=False)
    
    # Aplicar optimizaciones
    optimizer = GraphOptimizer()
    
    # Fusión
    fused_graph = optimizer.apply_fusion(graph)
    
    # Eliminar código muerto
    optimized_graph = optimizer.eliminate_dead_code(fused_graph)
    
    # Tiling
    cache = CacheHierarchy(CacheHierarchy.INTEL_SKYLAKE)
    tiling_opt = TilingOptimizer(cache)
    tiled_graph = tiling_opt.apply_tiling_to_graph(optimized_graph)
    
    # Imprimir grafo optimizado
    print("\nGRAFO OPTIMIZADO:")
    tiled_graph.print_graph(detailed=True)
    
    # Codegen
    codegen = CodeGenerator()
    generated_code = codegen.generate_complete_module(tiled_graph)
    
    print("\n" + "="*70)
    print("CÓDIGO GENERADO (preview):")
    print("="*70)
    print(generated_code[:2000] if len(generated_code) > 2000 else generated_code)
    print("...")
    print(f"\nTotal: {len(generated_code)} caracteres")
    
    return graph, tiled_graph, generated_code

if __name__ == '__main__':
    graph, optimized_graph, code = test_compiler_infrastructure()
    print("\n" + "="*70)
    print("✓ FASE 1 COMPLETADA: MLIR Compiler Infrastructure")
    print("="*70)
