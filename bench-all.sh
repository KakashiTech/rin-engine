#!/usr/bin/env bash
#
# bench-all.sh — Reproducible RIN Engine benchmark suite
#
# Usage:
#   ./bench-all.sh              # full benchmark
#   ./bench-all.sh --quick      # minimal (10 iters)
#   ./bench-all.sh --json       # output as JSON
#
# Requirements:
#   - RIN Engine built and pip install -e . done
#   - onnxruntime (pip install onnxruntime) for ONNX baseline
#
# Output: prints comparison table and saves to bench-result-*.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-$SCRIPT_DIR/models/tiny_rin.rin}"
MODEL_ONNX="${2:-$SCRIPT_DIR/models/rin_tiny.onnx}"
WARMUP=10
ITERATIONS=100
JSON_OUT=0

if [[ "${1:-}" == "--json" ]] || [[ "${2:-}" == "--json" ]] || [[ "${3:-}" == "--json" ]]; then
    JSON_OUT=1
fi
if [[ "${1:-}" == "--quick" ]] || [[ "${2:-}" == "--quick" ]] || [[ "${3:-}" == "--quick" ]]; then
    ITERATIONS=10
fi

# ----- Check model files -----
if [ ! -f "$MODEL" ]; then
    echo "ERROR: RIN Engine model not found: $MODEL"
    echo "Train one with: python scripts/train_tiny_gpt.py"
    exit 1
fi

# ----- Detect hardware -----
CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | sed 's/.*: //')
CPU_COUNT=$(nproc)
RIN_VERSION=$(python3 -c "from rin import RinEngine; print(RinEngine.version())" 2>/dev/null || echo "unknown")

echo "============================================="
echo " RIN Engine Benchmark Suite"
echo "============================================="
echo " Date:       $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo " Host:       $(hostname)"
echo " CPU:        $CPU_MODEL"
echo " Cores:      $CPU_COUNT"
echo " RIN Engine ver:   $RIN_VERSION"
echo " Model:      $MODEL"
echo " Iterations: $ITERATIONS"
echo "============================================="
echo ""

# ----- 1. RIN Engine inference benchmark (Python API) -----
echo "[1] RIN Engine INT8 (Python API) ..."
RIN_TPS=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from rin import RinEngine

eng = RinEngine('$MODEL', mode='transformer')
try:
    result = eng.benchmark(mode='transformer', warmup=$WARMUP, iterations=$ITERATIONS)
    print(f'{result[\"tokens_per_second\"]:.1f}')
finally:
    eng.close()
" 2>&1)
RIN_MS=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from rin import RinEngine

eng = RinEngine('$MODEL', mode='transformer')
try:
    result = eng.benchmark(mode='transformer', warmup=$WARMUP, iterations=$ITERATIONS)
    print(f'{result[\"ms_per_token\"]:.3f}')
finally:
    eng.close()
" 2>&1)
echo "   $RIN_TPS tok/s  ($RIN_MS ms/tok)"

# ----- 2. RIN Engine energy measurement -----
echo "[2] RIN Engine INT8 energy ..."
RIN_ENERGY=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from rin import RinEngine

eng = RinEngine('$MODEL', mode='transformer')
try:
    eng.generate('Hello world', max_tokens=10)
    energy = eng.energy_joules
    tokens = eng.total_tokens
    if tokens > 0:
        print(f'{energy / tokens * 1000000:.1f}')
    else:
        print('0')
finally:
    eng.close()
" 2>&1)
echo "   $RIN_ENERGY µJ/tok"

# ----- 3. ONNX Runtime baseline (if available) -----
ONNX_TPS="N/A"
ONNX_MS="N/A"
ONNX_OK=0

if python3 -c "import onnxruntime" 2>/dev/null; then
    if [ -f "$MODEL_ONNX" ]; then
        echo "[3] ONNX Runtime FP32 (baseline) ..."
        ONNX_RESULT=$(python3 -c "
import onnxruntime as ort
import numpy as np
import time

session = ort.InferenceSession('$MODEL_ONNX', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
dummy = np.random.randn(*input_shape).astype(np.float32)

# warmup
for _ in range($WARMUP):
    session.run(None, {input_name: dummy})

# measure
times = []
for _ in range($ITERATIONS):
    t0 = time.perf_counter()
    session.run(None, {input_name: dummy})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

avg = np.mean(times)
tps = 1000.0 / avg
print(f'{tps:.1f},{avg:.3f}')
" 2>&1) || ONNX_RESULT=""
        if [ -n "$ONNX_RESULT" ]; then
            ONNX_TPS=$(echo "$ONNX_RESULT" | cut -d, -f1)
            ONNX_MS=$(echo "$ONNX_RESULT" | cut -d, -f2)
            ONNX_OK=1
            echo "   $ONNX_TPS tok/s  ($ONNX_MS ms/tok)"
        else
            echo "   ONNX benchmark failed"
        fi
    else
        echo "[3] ONNX model not found: $MODEL_ONNX (skipped)"
    fi
else
    echo "[3] onnxruntime not installed (skipped)"
fi

echo ""

# ----- Results summary -----
echo "============================================="
echo " RESULTS"
echo "============================================="
printf " %-25s %12s %12s\n" "Backend" "tok/s" "ms/tok"
printf " %-25s %12s %12s\n" "-------------------------" "------------" "-----------"
printf " %-25s %12s %12s\n" "RIN Engine INT8" "$RIN_TPS" "$RIN_MS ms"

if [ "$ONNX_OK" -eq 1 ]; then
    printf " %-25s %12s %12s\n" "ONNX FP32" "$ONNX_TPS" "$ONNX_MS ms"
    SPEEDUP=$(echo "scale=2; $ONNX_MS / $RIN_MS" | bc 2>/dev/null || echo "?")
    printf " %-25s %12s\n" "Speedup vs ONNX" "${SPEEDUP}x"
fi
printf " %-25s %12s\n" "Energy" "$RIN_ENERGY µJ/tok"
echo ""

# ----- JSON output -----
if [ "$JSON_OUT" -eq 1 ]; then
    cat <<EOF
{
  "date": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "cpu": "$CPU_MODEL",
  "cores": $CPU_COUNT,
  "rin_version": "$RIN_VERSION",
  "rin_tps": $RIN_TPS,
  "rin_ms_per_token": $RIN_MS,
  "rin_energy_uj_per_token": $RIN_ENERGY,
  "onnx_tps": "$ONNX_TPS",
  "onnx_ms_per_token": "$ONNX_MS",
  "speedup_vs_onnx": "$SPEEDUP"
}
EOF
fi

echo "============================================="
echo " Done — results saved above"
echo "============================================="
