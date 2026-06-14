#!/usr/bin/env python3
"""
THOR Weight Packer v3 — 2:4 Structured Sparsity
Reads thor_genius_weights.bin, applies 2:4 pruning, writes thor_genius_v3.bin

For each group of 4 consecutive in_features, keeps the 2 largest-magnitude
weights and stores them as 2 INT8 values + a 4-bit index nibble.
Saves ~37.5% of hidden-layer weight memory.
"""
import struct
import os
import numpy as np


def pack_layer(weights_flat: bytes, rows: int, cols: int):
    """
    Apply 2:4 structured pruning to one layer's INT8 weights.

    For each group of 4 in_features:
      1. Find the 2 weights with largest absolute value
      2. Store them contiguously (sorted by original position)
      3. Create a 4-bit mask indicating non-zero positions

    Returns (packed_weight_bytes, index_nibble_bytes).
    """
    w = np.frombuffer(weights_flat, dtype=np.int8).reshape(rows, cols).astype(np.int16)
    num_groups = cols // 4
    packed = np.empty(rows * num_groups * 2, dtype=np.int8)
    indices = np.empty(rows * num_groups, dtype=np.uint8)

    for row in range(rows):
        for g in range(num_groups):
            off = g * 4
            group = w[row, off:off + 4]
            abv = np.abs(group)
            keep = np.argsort(abv)[-2:]          # indices of 2 largest magnitude
            keep_sorted = np.sort(keep)           # ascending by original position
            pi = row * num_groups * 2 + g * 2
            packed[pi] = np.int8(group[keep_sorted[0]])
            packed[pi + 1] = np.int8(group[keep_sorted[1]])
            mask = (1 << keep_sorted[0]) | (1 << keep_sorted[1])
            indices[row * num_groups + g] = mask

    idx_bytes = bytearray()
    for k in range(0, len(indices), 2):
        if k + 1 < len(indices):
            idx_bytes.append((indices[k] << 4) | indices[k + 1])
        else:
            idx_bytes.append(indices[k] << 4)

    return packed.tobytes(), bytes(idx_bytes)


def main():
    in_path = "thor_genius_weights.bin"
    out_path = "thor_genius_v3.bin"

    with open(in_path, "rb") as f:
        data = f.read()

    magic = data[0:4]
    if magic != b"THOR":
        print(f"ERROR: bad magic {magic!r}")
        return 1

    input_dim = struct.unpack("<I", data[4:8])[0]
    output_dim = struct.unpack("<I", data[8:12])[0]
    num_layers = struct.unpack("<I", data[12:16])[0]

    off = 16
    layer_dims = list(struct.unpack(f"<{num_layers}I", data[off:off + 4 * num_layers]))
    off += 4 * num_layers
    num_dims = struct.unpack("<I", data[off:off + 4])[0]
    off += 4

    print(f"Model: {input_dim} → {' → '.join(str(d) for d in layer_dims)} → {output_dim}")
    print(f"Layers: {num_layers} total ({num_layers - 1} hidden, 1 output)\n")

    with open(out_path, "wb") as f:
        # --- v3 header (same fields, magic='TH3R') ---
        f.write(b"TH3R")
        f.write(struct.pack("<III", input_dim, output_dim, num_layers))
        for d in layer_dims:
            f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", num_dims))

        prev = input_dim
        total_orig = 0
        total_v3 = 0

        # --- Hidden layers (packed INT8) ---
        for i in range(num_layers - 1):
            rows = layer_dims[i]
            cols = prev

            w_bytes = data[off:off + rows * cols]
            off += rows * cols
            scale = struct.unpack("<f", data[off:off + 4])[0]
            off += 4
            bias = data[off:off + rows * 4]
            off += rows * 4

            packed_w, idx = pack_layer(w_bytes, rows, cols)

            f.write(packed_w)
            f.write(idx)
            f.write(struct.pack("<f", scale))
            f.write(bias)

            orig_sz = rows * cols + 4 + rows * 4
            v3_sz = len(packed_w) + len(idx) + 4 + rows * 4
            total_orig += orig_sz
            total_v3 += v3_sz

            pct = (1 - v3_sz / orig_sz) * 100
            print(f"Layer {i} ({cols:>4}→{rows:<4}): "
                  f"{orig_sz / 1024:>8.1f} KB → {v3_sz / 1024:>8.1f} KB  ({pct:-5.1f}%)")
            prev = rows

        # --- Output layer (FP32, unchanged) ---
        r = output_dim
        c = prev
        w_out = data[off:off + r * c * 4]
        off += r * c * 4
        b_out = data[off:off + r * 4]

        f.write(w_out)
        f.write(b_out)

        out_sz = r * c * 4 + r * 4
        total_orig += out_sz
        total_v3 += out_sz
        print(f"Output    ({c:>4}→{r:<4}): "
              f"{out_sz / 1024:>8.1f} KB   (FP32, unchanged)")

    in_sz = os.path.getsize(in_path)
    out_sz = os.path.getsize(out_path)
    overall = (1 - out_sz / in_sz) * 100
    print(f"\n{'─' * 55}")
    print(f"  Original:  {in_sz / 1024:>8.1f} KB")
    print(f"  Packed:    {out_sz / 1024:>8.1f} KB")
    print(f"  Savings:   {overall:>7.2f}%")
    print(f"\n  Saved → {out_path}")

    # Quick validation: check packed dimensions
    total_packed_weights = 0
    total_indices = 0
    prev = input_dim
    for i in range(num_layers - 1):
        rows = layer_dims[i]
        cols = prev
        total_packed_weights += rows * cols // 2
        total_indices += rows * cols // 8
        prev = rows
    print(f"\n  Packed weight entries: {total_packed_weights}")
    print(f"  Index nibble bytes:    {total_indices}")
    print(f"  (vs original INT8:     {total_packed_weights * 2})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
