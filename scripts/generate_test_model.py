"""Generate a minimal .rin test model for CI and smoke tests.

Usage: python scripts/generate_test_model.py [output_path]
Default output: models/tiny_rin.rin
"""
import struct
import sys
import os

def generate_rin(path: str):
    """Generate a minimal 1-layer MLP .rin model."""
    dim = 16
    hidden = 32
    vocab = 64

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    with open(path, 'wb') as f:
        # Magic
        f.write(b'RIN1')

        # Header: num_layers=1, input_dim=16, hidden_dim=32, output_dim=64
        f.write(struct.pack('<4I', 1, dim, hidden, vocab))

        # Layer 0 weights: [hidden, dim], uint8
        w = bytearray([128] * (hidden * dim))
        f.write(struct.pack('<f', 0.01))  # scale
        f.write(bytes(w))

        # Layer 0 bias: [hidden], float32
        for _ in range(hidden):
            f.write(struct.pack('<f', 0.0))

        # Layer 1 (output) weights: [vocab, hidden], uint8
        w2 = bytearray([128] * (vocab * hidden))
        f.write(struct.pack('<f', 0.01))
        f.write(bytes(w2))

        # Layer 1 (output) bias: [vocab], float32
        for _ in range(vocab):
            f.write(struct.pack('<f', 0.0))

        # Charset (all printable ASCII up to vocab)
        charset = bytes(range(32, 32 + vocab))
        # Pad to 4 bytes
        while len(charset) % 4:
            charset += b'\x00'
        f.write(charset)

    print(f"Generated test model: {path} ({os.path.getsize(path)} bytes)")

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'models/tiny_rin.rin'
    generate_rin(out)
