#!/usr/bin/env python3
"""
train_softmax_baseline.py — Fast softmax attention baseline, RIN export.
Same architecture as BHRR model for direct comparison.
Export format: symmetric quant centered at 128, compatible with rin_core.c.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, struct, os, math, time

# ── Data ──────────────────────────────────────────────────────────────
import urllib.request, pathlib
DATA_PATH = '/tmp/shakespeare_softmax.txt'
if not os.path.exists(DATA_PATH):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        urllib.request.urlretrieve(url, DATA_PATH)
    except:
        # Fallback: write minimal text
        with open(DATA_PATH, 'w') as f:
            f.write("ROMEO: But soft what light through yonder window breaks? It is the east and Juliet is the sun.")

with open(DATA_PATH, 'r') as f:
    TEXT = f.read()

data = torch.tensor(np.frombuffer(TEXT.encode(), dtype=np.uint8).astype(np.int64))
print(f"Data: {len(data)} chars", flush=True)

# ── Model ──────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.W1 = nn.Linear(dim, ffn_dim, bias=False)
        self.W2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.W2(F.relu(self.W1(self.ln2(x))))
        return x


class SoftmaxTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, ffn_dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.unembed = nn.Linear(dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.embed(idx)
        pos = self.pos_embed(torch.arange(T, device=idx.device))
        h = tok + pos.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.unembed(h)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0):
        for _ in range(max_new):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ── Config ──────────────────────────────────────────────────────────────
# Same arch as BHRR (larger)
dim, num_layers, num_heads, ffn_dim = 256, 4, 4, 1024
vocab_size = 256
max_seq, batch_size, block_size = 128, 16, 64
lr = 3e-3
max_iters = 1500
eval_interval = 500

model = SoftmaxTransformer(vocab_size, dim, num_layers, num_heads, ffn_dim, max_seq)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

def get_batch(split):
    d = data[:int(0.9*len(data))] if split == 'train' else data[int(0.9*len(data)):]
    ix = torch.randint(len(d)-block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

t0 = time.time()
for step in range(max_iters):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % eval_interval == 0 or step == max_iters - 1:
        model.eval()
        xv, yv = get_batch('val')
        _, vl = model(xv, yv)
        print(f"step {step:5d} train {loss.item():.4f} val {vl.item():.4f} {time.time()-t0:.0f}s", flush=True)
        model.train()

print(f"Training: {time.time()-t0:.0f}s", flush=True)

# ── Generate ──────────────────────────────────────────────────────────────
model.eval()
ctx = torch.tensor([[ord(c) for c in "RO"]], dtype=torch.long)
out = model.generate(ctx, 50, 0.8)
gen_text = bytes(out[0].tolist()).decode('utf-8', errors='replace')
print(f"=== SOFTMAX GENERATED ===", flush=True)
print(gen_text, flush=True)

# ── Export to RIN (symmetric quant, same format as train_bhrr.py) ──────
def quantize_uint8(t):
    if isinstance(t, torch.Tensor):
        t_np = t.detach().float().numpy()
    else:
        t_np = t.astype(np.float32)
    max_abs = max(abs(t_np.min()), abs(t_np.max()))
    if max_abs < 1e-8:
        return np.full(t_np.shape, 128, dtype=np.uint8), 1.0
    scale = max_abs / 127.0
    w_int8 = np.clip(np.round(t_np / scale), -127, 127).astype(np.int16)
    q = (w_int8 + 128).clip(0, 255).astype(np.uint8)
    return q, scale

def write_layer(f, W, bias_size):
    if isinstance(W, torch.Tensor):
        W_np = W.detach().float().numpy()
    else:
        W_np = W.astype(np.float32)
    b_np = np.zeros(bias_size, dtype=np.float32)
    wq, sc = quantize_uint8(W_np)
    f.write(struct.pack('<f', sc))
    f.write(wq.tobytes())
    f.write(b_np.tobytes())

def write_u32(f, v):
    f.write(struct.pack('I', v))

output_path = '/tmp/softmax_model/softmax_model.rin'
os.makedirs('/tmp/softmax_model', exist_ok=True)

with open(output_path, 'wb') as f:
    f.write(b'RIN1')
    write_u32(f, 0)    # version
    write_u32(f, 1)    # architecture: 1 = Transformer
    write_u32(f, num_layers)
    write_u32(f, dim)
    write_u32(f, vocab_size)
    write_u32(f, num_heads)
    write_u32(f, max_seq)
    write_u32(f, ffn_dim)

    # Embedding
    write_layer(f, model.embed.weight, dim)

    for i, block in enumerate(model.blocks):
        write_layer(f, block.attn.Wq.weight, dim)
        write_layer(f, block.attn.Wk.weight, dim)
        write_layer(f, block.attn.Wv.weight, dim)
        write_layer(f, block.attn.Wo.weight, dim)
        write_layer(f, block.W1.weight, ffn_dim)
        write_layer(f, block.W2.weight, dim)

    # Output (unembed)
    write_layer(f, model.unembed.weight, vocab_size)

    # Position embeddings as INT16 Q15
    pos_np = model.pos_embed.weight.detach().float().numpy()
    pos_q15 = np.clip(np.round(pos_np * 256), -32768, 32767).astype(np.int16)
    f.write(pos_q15.tobytes())

    # LayerNorm params
    write_u32(f, num_layers * 2 + 1)  # count
    for l in range(num_layers):
        for ln_name in ['ln1', 'ln2']:
            block = model.blocks[l]
            ln = getattr(block, ln_name)
            g = np.clip(np.round(ln.weight.detach().float().numpy() * 256), -32768, 32767).astype(np.int16)
            b = np.clip(np.round(ln.bias.detach().float().numpy() * 256), -32768, 32767).astype(np.int16)
            f.write(g.tobytes())
            f.write(b.tobytes())
    ln_f = model.ln_f
    g = np.clip(np.round(ln_f.weight.detach().float().numpy() * 256), -32768, 32767).astype(np.int16)
    b = np.clip(np.round(ln_f.bias.detach().float().numpy() * 256), -32768, 32767).astype(np.int16)
    f.write(g.tobytes())
    f.write(b.tobytes())

    # Charset
    for i in range(256):
        f.write(struct.pack('<B', i))

sz = os.path.getsize(output_path)
print(f"Exported: {output_path} ({sz:,} bytes, {sz/1024/1024:.2f} MB)", flush=True)

# Verify
with open(output_path, 'rb') as f:
    magic = f.read(4)
    ver = struct.unpack('I', f.read(4))[0]
    arch = struct.unpack('I', f.read(4))[0]
    nl = struct.unpack('I', f.read(4))[0]
    dm = struct.unpack('I', f.read(4))[0]
    vs = struct.unpack('I', f.read(4))[0]
    nh = struct.unpack('I', f.read(4))[0]
    ms = struct.unpack('I', f.read(4))[0]
    ff = struct.unpack('I', f.read(4))[0]
print(f"Verified: magic={magic} ver={ver} arch={arch} layers={nl} dim={dm} vocab={vs} heads={nh} maxseq={ms} ffn={ff}", flush=True)
