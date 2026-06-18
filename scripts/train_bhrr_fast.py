#!/usr/bin/env python3
"""
train_bhrr_fast.py — Fast BHRR baseline, same format as softmax baseline.
For direct comparison: same arch, same data, same optimizer.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, struct, os, math, time

# ── Data (Shakespeare, same as softmax baseline) ──────────────────────
import urllib.request
DATA_PATH = '/tmp/shakespeare_bhrr.txt'
if not os.path.exists(DATA_PATH):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        urllib.request.urlretrieve(url, DATA_PATH)
    except:
        with open(DATA_PATH, 'w') as f:
            f.write("ROMEO: But soft what light through yonder window breaks?")
with open(DATA_PATH, 'r') as f:
    TEXT = f.read()
data = torch.tensor(np.frombuffer(TEXT.encode(), dtype=np.uint8).astype(np.int64))
print(f"Data: {len(data)} chars", flush=True)

# ── Sign Straight-Through Estimator ──────────────────────────────────
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.sign()
    @staticmethod
    def backward(ctx, grad_output): return grad_output.clamp(-1, 1)
def sign_ste(x): return SignSTE.apply(x)

# ── Slot hash (simple weighted-sum, matches C engine) ─────────────────
def slot_id(signs, num_slots):
    """signs: [B, hd] with values -1 or +1.
    Returns slot index per batch item.
    Must match rin_bhrr_slot_id in C engine."""
    B, hd = signs.shape
    bits = ((signs + 1) // 2).long()  # {0, 1}
    weights = torch.arange(1, hd + 1, device=signs.device)
    h = (bits * weights).sum(dim=-1)
    return h % num_slots

# ── Multi-slot BHRR Attention ────────────────────────────────────────
class BHRRAttention(nn.Module):
    def __init__(self, dim, num_heads, num_slots=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_slots = num_slots
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        hd = self.head_dim
        nh = self.num_heads
        S = self.num_slots

        Q = self.Wq(x).view(B, T, nh, hd)
        K = self.Wk(x).view(B, T, nh, hd)
        V = self.Wv(x).view(B, T, nh, hd)
        Qs = sign_ste(Q)
        Ks = sign_ste(K)
        Vs = sign_ste(V)

        kv_prod = Ks * Vs  # [B, T, nh, hd]

        # Attention scaling (matches softmax 1/sqrt(hd))
        scale_inv = hd ** 0.5

        # Slot assignment from Ks (vectorized over all heads)
        Ks_flat = Ks.reshape(-1, hd)
        bits = ((Ks_flat + 1) // 2).long()
        weights = torch.arange(1, hd + 1, device=x.device, dtype=torch.long)
        h = (bits * weights).sum(dim=-1)
        slot_ids = (h % S).view(B, T, nh)  # [B, T, nh]

        # Multi-slot retrieval: fully vectorized over heads and slots
        # oh: [B, T, nh, S, 1] — one-hot slot indicator
        # masked: [B, T, nh, S, hd] — kv broadcast to each slot, masked by membership
        oh = F.one_hot(slot_ids, S).float().unsqueeze(-1)
        masked = kv_prod.unsqueeze(3) * oh
        slot_cum = torch.cumsum(masked, dim=1)
        slot_cum_shifted = torch.cat([torch.zeros_like(slot_cum[:, :1]), slot_cum[:, :-1]], dim=1)
        slot_idx = slot_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, hd)
        R = slot_cum_shifted.gather(3, slot_idx).squeeze(3)

        # Multiply by query sign
        R = R * Qs
        R = R.contiguous().view(B, T, D)
        return self.Wo(R)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, num_slots=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = BHRRAttention(dim, num_heads, num_slots)
        self.ln2 = nn.LayerNorm(dim)
        self.W1 = nn.Linear(dim, ffn_dim, bias=False)
        self.W2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.W2(F.relu(self.W1(self.ln2(x))))
        return x

class BHRRTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, ffn_dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.num_slots = 8
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim, self.num_slots) for _ in range(num_layers)])
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

# ── Config (multi-slot test) ────────────────────────────────
dim, num_layers, num_heads, ffn_dim = 128, 2, 4, 512
vocab_size = 256
max_seq, batch_size, block_size = 128, 16, 64
lr = 3e-3
max_iters = 1500
eval_interval = 500

model = BHRRTransformer(vocab_size, dim, num_layers, num_heads, ffn_dim, max_seq)
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

# ── Generate ────────────────────────────────────────────────────────────
model.eval()
ctx = torch.tensor([[ord(c) for c in "RO"]], dtype=torch.long)
out = model.generate(ctx, 50, 0.8)
gen_text = bytes(out[0].tolist()).decode('utf-8', errors='replace')
print(f"=== BHRR GENERATED ===", flush=True)
print(gen_text, flush=True)

# ── Export to RIN ──────────────────────────────────────────────────────
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

def write_u32(f, v): f.write(struct.pack('I', v))

output_path = '/tmp/bhrr_model/bhrr_model.rin'
os.makedirs('/tmp/bhrr_model', exist_ok=True)

with open(output_path, 'wb') as f:
    f.write(b'RIN1')
    write_u32(f, 0)
    write_u32(f, 1)
    write_u32(f, num_layers)
    write_u32(f, dim)
    write_u32(f, vocab_size)
    write_u32(f, num_heads)
    write_u32(f, max_seq)
    write_u32(f, ffn_dim)

    write_layer(f, model.embed.weight, dim)

    for i, block in enumerate(model.blocks):
        write_layer(f, block.attn.Wq.weight, dim)
        write_layer(f, block.attn.Wk.weight, dim)
        write_layer(f, block.attn.Wv.weight, dim)
        write_layer(f, block.attn.Wo.weight, dim)
        write_layer(f, block.W1.weight, ffn_dim)
        write_layer(f, block.W2.weight, dim)

    write_layer(f, model.unembed.weight, vocab_size)

    pos_np = model.pos_embed.weight.detach().float().numpy()
    pos_q15 = np.clip(np.round(pos_np * 256), -32768, 32767).astype(np.int16)
    f.write(pos_q15.tobytes())

    write_u32(f, num_layers * 2 + 1)
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

    for i in range(256):
        f.write(struct.pack('<B', i))

sz = os.path.getsize(output_path)
print(f"Exported: {output_path} ({sz:,} bytes, {sz/1024/1024:.2f} MB)", flush=True)

with open(output_path, 'rb') as f:
    magic = f.read(4); ver = struct.unpack('I', f.read(4))[0]; arch = struct.unpack('I', f.read(4))[0]
    nl = struct.unpack('I', f.read(4))[0]; dm = struct.unpack('I', f.read(4))[0]
    vs = struct.unpack('I', f.read(4))[0]; nh = struct.unpack('I', f.read(4))[0]
    ms = struct.unpack('I', f.read(4))[0]; ff = struct.unpack('I', f.read(4))[0]
print(f"Verified: magic={magic} ver={ver} arch={arch} layers={nl} dim={dm} vocab={vs} heads={nh} maxseq={ms} ffn={ff}", flush=True)
