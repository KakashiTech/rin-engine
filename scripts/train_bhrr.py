#!/usr/bin/env python3
"""
train_bhrr.py — Entrena transformer con atención BHRR (Bipolar Holographic Reduced Representations)
Exporta a formato .rin para inferencia en rin_core.

Pipeline:
  1. Carga Wikitext-2 como chars (vocab=256)
  2. Transformer con BHRRAttention en lugar de softmax attention
  3. Entrena con CrossEntropyLoss
  4. Exporta pesos a .rin binario

Arquitectura:
  embed → [BHRRAttn + FFN] × num_layers → unembed
  BHRRAttn: Q,K,V = Linear(x); sign(Q,K,V); C = cumsum(sign(K)·sign(V)); R = C_prev · sign(Q); out = Wo(R)
"""

import os, sys, struct, math, time, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# === Config ===
CONFIG = {
    'dim': 256,            # model dimension
    'num_heads': 4,        # number of attention heads
    'num_layers': 4,       # transformer layers
    'ffn_dim': 1024,       # FFN hidden dimension
    'vocab_size': 256,     # byte-level vocabulary
    'max_seq_len': 256,    # training sequence length
    'num_slots': 4,        # BHRR multi-slots (used by C engine)
    'batch_size': 32,
    'lr': 3e-4,
    'weight_decay': 0.01,
    'epochs': 10,
    'warmup_steps': 100,
    'eval_interval': 200,
    'save_interval': 1000,
    'data_path': '/tmp/shakespeare.txt',  # will download if missing
    'output_dir': '/tmp/bhrr_model',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# === Straight-Through Estimator for sign() ===
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(-1, 1)  # pass gradient for |x| < 1

def sign_ste(x):
    return SignSTE.apply(x)

# === BHRR Attention (replaces softmax QKV attention) ===
class BHRRAttention(nn.Module):
    """Bipolar HRR attention: C = Σ sign(K)·sign(V), R = C_prev · sign(Q)"""
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim)
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim)

        # Bipolar quantization (STE for backward)
        Qs = sign_ste(Q)
        Ks = sign_ste(K)
        Vs = sign_ste(V)

        # C[t] = Σ_{i<t} sign(K_i) · sign(V_i)
        kv_prod = Ks * Vs  # [B, T, nh, hd], each element ∈ {-1, +1}
        C = torch.cumsum(kv_prod, dim=1)  # cumulative sum

        # Shift by 1: C_prev[t] = C[t-1], C_prev[0] = 0
        C_shifted = torch.cat([torch.zeros_like(C[:, :1]), C[:, :-1]], dim=1)

        # Retrieve: R[t] = C_prev[t] · sign(Q[t])
        R = C_shifted * Qs  # [B, T, nh, hd] element-wise multiply

        # Combine heads and project
        R = R.contiguous().view(B, T, D)
        out = self.Wo(R)
        return out


# === Transformer Block ===
class BHRRTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = BHRRAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# === Full BHRR Transformer Model ===
class BHRRTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config['dim']

        self.embed = nn.Embedding(config['vocab_size'], config['dim'])
        self.pos_embed = nn.Embedding(config['max_seq_len'], config['dim'])
        self.blocks = nn.ModuleList([
            BHRRTransformerBlock(config['dim'], config['num_heads'], config['ffn_dim'])
            for _ in range(config['num_layers'])
        ])
        self.ln_f = nn.LayerNorm(config['dim'])
        self.unembed = nn.Linear(config['dim'], config['vocab_size'], bias=False)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.config['max_seq_len']

        # Embeddings
        tok = self.embed(x)
        pos = self.pos_embed(torch.arange(T, device=x.device))
        h = tok + pos.unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.unembed(h)
        return logits

    def generate(self, prompt, max_new_tokens=100, temperature=1.0):
        """Autoregressive generation"""
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = prompt.clone().detach()
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            if prompt_ids.size(1) > self.config['max_seq_len']:
                ctx = prompt_ids[:, -self.config['max_seq_len']:]
            else:
                ctx = prompt_ids
            
            logits = self.forward(ctx)[:, -1, :]
            
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            
            prompt_ids = torch.cat([prompt_ids, next_id], dim=1)
        
        return prompt_ids


# === Data ===
def prepare_dataset(path):
    """Prepare text dataset. Try Shakespeare, fallback to synthetic."""
    if os.path.exists(path):
        return
    import urllib.request
    urls = [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
    ]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded {url}")
            return
        except:
            continue
    
    print("Synthetic dataset (no network)")
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "hello", "world", "this", "is", "test", "of", "bhrr", "attention",
             "transformer", "neural", "network", "deep", "learning", "bipolar"]
    with open(path, 'w') as f:
        for i in range(50000):
            import random
            line = ' '.join(random.choice(words) for _ in range(random.randint(5, 20)))
            f.write(f"{line} {i}.\n")


class CharDataset(Dataset):
    """Byte-level character dataset from text file"""
    def __init__(self, path, seq_len):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        # Convert to bytes (0-255)
        self.data = np.frombuffer(text.encode('utf-8', errors='replace'), dtype=np.uint8)
        self.data = torch.from_numpy(self.data.astype(np.int64))
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# === Training ===
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'] * 100
        )
        self.best_loss = float('inf')
        self.step = 0
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for x, y in loader:
            x, y = x.to(self.config['device']), y.to(self.config['device'])
            
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            self.step += 1
            
            if self.step % self.config['eval_interval'] == 0:
                avg_loss = total_loss / n_batches
                print(f"  step {self.step}: loss={avg_loss:.4f}, ppl={math.exp(avg_loss):.2f}")
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save('best.pt')
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(self.config['device']), y.to(self.config['device'])
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)
    
    def save(self, name):
        path = os.path.join(self.config['output_dir'], name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'step': self.step,
            'best_loss': self.best_loss,
        }, path)
        print(f"  Saved {path}")


# === Export to RIN binary format ===
def export_rin(model, config, output_path):
    """Export trained BHRRTransformer to RIN binary format.
    Format matches C loader in rin_core.c (load_transformer_weight_layer_bias):
      - float scale (4 bytes)
      - INT8 weights (rows * cols bytes)
      - float biases (bias_count * 4 bytes)
    """
    model.eval()
    state = model.state_dict()

    def quantize_uint8(t):
        """Symmetric quantization to uint8 [0,255] centered at 128.
        w_fp32 ≈ w_int8 * scale, w_int8 ∈ [-127, 127].
        Stored as uint8 = w_int8 + 128.
        C VPMADDUBSW computes sum(uint8 * int8), correction -128*sum_x.
        Scale is small (~0.03) → no overflow."""
        if isinstance(t, torch.Tensor):
            t_np = t.detach().cpu().numpy().astype(np.float32)
        else:
            t_np = t.astype(np.float32)
        max_abs = max(abs(t_np.min()), abs(t_np.max()))
        if max_abs < 1e-8:
            q = np.full(t_np.shape, 128, dtype=np.uint8)
            return q, 1.0
        scale = max_abs / 127.0
        w_int8 = np.clip(np.round(t_np / scale), -127, 127).astype(np.int16)
        q = (w_int8 + 128).clip(0, 255).astype(np.uint8)
        return q, scale

    def write_weight_layer(f, W_np, bias_np):
        """Write weight layer: float scale + uint8 weights + float biases
        PyTorch stores Linear.weight as [out_features, in_features].
        C GEMV expects [M, K] = [out_features, in_features] row-major.
        No transpose needed."""
        W_q, sc = quantize_uint8(W_np)
        f.write(struct.pack('<f', sc))
        f.write(W_q.tobytes())
        f.write(bias_np.astype(np.float32).tobytes())

    dim = config['dim']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    ffn_dim = config['ffn_dim']
    vocab_size = config['vocab_size']
    max_seq_len = config['max_seq_len']

    with open(output_path, 'wb') as f:
        # Magic + version + architecture flag
        f.write(b'RIN1')
        f.write(struct.pack('<I', 0))  # version
        f.write(struct.pack('<I', 1))  # architecture: 1 = Transformer

        # Metadata (order must match load_transformer_header)
        f.write(struct.pack('<IIIIII',
            num_layers, dim, vocab_size, num_heads, max_seq_len, ffn_dim))

        def get_bias(name_prefix):
            """Get bias array for a layer, zeros if no bias."""
            key = name_prefix + '.bias'
            if key in state:
                return state[key].detach().cpu().numpy().astype(np.float32)
            return np.zeros(dim, dtype=np.float32)

        def get_bias_dim(name_prefix, d):
            key = name_prefix + '.bias'
            if key in state:
                return state[key].detach().cpu().numpy().astype(np.float32)
            return np.zeros(d, dtype=np.float32)

        # Embedding table: [vocab_size, dim] INT8 with scale + bias
        W_emb = state['embed.weight']
        W_emb_np = W_emb.detach().cpu().numpy().astype(np.float32)
        write_weight_layer(f, W_emb_np, np.zeros(dim, dtype=np.float32))

        # Per-layer weights
        for l in range(num_layers):
            # Wq, Wk, Wv, Wo: each [dim, dim]
            for name in ['Wq', 'Wk', 'Wv', 'Wo']:
                key = f'blocks.{l}.attn.{name}'
                W_np = state[f'{key}.weight'].detach().cpu().numpy().astype(np.float32)
                b_np = get_bias(key)
                write_weight_layer(f, W_np, b_np)

            # W1: [ffn_dim, dim], W2: [dim, ffn_dim]
            b1 = get_bias_dim(f'blocks.{l}.ffn.0', ffn_dim)
            W1_np = state[f'blocks.{l}.ffn.0.weight'].detach().cpu().numpy().astype(np.float32)
            write_weight_layer(f, W1_np, b1)

            b2 = get_bias_dim(f'blocks.{l}.ffn.2', dim)
            W2_np = state[f'blocks.{l}.ffn.2.weight'].detach().cpu().numpy().astype(np.float32)
            write_weight_layer(f, W2_np, b2)

        # Output layer (unembed): [vocab_size, dim]
        W_out_np = state['unembed.weight'].detach().cpu().numpy().astype(np.float32)
        b_out = get_bias_dim('unembed', vocab_size)
        write_weight_layer(f, W_out_np, b_out)

        # Position embeddings: [max_seq_len, dim] as INT16 Q15
        W_pos = state['pos_embed.weight']
        W_pos_np = W_pos.detach().cpu().numpy().astype(np.float32)
        W_pos_q15 = np.clip(np.round(W_pos_np * 256), -32768, 32767).astype(np.int16)
        f.write(W_pos_q15.tobytes())

        # LayerNorm weights: per-layer ln1, ln2, plus final ln_f
        f.write(struct.pack('<I', num_layers * 2 + 1))  # 2 per layer + final ln_f
        for l in range(num_layers):
            for ln_name in ['ln1', 'ln2']:
                prefix = f'blocks.{l}.{ln_name}'
                gamma = state[f'{prefix}.weight'].detach().cpu().numpy().astype(np.float32)
                beta = state[f'{prefix}.bias'].detach().cpu().numpy().astype(np.float32)
                # Store as int16 Q15 (gamma/beta are ~1.0, Q15 range is fine)
                gamma_q15 = np.clip(np.round(gamma * 256), -32768, 32767).astype(np.int16)
                beta_q15 = np.clip(np.round(beta * 256), -32768, 32767).astype(np.int16)
                f.write(gamma_q15.tobytes())
                f.write(beta_q15.tobytes())
        # Final ln_f
        gamma_f = state['ln_f.weight'].detach().cpu().numpy().astype(np.float32)
        beta_f = state['ln_f.bias'].detach().cpu().numpy().astype(np.float32)
        gamma_f_q15 = np.clip(np.round(gamma_f * 256), -32768, 32767).astype(np.int16)
        beta_f_q15 = np.clip(np.round(beta_f * 256), -32768, 32767).astype(np.int16)
        f.write(gamma_f_q15.tobytes())
        f.write(beta_f_q15.tobytes())

        # Char set (simple ASCII map 0-255)
        for i in range(256):
            f.write(struct.pack('<B', i))

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported {output_path} ({size_mb:.2f} MB)")


# === Main ===
def main():
    cfg = CONFIG
    print(f"Device: {cfg['device']}")
    print(f"Config: dim={cfg['dim']}, heads={cfg['num_heads']}, layers={cfg['num_layers']}")

    # Prepare data
    prepare_dataset(cfg['data_path'])

    # Dataset
    dataset = CharDataset(cfg['data_path'], cfg['max_seq_len'])
    split = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
    model = BHRRTransformer(cfg).to(cfg['device'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Training
    trainer = Trainer(model, cfg)
    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_ppl={math.exp(val_loss):.2f}")
        trainer.scheduler.step()

    # Export best model
    best_path = os.path.join(cfg['output_dir'], 'best.pt')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Export to RIN format
    rin_path = os.path.join(cfg['output_dir'], 'bhrr_model.rin')
    export_rin(model, cfg, rin_path)

    # Test generation
    print("\n=== Generation Test ===")
    model.eval()
    prompt = torch.tensor([[ord(c) for c in "The quick"]]).to(cfg['device'])
    output = model.generate(prompt, max_new_tokens=50, temperature=0.8)
    generated = bytes(output[0].tolist()).decode('utf-8', errors='replace')
    print(f"Prompt: 'The quick'")
    print(f"Generated: '{generated}'")

    print("\nDone! Use with: RIN_USE_BHRR ./rin_demo --model /tmp/bhrr_model/bhrr_model.rin")


if __name__ == '__main__':
    main()
