"""train_tiny_gpt.py — Train tiny GPT, export to RIN format (SIMPLIFIED, FAST)"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, struct, os, time, sys

# ── Data ────────────────────────────────────────────────────────────────────
TEXT = """ROMEO: But soft what light through yonder window breaks? It is the east and Juliet is the sun. Arise fair sun and kill the envious moon who is already sick and pale with grief. It is my lady O it is my love! O that she knew she were! She speaks yet she says nothing. What of that? Her eye discourses I will answer it. I am too bold tis not to me she speaks. Two of the fairest stars in all the heaven having some business do entreat her eyes to twinkle in their spheres till they return. What if her eyes were there they in her head? The brightness of her cheek would shame those stars as daylight doth a lamp. See how she leans her cheek upon her hand! O that I were a glove upon that hand that I might touch that cheek! JULIET: Ay me! ROMEO: She speaks O speak again bright angel for thou art as glorious to this night being o'er my head as is a winged messenger of heaven unto the white upturned wondering eyes of mortals that fall back to gaze on him when he bestrides the lazy pacing clouds and sails upon the bosom of the air. JULIET: O Romeo Romeo wherefore art thou Romeo? Deny thy father and refuse thy name or if thou wilt not be but sworn my love and I'll no longer be a Capulet. ROMEO: Shall I hear more or shall I speak at this? JULIET: Tis but thy name that is my enemy. Thou art thyself though not a Montague. What's Montague? it is nor hand nor foot nor arm nor face nor any other part belonging to a man. O be some other name! What's in a name? that which we call a rose by any other name would smell as sweet. So Romeo would were he not Romeo called retain that dear perfection which he owes without that title. Romeo doff thy name and for that name which is no part of thee take all myself. ROMEO: I take thee at thy word. Call me but love and I'll be new baptized. Henceforth I never will be Romeo. JULIET: What man art thou that thus bescreen'd in night so stumblest on my counsel? ROMEO: By a name I know not how to tell thee who I am. My name dear saint is hateful to myself because it is an enemy to thee. Had I it written I would tear the word. JULIET: My ears have not yet drunk a hundred words of that tongue's utterance yet I know the sound. Art thou not Romeo and a Montague? ROMEO: Neither fair saint if either thee dislike. JULIET: How camest thou hither tell me and wherefore? The orchard walls are high and hard to climb and the place death considering who thou art if any of my kinsmen find thee here. ROMEO: With love's light wings did I o'erperch these walls for stony limits cannot hold love out and what love can do that dares love attempt. Therefore thy kinsmen are no let to me. JULIET: If they do see thee they will murder thee. ROMEO: Alack there lies more peril in thine eye than twenty of their swords. Look thou but sweet and I am proof against their enmity. JULIET: I would not for the world they saw thee here. ROMEO: I have night's cloak to hide me from their sight. And but thou love me let them find me here. My life were better ended by their hate than death prorogued wanting of thy love. JULIET: By whose direction found'st thou out this place? ROMEO: By love who first did prompt me to inquire. He lent me counsel and I lent him eyes. I am no pilot yet wert thou as far as that vast shore washed with the farthest sea I would adventure for such merchandise."""

chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(TEXT), dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]
print(f"Vocab: {vocab_size} chars, Data: {len(data)} chars", flush=True)

# ── Model ───────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads; self.head_dim = dim//num_heads
        self.Wq = nn.Linear(dim,dim,bias=True)
        self.Wk = nn.Linear(dim,dim,bias=True)
        self.Wv = nn.Linear(dim,dim,bias=True)
        self.Wo = nn.Linear(dim,dim,bias=True)
    def forward(self, x):
        B,T,C = x.shape
        q = self.Wq(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = self.Wk(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = self.Wv(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.Wo(y)

class TransformerBlock(nn.Module):
    def __init__(self,dim,num_heads,ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim,num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.W1 = nn.Linear(dim,ffn_dim,bias=True)
        self.W2 = nn.Linear(ffn_dim,dim,bias=True)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.W2(F.relu(self.W1(self.ln2(x))))
        return x

class TinyGPT(nn.Module):
    def __init__(self,vocab_size,dim,num_layers,num_heads,ffn_dim,max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,dim)
        self.pos_embedding = nn.Embedding(max_seq_len,dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim,num_heads,ffn_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim,vocab_size,bias=True)
    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0,T,device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        for b in self.blocks: x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
            return logits,loss
        return logits
    @torch.no_grad()
    def generate(self,idx,max_new,temperature=1.0):
        for _ in range(max_new):
            idx_cond = idx[:,-128:]
            logits = self(idx_cond)
            logits = logits[:,-1,:]/temperature
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

# ── Config ──────────────────────────────────────────────────────────────────
dim, num_layers, num_heads, ffn_dim = 256, 4, 4, 1024
max_seq, batch_size, block_size = 128, 32, 64
lr, max_iters, eval_interval = 3e-3, 1500, 500

model = TinyGPT(vocab_size,dim,num_layers,num_heads,ffn_dim,max_seq)
opt = torch.optim.AdamW(model.parameters(),lr=lr)

def get_batch(split):
    d = train_data if split=='train' else val_data
    ix = torch.randint(len(d)-block_size,(batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x,y

print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
t0 = time.time()

for step in range(max_iters):
    xb,yb = get_batch('train')
    _,loss = model(xb,yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%eval_interval==0 or step==max_iters-1:
        model.eval()
        xv,yv = get_batch('val')
        _,vl = model(xv,yv)
        print(f"step {step:5d} train {loss.item():.4f} val {vl.item():.4f} {time.time()-t0:.0f}s", flush=True)
        model.train()

print(f"Training: {time.time()-t0:.0f}s", flush=True)

# ── Generate sample ─────────────────────────────────────────────────────────
model.eval()
ctx = torch.tensor([encode("ROMEO:")],dtype=torch.long)
out = model.generate(ctx,200,0.8)
print("=== GENERATED ===", flush=True)
print(decode(out[0].tolist()), flush=True)

# ── Export to RIN format ────────────────────────────────────────────────────
def quantize_int8(t):
    w = t.detach().float()
    wmin,wmax = w.min(),w.max()
    scale = (wmax-wmin)/254.0
    if scale<1e-10: scale=1e-10
    wq = ((w-wmin)/scale).clamp(0,255).to(torch.uint8)
    return wq.numpy().reshape(-1), scale.item(), wmin.item()

def write_layer(f,W,b):
    wq,sc,_ = quantize_int8(W)
    f.write(struct.pack('f',sc))
    f.write(wq.tobytes())
    f.write(b.detach().float().numpy().astype(np.float32).tobytes())

def write_u32(f,v): f.write(struct.pack('I',v))

print("Exporting...", flush=True)
f = open('tiny_gpt.rin','wb')
f.write(b'RIN1')
write_u32(f,1)   # version
write_u32(f,1)   # architecture: 1=Transformer
write_u32(f,num_layers)
write_u32(f,dim)
write_u32(f,vocab_size)
write_u32(f,num_heads)
write_u32(f,max_seq)
write_u32(f,ffn_dim)

write_layer(f, model.token_embedding.weight, torch.zeros(dim))

for i,block in enumerate(model.blocks):
    write_layer(f, block.attn.Wq.weight, block.attn.Wq.bias)
    write_layer(f, block.attn.Wk.weight, block.attn.Wk.bias)
    write_layer(f, block.attn.Wv.weight, block.attn.Wv.bias)
    write_layer(f, block.attn.Wo.weight, block.attn.Wo.bias)
    write_layer(f, block.W1.weight, block.W1.bias)
    write_layer(f, block.W2.weight, block.W2.bias)

write_layer(f, model.lm_head.weight, model.lm_head.bias)

pos_emb = model.pos_embedding.weight.detach().float().numpy()
pos_emb_int16 = (pos_emb*256).astype(np.int16)
f.write(pos_emb_int16.tobytes())

# Write character set (vocab_size bytes, one per char)
charset_bytes = [ord(c) for c in chars]  # ASCII byte values
f.write(bytes(charset_bytes))  # raw bytes, one per character
# Pad to 4-byte alignment
pad = (4 - (len(charset_bytes) % 4)) % 4
if pad: f.write(b'\x00' * pad)

f.close()

sz = os.path.getsize('tiny_gpt.rin')
print(f"Exported: tiny_gpt.rin ({sz:,} bytes, {sz/1024/1024:.2f} MB)", flush=True)

# Verify
f=open('tiny_gpt.rin','rb')
magic=f.read(4); ver=struct.unpack('I',f.read(4))[0]; arch=struct.unpack('I',f.read(4))[0]
nl=struct.unpack('I',f.read(4))[0]; dm=struct.unpack('I',f.read(4))[0]
vs=struct.unpack('I',f.read(4))[0]; nh=struct.unpack('I',f.read(4))[0]
ms=struct.unpack('I',f.read(4))[0]; ff=struct.unpack('I',f.read(4))[0]
f.close()
print(f"Verified: magic={magic} ver={ver} arch={arch} layers={nl} dim={dm} vocab={vs} heads={nh} maxseq={ms} ffn={ff}", flush=True)
