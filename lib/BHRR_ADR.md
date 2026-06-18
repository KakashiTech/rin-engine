# BHRR — Bounded Hash Retrieval & Replace  (Multi-Slot Attention)

## ADR (Architecture Decision Record)

### Estado: Implementado y Verificado (C engine + Python training, dim=256)

---

## Contexto

El blueprint PROGRES.MD (Idea 3) proponía reemplazar la KV cache de transformers
(1 GB para 2048 tokens) con Memoria Holográfica usando convolución circular vía DCT.

El análisis reveló 4 errores fatales en el diseño original:

| Error | Síntoma | Causa raíz |
|-------|---------|------------|
| DCT no diagonaliza convolución | Implementación incorrecta | Martucci (1994) demostró que solo DFT/FFT funciona |
| Overflow int32 en dim>256 | Valores saturados a ±32767 | FFT escala valores por N por etapa |
| O(d log d) por token | Más lento que KV cache para seq<16K | FFT inherentemente O(n log n) |
| Colapsa multi-head | Un vector por capa pierde información | No escala con número de cabezas |

Tras análisis iterativo se descartaron también:
- **Bipolar Spatter Codes** (Kanerva 1996): bind/unbind mediante sign XOR pero SNR insuficiente con 1 slot
- **Jerárquico (L1 exacto + L2 BHRR)**: las 256 dimensiones caben en 1 cache line, L1 sobra

## Decisión Final

**BHRR multi-slot** (S=8): en lugar de acumular en 1 vector todas las claves,
se distribuyen en S=8 slots según el hash de K_sign.

### Álgebra

```
K_sign[i] = sign(K[i]) ∈ {-1, +1}     Q_sign[i] = sign(Q[i]) ∈ {-1, +1}
V_sign[i] = sign(V[i]) ∈ {-1, +1}

slot_id = hash(K_sign) = Σ[ (K_sign[i] + 1)/2 · (i+1) ] mod S

Almacenar: slot[slot_id][i] += V_sign[i] · K_sign[i]   (acumulador int32)
Recuperar: ctx[i] = slot[slot_id][i] · Q_sign[i]       (element-wise)
```

Nota: el producto K_sign·V_sign en el store no es necesario para BHRR puro
(basta acumular V_sign), pero se incluye porque K_sign se necesita en retrieve
— y K_sign·V_sign·Q_sign es el triple producto canónico BHRR.

### SNR

```
Por slot:     SNR = √(d_head · S / K)
```

Para d_head=32, S=8, K=64: SNR = √(32·8/64) = √4 = 2.0

La SNR se confirmó empíricamente: con 8 slots la recuperación funciona;
con 1 slot el modelo no converge.

## Implementación

### Archivos creados

| Archivo | Propósito |
|---------|-----------|
| `lib/include/experimental/rin_bhrr.h` | `rin_bhrr_slot_id`, store, retrieve, init inline (70 líneas) |
| `lib/include/experimental/rin_hrr.h` | HRR bind/unbind/dot original (no usado en pipeline actual) |
| `lib/src/rin_core.c` | Integración transformer vía `#ifdef RIN_USE_BHRR` |
| `scripts/train_bhrr_fast.py` | Entrenamiento vectorizado multi-slot + export .rin |
| `scripts/train_softmax_baseline.py` | Softmax baseline para comparación |
| `lib/BHRR_ADR.md` | Este documento |

### API actual

```c
// Deterministic slot assignment
uint32_t rin_bhrr_slot_id(const int8_t* vec, uint32_t hd, uint32_t num_slots);

// Store K,V sign bits into slot
void rin_bhrr_store(RIN_BHRR_Mem* mem, const int8_t* ksign, const int8_t* vsign);
  // mem->slots[slot_id][i] += (int32_t)ksign[i] * (int32_t)vsign[i];
  // mem->total_stored++

// Retrieve context from slot using Q_sign
void rin_bhrr_ctx_retrieve(int32_t* out, const int32_t* ctx,
                           const int8_t* qsign, uint32_t hd);
  // out[i] = ctx[i] * qsign[i];
```

### Pipeline actual (`rin_core.c`)

```
Para cada capa l, cada cabeza h:
  1. Q = proj_Q(h_norm), K = proj_K(h_norm), V = proj_V(h_norm)
  2. ksign = sign(K), qsign = sign(Q), vsign = sign(V)
  3. slot = rin_bhrr_slot_id(ksign, hd, num_slots)
  4. ctx = mem->slots[slot];  // O(1) lookup
  5. attn = ctx ⊙ qsign       // retrieve
  6. attn *= 256               // scale to Q15 range
  7. rin_bhrr_store(mem, ksign, vsign)  // store for future tokens
  8. attn_out = attn @ Wo      // project back to d_model
```

### Cuantización

- Pesos: uint8 simétrico, escala float32 por matriz, bias int32
- GEMV: `W_u8 @ x_i8` con corrección VPMADDUBSW (offset 128)
- Atención: Q15 signed arithmetic (int16_t)
- Softmax parallel tail (PT): tabla exp basada en LUT en Q16

## Verificación (C engine, Shakespeare)

### Slot assignment: PASS

`rin_bhrr_slot_id` en C coincide exactamente con Python:
- `Σ bits[i]·(i+1) mod S` donde `bits = (K_sign + 1)/2`
- Verificado con sign vectors reales del modelo entrenado

### dim=128, 2 layers, 4 heads

| Métrica | BHRR (476K params) | Softmax baseline (460K params) |
|---------|-------------------|-------------------------------|
| Val loss (1500 steps) | 2.45 | 2.40 |
| Tiempo | 55s | 53s |
| Output C (greedy) | "RD the the than the be the be th" | — |
| Output Python (sample) | "RORD thand the thand " | — |

Output Python vs C difieren por diseño (multinomial vs argmax),
pero ambos generan texto reconocible del mismo dominio.

### dim=256, 2 layers, 8 heads

| Config | Best val loss | Output |
|--------|---------------|--------|
| BHRR (1.7M params) | 2.40 (step 2000) | "MI thand thay the thand the hand..." |
| Softmax (3.3M params, 4L) | 2.33 | — |

El output "MI thand thay..." demuestra que BHRR aprende patrones lingüísticos,
pero la cuantización uint8 a dim=256 acumula error suficiente para cambiar
el argmax en algunos tokens (R(82) → M(77) en el primer token generado).

A dim=128 el error de cuantización es menor: output coherente
("LARI:\\nS:\\nWhererere t c..." con formato de diálogo).

### 4-layer dim=256: Inestable

BHRR con 4 capas diverge después de step 500 (lr=3e-3 o lr=1e-3).
Posible causa: gradientes de sign() combinados con deep stack.
Softmax equivalente (4L) converge a val_loss 2.33.

## Trade-offs y Riesgos (actualizado)

### Lo que se gana
- **Memoria**: O(S·d_head) por cabeza, sin crecimiento con seq_len
- **Cómputo**: O(d) retrieve + O(d) store por token/head (sin O(seq²))
- **Contexto**: degradación graceful con SNR √(d·S/K)

### Lo que se pierde
- **Calidad**: ~0.05 nats gap vs softmax a dim=128 (val_loss 2.45 vs 2.40)
- **Cuantización**: error acumulado escala con d, argmax cambia a dim=256
- **Estabilidad**: 4 capas no convergen; softmax sí (posible: STE + deep stack)

### Preguntas abiertas
1. ¿Cuantización-aware training (simulate uint8 en forward) cierra el gap a dim≥256?
2. ¿La inestabilidad de 4 capas se resuelve con warmup + gradient clipping?
3. ¿Escala a dim=512, 8 capas, Wikitext-2 completo?
4. ¿La atención BHRR ofrece ventajas en latencia real (tok/s) vs KV cache?

## Próximos pasos

1. Entrenar softmax baseline a igual parámetros que BHRR en dim=256 para comparación justa
2. Investigar inestabilidad 4-capas (lr schedule, gradient clipping, warmup)
3. Implementar cuantización-aware training (fake-quant en forward pass)
4. Comparar tok/s entre BHRR y KV cache en C engine con seq lengths variables
5. Escalar a dim=512 con Wikitext-2 si 4-capas se estabiliza
