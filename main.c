/*
 * main.c - RIN Inference System Demo & Benchmark
 *
 * Usage:
 *   ./rin_demo <model>                             # MLP argmax (default)
 *   ./rin_demo <model> --snn                       # SNN mode
 *   ./rin_demo <model> --attn                      # Attention mode
 *   ./rin_demo <model> --thor                      # THOR heritage mode
 *   ./rin_demo <model> --transformer               # Transformer text generation
 *   ./rin_demo <model> --transformer --prompt="Hi" # Transformer with prompt
 *   ./rin_demo <model> --transformer --gen=50      # Generate 50 tokens
 *   ./rin_demo <model> --benchmark [--runs=N]
 *   ./rin_demo <model> --energy [--runs=N]
 */

#include "rin_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static uint64_t read_rapl_uj(void) {
    int fd = open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", O_RDONLY);
    if (fd < 0) return 0;
    char buf[64];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n < 0) return 0;
    buf[n] = 0;
    return strtoull(buf, NULL, 10);
}

static RIN_Context setup_model(const char* path, int mode, int energy) {
    RIN_Context ctx;
    RIN_Config cfg = RIN_GetTinyConfig();
    cfg.enable_energy_monitoring = energy;
    cfg.model_dim = 2048;
    cfg.num_layers = 4;
    cfg.vocab_size = 10;
    cfg.max_seq_len = 784;
    cfg.timesteps = 4;
    cfg.inference_mode = mode;

    if (RIN_Init(&ctx, &cfg) != RIN_STATUS_OK) {
        printf("  ERROR: RIN_Init failed\n"); ctx.initialized = 0; return ctx;
    }
    if (RIN_LoadWeights(&ctx, path) != RIN_STATUS_OK) {
        printf("  ERROR: RIN_LoadWeights failed\n"); RIN_Destroy(&ctx); ctx.initialized = 0;
    }
    return ctx;
}

static const char* mode_name(int mode) {
    switch (mode) {
        case RIN_MODE_SNN:  return "SNN (LIF)";
        case RIN_MODE_ATTN: return "Attention";
        case RIN_MODE_THOR: return "THOR (heritage)";
        case RIN_MODE_TRANSFORMER: return "Transformer";
        default:            return "MLP";
    }
}

static void benchmark_speed(const char* path, int mode, int runs) {
    printf("\n  SPEED BENCHMARK [%s]\n", mode_name(mode));
    printf("  -----------------------\n");

    RIN_Context ctx = setup_model(path, mode, 0);
    if (!ctx.initialized) return;

    uint32_t input[784];
    for (int i = 0; i < 784; i++) input[i] = (uint32_t)(i % 10);
    RIN_InferenceResult result;
    RIN_Token tokens[10];
    result.tokens = tokens;

    for (int i = 0; i < 200; i++)
        RIN_Inference(&ctx, input, 784, 1, &result);

    double t0 = now_us();
    for (int i = 0; i < runs; i++)
        RIN_Inference(&ctx, input, 784, 1, &result);
    double t1 = now_us();

    double us_i = (t1 - t0) / runs;
    printf("  Runs:       %d\n", runs);
    printf("  Per inf:    %.1f us\n", us_i);
    printf("  Throughput: %.0f inf/s\n", 1e6 / us_i);
    printf("  Predicted:  %u\n", result.tokens[0].id);
    RIN_Destroy(&ctx);
}

static void benchmark_energy(const char* path, int mode, int runs) {
    printf("\n  ENERGY BENCHMARK [%s]\n", mode_name(mode));
    printf("  -----------------------\n");

    RIN_Context ctx = setup_model(path, mode, 0);
    if (!ctx.initialized) return;

    uint32_t input[784];
    for (int i = 0; i < 784; i++) input[i] = (uint32_t)(i % 10);
    RIN_InferenceResult result;
    RIN_Token tokens[10];
    result.tokens = tokens;

    for (int i = 0; i < 50; i++)
        RIN_Inference(&ctx, input, 784, 1, &result);

    uint64_t rapl_before = read_rapl_uj();
    if (rapl_before == 0) {
        printf("  ERROR: Cannot read RAPL (try sudo)\n"); RIN_Destroy(&ctx); return;
    }

    double t0 = now_us();
    for (int i = 0; i < runs; i++)
        RIN_Inference(&ctx, input, 784, 1, &result);
    double t1 = now_us();

    uint64_t rapl_after = read_rapl_uj();
    double uj_i = (double)(rapl_after - rapl_before) / runs;
    printf("  Runs:       %d\n", runs);
    printf("  Per inf:    %.1f us\n", (t1 - t0) / runs);
    printf("  Per inf:    %.0f uJ/pkg\n", uj_i);
    printf("  Avg power:  %.1f W\n", (double)(rapl_after - rapl_before) / (t1 - t0));
    printf("  Predicted:  %u\n", result.tokens[0].id);
    RIN_Destroy(&ctx);
}

/* Encode string to token IDs using model's charset */
static int encode_string(RIN_Context* ctx, const char* text, uint32_t* ids, int max_ids) {
    int vs;
    const char* cs = RIN_GetCharSet(ctx, &vs);
    if (!cs || vs <= 0) return -1;

    int n = 0;
    for (int i = 0; text[i] && n < max_ids; i++) {
        char c = text[i];
        int found = 0;
        for (int j = 0; j < vs; j++) {
            if (cs[j] == c) { ids[n++] = (uint32_t)j; found = 1; break; }
        }
        if (!found) ids[n++] = 0;  /* unknown → token 0 */
    }
    return n;
}

/* Decode token IDs to string using model's charset */
static void decode_ids(RIN_Context* ctx, const uint32_t* ids, int n, char* out, int max_out) {
    int vs;
    const char* cs = RIN_GetCharSet(ctx, &vs);
    if (!cs || vs <= 0) {
        for (int i = 0; i < n && i < max_out - 1; i++)
            out[i] = (char)('0' + ids[i] % 10);
        out[n < max_out - 1 ? n : max_out - 1] = 0;
        return;
    }
    int oi = 0;
    for (int i = 0; i < n && oi < max_out - 1; i++) {
        int id = (int)ids[i];
        if (id >= 0 && id < vs) out[oi++] = cs[id];
        else out[oi++] = '?';
    }
    out[oi] = 0;
}

int main(int argc, char** argv) {
    printf("\n  RIN v1.0 - 5 inference modes: MLP | SNN | Attention | THOR | Transformer\n");
    printf("  Backend: THOR INT8 SIMD (AVX2) + BSPN + PTsoftmax\n\n");

    if (argc < 2) {
        printf("  Usage: %s <model> [--snn|--attn|--thor|--transformer|--sample|--benchmark|--energy]\n", argv[0]);
        printf("         %s <model> --transformer --gen=N --prompt=\"text\"\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int mode = RIN_MODE_MLP;
    int action = 0;  /* 0=infer, 1=benchmark, 2=energy, 3=sample */
    int runs = 5000;
    int gen_count = 20;
    const char* prompt = "";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--snn") == 0) mode = RIN_MODE_SNN;
        else if (strcmp(argv[i], "--attn") == 0) mode = RIN_MODE_ATTN;
        else if (strcmp(argv[i], "--thor") == 0) mode = RIN_MODE_THOR;
        else if (strcmp(argv[i], "--transformer") == 0) mode = RIN_MODE_TRANSFORMER;
        else if (strcmp(argv[i], "--sample") == 0) { mode = RIN_MODE_MLP; action = 3; }
        else if (strcmp(argv[i], "--benchmark") == 0) action = 1;
        else if (strcmp(argv[i], "--energy") == 0) action = 2;
        else if (strncmp(argv[i], "--runs=", 7) == 0) runs = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "--gen=", 6) == 0) gen_count = atoi(argv[i] + 6);
        else if (strncmp(argv[i], "--prompt=", 9) == 0) prompt = argv[i] + 9;
    }

    if (action == 1) { benchmark_speed(model_path, mode, runs); return 0; }
    if (action == 2) { benchmark_energy(model_path, mode, runs); return 0; }

    RIN_Context ctx = setup_model(model_path, mode, 1);
    if (!ctx.initialized) return 1;

    RIN_InferenceResult result;
    uint32_t input[256];
    RIN_Token tokens[64];
    result.tokens = tokens;

    if (mode == RIN_MODE_TRANSFORMER) {
        /* --- TRANSFORMER TEXT GENERATION --- */
        int n_prompt = 0;
        if (prompt && prompt[0]) {
            n_prompt = encode_string(&ctx, prompt, input, 256);
            if (n_prompt <= 0) {
                printf("  WARNING: Could not encode prompt, charset missing in model\n");
                printf("  Using empty prompt\n");
                n_prompt = 1;
                input[0] = 0;
            }
        } else {
            /* Default prompt: start with a common character */
            input[0] = 0;  /* typically space or first char in vocab */
            n_prompt = 1;
        }

        printf("  Mode:       Transformer\n");
        printf("  Prompt:     \"%s\" (%d tokens)\n", prompt, n_prompt);
        printf("  Generate:   %d tokens\n", gen_count);

        uint64_t t0 = now_us();
        RIN_Status st = RIN_Inference(&ctx, input, (uint32_t)n_prompt,
                                       (uint32_t)gen_count, &result);
        uint64_t t1 = now_us();

        if (st == RIN_STATUS_OK) {
            printf("  Latency:    %.0f us\n", t1 - t0);
            printf("  Energy:     %.2f mJ/pkg\n", result.energy_joules * 1000);

            printf("  Tokens:    ");
            for (uint32_t i = 0; i < ctx.seq_len && i < 64; i++)
                printf(" %u", ctx.sequence[i].id);
            printf("\n");

            /* Decode sequence to text */
            char out_buf[512];
            uint32_t id_buf[64];
            uint32_t nd = ctx.seq_len < 64 ? ctx.seq_len : 64;
            for (uint32_t ii = 0; ii < nd; ii++) id_buf[ii] = ctx.sequence[ii].id;
            decode_ids(&ctx, id_buf, (int)nd, out_buf, 512);
            printf("\n  Generated text:\n");
            printf("  \"%s\"\n", out_buf);
        } else {
            printf("  ERROR: Inference failed\n");
        }

    } else if (mode == RIN_MODE_MLP && action == 3) {
        /* --- MLP SAMPLE MODE --- */
        for (int i = 0; i < 784; i++) input[i] = (uint32_t)(i % 10);
        double t0 = now_us();
        RIN_Inference(&ctx, input, 784, 1, &result);
        double t1 = now_us();

        printf("  Mode:       MLP + Sample\n");
        printf("  Latency:    %.0f us\n", t1 - t0);
        printf("  Predicted:  %u\n", result.tokens[0].id);

        printf("\n  Generating %d tokens via RIN_GenerateToken:\n", gen_count);
        for (int i = 0; i < gen_count; i++) {
            RIN_Token tok;
            if (RIN_GenerateToken(&ctx, &tok) == RIN_STATUS_OK) {
                printf("  [%d] token=%u prob=%.3f\n", i, tok.id, tok.probability);
            }
        }
        printf("\n  Sequence:");
        for (uint32_t i = 0; i < ctx.seq_len; i++)
            printf(" %u", ctx.sequence[i].id);
        printf("\n");

    } else {
        /* --- MLP / SNN / ATTN / THOR CLASSIFICATION --- */
        for (int i = 0; i < 784; i++) input[i] = (uint32_t)(i % 10);
        double t0 = now_us();
        RIN_Inference(&ctx, input, 784, 1, &result);
        double t1 = now_us();

        printf("  Mode:       %s\n", mode_name(mode));
        printf("  Latency:    %.0f us\n", t1 - t0);
        printf("  Predicted:  %u\n", result.tokens[0].id);
    }

    RIN_Destroy(&ctx);
    return 0;
}
