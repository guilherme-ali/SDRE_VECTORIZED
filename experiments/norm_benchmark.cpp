#include <Arduino.h>
#include "AutoLQR.h"
#include "FixedPointQ.h"
#include <math.h>

#if defined(ESP32)
#include <esp_cpu.h>
#endif

// Microbenchmark do custo da norma de Frobenius e da aritmética de ponto fixo
// Mede ciclos de CPU exatos e microssegundos no ESP32-S2.

using namespace fxq;

volatile float g_sink_float = 0.0f;
volatile uint32_t g_sink_u32 = 0;
volatile q_t g_sink_q = 0;

static inline uint32_t get_ccount() {
    return esp_cpu_get_ccount();
}

static inline float q2f_mul(q_t x, float inv_scale) {
    return (float)x * inv_scale;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("=================================================");
    Serial.println("MICROBENCHMARK: NORMA DE FROBENIUS & KERNEL Q13.18");
    Serial.println("=================================================");

    const int n = 6;
    const int nn = 36;
    const int sh = 18;
    const float inv_scale = 1.0f / (float)(1 << sh);
    const int N_REPS = 2000;

    Status st;
    q_t Hk[36], Hkn[36], Ak[36], Gk[36];
    for (int i = 0; i < nn; i++) {
        Hk[i] = f2q(0.44f + 0.01f * (float)(i % 5), sh, &st);
        Hkn[i] = Hk[i] + f2q(0.001f * (float)(i % 3), sh, &st);
        Ak[i] = f2q(0.9f - 0.02f * (float)i, sh, &st);
        Gk[i] = f2q(0.1f + 0.01f * (float)(i % 4), sh, &st);
    }

    // 1. Norma Atual (com divisao em q2f)
    uint32_t c_start = get_ccount();
    float sum_relF_1 = 0.0f;
    for (int r = 0; r < N_REPS; r++) {
        float diffSq = 0.0f, hSq = 0.0f;
        for (int i = 0; i < nn; i++) {
            float d = q2f(Hkn[i], sh) - q2f(Hk[i], sh);
            float h = q2f(Hk[i], sh);
            diffSq += d * d;
            hSq += h * h;
        }
        float relF = (hSq > 1e-20f) ? sqrtf(diffSq / hSq) : sqrtf(diffSq);
        sum_relF_1 += relF;
    }
    uint32_t c_norm_div = get_ccount() - c_start;
    g_sink_float = sum_relF_1;

    // 2. Apenas os 72 chamadas de q2f() com divisao
    c_start = get_ccount();
    float sum_q2f = 0.0f;
    for (int r = 0; r < N_REPS; r++) {
        for (int i = 0; i < nn; i++) {
            sum_q2f += q2f(Hkn[i], sh);
            sum_q2f += q2f(Hk[i], sh);
        }
    }
    uint32_t c_72_q2f_div = get_ccount() - c_start;
    g_sink_float = sum_q2f;

    // 3. sqrtf isolado (soft-float)
    c_start = get_ccount();
    float sum_sqrt = 0.0f;
    for (int r = 0; r < N_REPS; r++) {
        sum_sqrt += sqrtf(0.00012345f + (float)r * 1e-7f);
    }
    uint32_t c_sqrtf = get_ccount() - c_start;
    g_sink_float = sum_sqrt;

    // 4. Norma Otimizada (com multiplicacao por reciproca constante)
    c_start = get_ccount();
    float sum_relF_2 = 0.0f;
    for (int r = 0; r < N_REPS; r++) {
        float diffSq = 0.0f, hSq = 0.0f;
        for (int i = 0; i < nn; i++) {
            float d = q2f_mul(Hkn[i], inv_scale) - q2f_mul(Hk[i], inv_scale);
            float h = q2f_mul(Hk[i], inv_scale);
            diffSq += d * d;
            hSq += h * h;
        }
        float relF = (hSq > 1e-20f) ? sqrtf(diffSq / hSq) : sqrtf(diffSq);
        sum_relF_2 += relF;
    }
    uint32_t c_norm_mul = get_ccount() - c_start;
    g_sink_float = sum_relF_2;

    // 5. Norma 100% inteira em int64_t
    c_start = get_ccount();
    uint32_t sum_int = 0;
    for (int r = 0; r < N_REPS; r++) {
        int64_t diffSq = 0, hSq = 0;
        for (int i = 0; i < nn; i++) {
            int32_t d = Hkn[i] - Hk[i];
            int32_t h = Hk[i];
            diffSq += (int64_t)d * (int64_t)d;
            hSq += (int64_t)h * (int64_t)h;
        }
        // diffSq / hSq < tol^2 -> diffSq * (1/tol^2) < hSq
        bool conv = (diffSq * 1000000LL < hSq);
        if (conv) sum_int++;
    }
    uint32_t c_norm_int = get_ccount() - c_start;
    g_sink_u32 = sum_int;

    // 6. 1 Iteracao pura de SDA-fx (8 matmuls 6x6 + 1 invert 6x6, sem norma)
    q_t Akn_buf[36], Gkn_buf[36], Hkn_buf[36];
    q_t AT_buf[36], V_buf[36], T1_buf[36], T2_buf[36], T3_buf[36];
    const q_t one = f2q(1.0f, sh, &st);

    c_start = get_ccount();
    for (int r = 0; r < N_REPS; r++) {
        matmul_q(Gk, Hk, T1_buf, n, n, n, sh, &st);
        for (int i = 0; i < n; i++) T1_buf[i * n + i] += one;
        invert_q(T1_buf, V_buf, n, sh, &st);

        matmul_q(Ak, V_buf, T1_buf, n, n, n, sh, &st);
        matmul_q(T1_buf, Ak, Akn_buf, n, n, n, sh, &st);

        transpose_q(Ak, AT_buf, n, n);
        matmul_q(Gk, AT_buf, T2_buf, n, n, n, sh, &st);
        matmul_q(T1_buf, T2_buf, T3_buf, n, n, n, sh, &st);
        add_q(Gk, T3_buf, Gkn_buf, nn);

        matmul_q(V_buf, Ak, T2_buf, n, n, n, sh, &st);
        matmul_q(Hk, T2_buf, T3_buf, n, n, n, sh, &st);
        matmul_q(AT_buf, T3_buf, T2_buf, n, n, n, sh, &st);
        add_q(Hk, T2_buf, Hkn_buf, nn);
    }
    uint32_t c_iter_sda_pure = get_ccount() - c_start;
    g_sink_q = Hkn_buf[0];

    // 7. 1 Iteracao pura de ADDA-fx (8 matmuls + 2 invert 6x6, sem norma)
    q_t W_buf[36];
    c_start = get_ccount();
    for (int r = 0; r < N_REPS; r++) {
        matmul_q(Gk, Hk, T1_buf, n, n, n, sh, &st);
        for (int i = 0; i < n; i++) T1_buf[i * n + i] += one;
        invert_q(T1_buf, V_buf, n, sh, &st);

        matmul_q(Hk, Gk, T1_buf, n, n, n, sh, &st);
        for (int i = 0; i < n; i++) T1_buf[i * n + i] += one;
        invert_q(T1_buf, W_buf, n, sh, &st);

        matmul_q(Ak, V_buf, T1_buf, n, n, n, sh, &st);
        matmul_q(T1_buf, Ak, Akn_buf, n, n, n, sh, &st);

        transpose_q(Ak, AT_buf, n, n);
        matmul_q(Gk, AT_buf, T2_buf, n, n, n, sh, &st);
        matmul_q(T1_buf, T2_buf, T3_buf, n, n, n, sh, &st);
        add_q(Gk, T3_buf, Gkn_buf, nn);

        matmul_q(W_buf, Hk, T2_buf, n, n, n, sh, &st);
        matmul_q(T2_buf, Ak, T3_buf, n, n, n, sh, &st);
        matmul_q(AT_buf, T3_buf, T2_buf, n, n, n, sh, &st);
        add_q(Hk, T2_buf, Hkn_buf, nn);
    }
    uint32_t c_iter_adda_pure = get_ccount() - c_start;
    g_sink_q = Hkn_buf[0];

    // 8. 1 Iteracao pura de VI-fx (matmuls + 1 invert 3x3, sem norma)
    const int m = 3;
    q_t Bq[18], Rq[9], eps = f2q(1e-5f, sh, &st);
    for (int i = 0; i < 18; i++) Bq[i] = f2q(0.1f * (float)(i % 3), sh, &st);
    for (int i = 0; i < 9; i++) Rq[i] = f2q(1.0f + 0.1f * (float)i, sh, &st);
    q_t BT_buf[18];
    transpose_q(Bq, BT_buf, n, m);

    q_t PA[36], PB[18], ATPA[36], BTPB[9], BTPA[18], S[9], Sinv[9], Ktmp[18], ATPB[18], corr[36], Pnext[36];
    c_start = get_ccount();
    for (int r = 0; r < N_REPS; r++) {
        matmul_q(Hk, Ak, PA, n, n, n, sh, &st);
        matmul_q(Hk, Bq, PB, n, n, m, sh, &st);
        matmul_q(AT_buf, PA, ATPA, n, n, n, sh, &st);
        matmul_q(BT_buf, PB, BTPB, m, n, m, sh, &st);
        matmul_q(BT_buf, PA, BTPA, m, n, n, sh, &st);

        add_q(Rq, BTPB, S, m * m);
        for (int i = 0; i < m; i++) S[i * m + i] += eps;

        invert_q(S, Sinv, m, sh, &st);

        matmul_q(Sinv, BTPA, Ktmp, m, m, n, sh, &st);
        matmul_q(AT_buf, PB, ATPB, n, n, m, sh, &st);
        matmul_q(ATPB, Ktmp, corr, n, m, n, sh, &st);

        for (int i = 0; i < nn; i++) Pnext[i] = Hk[i] + ATPA[i] - corr[i];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                q_t avg = (Pnext[i * n + j] + Pnext[j * n + i]) / 2;
                Pnext[i * n + j] = avg;
                Pnext[j * n + i] = avg;
            }
    }
    uint32_t c_iter_vi_pure = get_ccount() - c_start;
    g_sink_q = Pnext[0];

    // 9. Heap vs Static buffer em float invert6x6
    c_start = get_ccount();
    for (int r = 0; r < N_REPS; r++) {
        float* buf = new float[72];
        buf[0] = (float)r;
        delete[] buf;
    }
    uint32_t c_heap = get_ccount() - c_start;

    // Frequencia da CPU
    float cpu_mhz = 240.0f;

    auto to_us = [cpu_mhz, N_REPS](uint32_t cycles) -> float {
        return (float)cycles / (cpu_mhz * (float)N_REPS);
    };
    auto to_cyc = [N_REPS](uint32_t cycles) -> float {
        return (float)cycles / (float)N_REPS;
    };

    float t_norm_div = to_us(c_norm_div);
    float t_72_q2f = to_us(c_72_q2f_div);
    float t_sqrtf = to_us(c_sqrtf);
    float t_norm_mul = to_us(c_norm_mul);
    float t_norm_int = to_us(c_norm_int);
    float t_sda_pure = to_us(c_iter_sda_pure);
    float t_adda_pure = to_us(c_iter_adda_pure);
    float t_vi_pure = to_us(c_iter_vi_pure);
    float t_heap = to_us(c_heap);

    Serial.println("\n--- RESULTADOS MICROBENCHMARK (ESP32-S2 @ 240MHz) ---");
    Serial.printf("N_REPS: %d\n", N_REPS);
    Serial.printf("1. Norma atual (soft-div q2f):    %7.2f us (%7.0f ciclos)\n", t_norm_div, to_cyc(c_norm_div));
    Serial.printf("   - so 72 chamadas q2f (div):    %7.2f us (%7.0f ciclos, %.1f cyc/div)\n", t_72_q2f, to_cyc(c_72_q2f_div), to_cyc(c_72_q2f_div)/72.0f);
    Serial.printf("   - 1x sqrtf isolado (soft):     %7.2f us (%7.0f ciclos)\n", t_sqrtf, to_cyc(c_sqrtf));
    Serial.printf("2. Norma otimizada (soft-mul):    %7.2f us (%7.0f ciclos)\n", t_norm_mul, to_cyc(c_norm_mul));
    Serial.printf("3. Norma inteira (int64_t):       %7.2f us (%7.0f ciclos)\n", t_norm_int, to_cyc(c_norm_int));
    Serial.println("--- ARITMETICA PURA POR ITERACAO (sem norma) ---");
    Serial.printf("4. SDA-fx iter pura:              %7.2f us (%7.0f ciclos)\n", t_sda_pure, to_cyc(c_iter_sda_pure));
    Serial.printf("5. ADDA-fx iter pura:             %7.2f us (%7.0f ciclos)\n", t_adda_pure, to_cyc(c_iter_adda_pure));
    Serial.printf("6. VI-fx iter pura:               %7.2f us (%7.0f ciclos)\n", t_vi_pure, to_cyc(c_iter_vi_pure));
    Serial.println("--- ITERACAO TOTAL (aritmetica pura + norma atual) ---");
    Serial.printf("SDA-fx total/iter:                %7.2f us (norma = %.1f%%)\n", t_sda_pure + t_norm_div, 100.0f * t_norm_div / (t_sda_pure + t_norm_div));
    Serial.printf("ADDA-fx total/iter:               %7.2f us (norma = %.1f%%)\n", t_adda_pure + t_norm_div, 100.0f * t_norm_div / (t_adda_pure + t_norm_div));
    Serial.printf("VI-fx total/iter:                 %7.2f us (norma = %.1f%%)\n", t_vi_pure + t_norm_div, 100.0f * t_norm_div / (t_vi_pure + t_norm_div));
    Serial.println("--- HEAP ALLOCATION (M-05) ---");
    Serial.printf("7. new float[72] + delete[]:      %7.2f us (%7.0f ciclos)\n", t_heap, to_cyc(c_heap));
    Serial.println("=================================================");
    Serial.println("FIM DO MICROBENCHMARK");
}

void loop() {
    delay(1000);
}
