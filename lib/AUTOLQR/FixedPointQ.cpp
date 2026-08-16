#include "FixedPointQ.h"
#include <math.h>
#include <string.h>

namespace fxq {

q_t f2q(float x, int sh, Status* st) {
    double v = (double)x * (double)(1u << sh);
    if (v > 2147483647.0) { st->overflow = true; return INT32_MAX; }
    if (v < -2147483648.0) { st->overflow = true; return INT32_MIN; }
    return (q_t)llround(v);
}

float q2f(q_t x, int sh) {
    return (float)x / (float)(1u << sh);
}

q_t requant(q_t x, int shFrom, int shTo, Status* st) {
    int d = shTo - shFrom;
    if (d == 0) return x;
    if (d > 0) {
        int64_t r = (int64_t)x << d;
        if (r > INT32_MAX) { st->overflow = true; return INT32_MAX; }
        if (r < INT32_MIN) { st->overflow = true; return INT32_MIN; }
        return (q_t)r;
    }
    // d < 0: perde resolução ao arredondar para o shift menor (mesma
    // convenção de truncamento dos demais operadores deste kernel).
    return (q_t)(x >> (-d));
}

// matmul_q/transpose_q/add_q/sub_q agora `inline` em FixedPointQ.h (ver
// comentário lá — recuperar o inlining perdido em 13b33bb).

FXQ_FAST_ATTR bool invert_q(const q_t* src, q_t* dst, int n, int sh, Status* st) {
    q_t aug[12 * 24]; // n até 12 (matrizes 6x6 do laço e o pencil 12x12 do setup do SDA_SS)
    const int n2 = 2 * n;
    const q_t one = f2q(1.0f, sh, st);
    const q_t pivot_floor = f2q(1e-4f, sh, st); // piso > resolução do shift em uso
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * n2 + j] = src[i * n + j];
        for (int j = 0; j < n; j++) aug[i * n2 + n + j] = (i == j) ? one : 0;
    }
    for (int i = 0; i < n; i++) {
        q_t maxv = aug[i * n2 + i]; if (maxv < 0) maxv = -maxv;
        int mr = i;
        for (int k = i + 1; k < n; k++) {
            q_t v = aug[k * n2 + i]; if (v < 0) v = -v;
            if (v > maxv) { maxv = v; mr = k; }
        }
        if (maxv < pivot_floor) return false; // ~singular no domínio fixed-point
        if (mr != i)
            for (int j = 0; j < n2; j++) { q_t t = aug[i * n2 + j]; aug[i * n2 + j] = aug[mr * n2 + j]; aug[mr * n2 + j] = t; }
        // Recíproco do pivô calculado UMA vez e multiplicado nas n2 colunas,
        // em vez de n2 divisões — o Xtensa LX7 não tem divisão de 64 bits em
        // hardware (fase 2 da otimização, ver docs/auditoria_solvers_riccati.md
        // Seção 12; MUDA o arredondamento — inv_piv perde a resolução de
        // qdiv(x,piv) direto, então este resultado NÃO é mais bit-a-bit igual
        // ao de antes; revalidado por tolerância contra o scipy, não por
        // igualdade exata).
        q_t piv = aug[i * n2 + i];
        q_t inv_piv = qdiv(one, piv, sh, st);
        for (int j = 0; j < n2; j++) aug[i * n2 + j] = qmul(aug[i * n2 + j], inv_piv, sh, st);
        for (int k = 0; k < n; k++)
            if (k != i) {
                q_t f = aug[k * n2 + i];
                if (f != 0)
                    for (int j = 0; j < n2; j++) aug[k * n2 + j] -= qmul(f, aug[i * n2 + j], sh, st);
            }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) dst[i * n + j] = aug[i * n2 + n + j];
    return true;
}

// ----------------------------------------------------------------------------
// Laço de duplicação. n<=6 assumido (buffers de 36 elementos).
// ----------------------------------------------------------------------------
FXQ_FAST_ATTR bool doubling_loop_q(q_t* Ak, q_t* Gk, q_t* Hk, int n, int sh, Variant variant,
                      int maxIterations, int invRelTolerance,
                      float* cum_s_out, Status* st) {
    const int nn = n * n;
    const q_t one = f2q(1.0f, sh, st);
    float cum_s = 1.0f;
    st->iterations = maxIterations;

    q_t Akn[36], Gkn[36], Hkn[36];
    q_t AT[36], V[36], W[36], T1[36], T2[36], T3[36];

    for (int it = 0; it < maxIterations; it++) {
        if (variant == Variant::AdaptiveScaling) {
            // s = sqrt(||H||/||G||), saturado em [0.1,10], aplicado a G e H.
            // A norma e o fator s em si são calculados em float (poucas
            // chamadas de sqrtf por execução, desprezível frente às 8 matmuls
            // por iteração) — só os dados de Ak/Gk/Hk permanecem inteiros.
            int64_t sumG = 0, sumH = 0;
            for (int i = 0; i < nn; i++) {
                sumG += (int64_t)Gk[i] * (int64_t)Gk[i];
                sumH += (int64_t)Hk[i] * (int64_t)Hk[i];
            }
            float scale = 1.0f / (float)(1LL << sh);
            float normG = sqrtf((float)sumG) * scale;
            float normH = sqrtf((float)sumH) * scale;
            float s_f = 1.0f;
            if (normG > 1e-10f && normH > 1e-10f) {
                s_f = sqrtf(normH / normG);
                s_f = fminf(fmaxf(s_f, 0.1f), 10.0f);
            }
            q_t s_q = f2q(s_f, sh, st);
            for (int i = 0; i < nn; i++) {
                Gk[i] = qmul(Gk[i], s_q, sh, st);
                Hk[i] = qdiv(Hk[i], s_q, sh, st);
            }
            cum_s *= s_f;
        }

        if (variant == Variant::AlternatingVW) {
            // V = (I + Gk·Hk)^-1 ; W = (I + Hk·Gk)^-1 — duas inversões, para
            // medir o efeito da ordem de multiplicação sob quantização (ver
            // docs/auditoria_solvers_riccati.md; em float as duas formas são
            // idênticas por push-through, em Q13.18 não são bit-a-bit iguais).
            matmul_q(Gk, Hk, T1, n, n, n, sh, st);
            for (int i = 0; i < n; i++) T1[i * n + i] += one;
            if (!invert_q(T1, V, n, sh, st)) return false;

            matmul_q(Hk, Gk, T1, n, n, n, sh, st);
            for (int i = 0; i < n; i++) T1[i * n + i] += one;
            if (!invert_q(T1, W, n, sh, st)) return false;

            matmul_q(Ak, V, T1, n, n, n, sh, st);      // T1 = Ak·V
            matmul_q(T1, Ak, Akn, n, n, n, sh, st);    // Akn = Ak·V·Ak

            transpose_q(Ak, AT, n, n);
            matmul_q(Gk, AT, T2, n, n, n, sh, st);     // T2 = Gk·Ak'
            matmul_q(T1, T2, T3, n, n, n, sh, st);     // T3 = Ak·V·Gk·Ak'
            add_q(Gk, T3, Gkn, nn);

            matmul_q(W, Hk, T2, n, n, n, sh, st);      // T2 = W·Hk
            matmul_q(T2, Ak, T3, n, n, n, sh, st);     // T3 = W·Hk·Ak
            matmul_q(AT, T3, T2, n, n, n, sh, st);     // T2 = Ak'·W·Hk·Ak
            add_q(Hk, T2, Hkn, nn);
        } else {
            // Standard (também usado, com o rescale acima, por AdaptiveScaling):
            // V = (I + Gk·Hk)^-1, mesma inversa para as três atualizações.
            matmul_q(Gk, Hk, T1, n, n, n, sh, st);
            for (int i = 0; i < n; i++) T1[i * n + i] += one;
            if (!invert_q(T1, V, n, sh, st)) return false;

            matmul_q(Ak, V, T1, n, n, n, sh, st);      // T1 = Ak·V
            matmul_q(T1, Ak, Akn, n, n, n, sh, st);    // Akn = Ak·V·Ak

            transpose_q(Ak, AT, n, n);
            matmul_q(Gk, AT, T2, n, n, n, sh, st);     // T2 = Gk·Ak'
            matmul_q(T1, T2, T3, n, n, n, sh, st);     // T3 = Ak·V·Gk·Ak'
            add_q(Gk, T3, Gkn, nn);

            matmul_q(V, Ak, T2, n, n, n, sh, st);      // T2 = V·Ak
            matmul_q(Hk, T2, T3, n, n, n, sh, st);     // T3 = Hk·V·Ak
            matmul_q(AT, T3, T2, n, n, n, sh, st);     // T2 = Ak'·Hk·V·Ak
            add_q(Hk, T2, Hkn, nn);
        }

        // Convergência: max|ΔHk| * invRelTolerance < max|Hk| (piso ~3.8e-6 em Q13.18)
        q_t dmax = 0, hmax = 0;
        for (int i = 0; i < nn; i++) {
            q_t d = Hkn[i] - Hk[i]; if (d < 0) d = -d;
            q_t h = Hk[i];          if (h < 0) h = -h;
            if (d > dmax) dmax = d;
            if (h > hmax) hmax = h;
        }
        memcpy(Ak, Akn, nn * sizeof(q_t));
        memcpy(Gk, Gkn, nn * sizeof(q_t));
        memcpy(Hk, Hkn, nn * sizeof(q_t));
        if ((int64_t)dmax * invRelTolerance < (int64_t)hmax) { st->iterations = it + 1; break; }
    }

    if (cum_s_out) *cum_s_out = cum_s;
    return true;
}

} // namespace fxq
