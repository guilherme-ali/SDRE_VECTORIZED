#ifndef FIXED_POINT_Q_H
#define FIXED_POINT_Q_H

#include <stdint.h>
#include <Arduino.h> // p/ IRAM_ATTR (garante o macro definido mesmo se FixedPointQ.cpp
                      // for compilado sem AutoLQR.h/ArduinoEigen.h por perto)

// Mesmo padrão de MatrixOperations.h:7-15 — evita cache-miss de instrução na
// flash para o código que fica no laço quente (regressão de ~15% do
// SDA_FIXED após 13b33bb; ver docs/auditoria_solvers_riccati.md, Seção 12).
#if defined(ESP32)
    #define FXQ_FAST_ATTR IRAM_ATTR
#else
    #define FXQ_FAST_ATTR
#endif

// ============================================================================
// Kernel de ponto fixo Q-format para os solvers de Riccati em ESP32-S2 (sem
// FPU). Extraído do SDA_FIXED original (ver docs/auditoria_solvers_riccati.md)
// e generalizado para servir aos cinco métodos de doubling — o laço de
// duplicação é idêntico entre eles; só o setup de (A0,G0,H0), um passo
// opcional dentro do laço, e a extração de P mudam por método (ficam em
// AutoLQR.cpp, junto do par float correspondente, para leitura lado a lado).
// ============================================================================
namespace fxq {

typedef int32_t q_t;
constexpr int Q_SHIFT_DEFAULT = 18; // Q13.18: 13 bits inteiros (±8192), 18 fracionários (res. 3.8e-6)

// Telemetria/estado por chamada — substitui os globais g_q_ovf/g_q_iters do
// SDA_FIXED original (não reentrantes, escondiam qual método saturou).
struct Status {
    bool overflow = false;   // saturação/estouro em qualquer conversão ou produto
    int  iterations = 0;     // iterações efetivamente executadas até convergir
    q_t  max_abs_seen = 0;   // maior |valor| visto em qualquer matmul_q (para dimensionar o shift)
    float rel_step = 1.0f;   // ‖ΔH‖_F/‖H_k‖_F da última iteração executada (ver doubling_loop_q)
    bool  bit_exact_zero = false; // true se Hkn==Hk elemento-a-elemento (int32) na última iteração
};

enum class Variant {
    Standard,        // SDA, SDA_SS, SDA_SCALED — setup/extração diferem, laço igual
    AdaptiveScaling, // ASDA — rescale (G,H)->(sG,H/s) a cada iteração, acumula cum_s
    AlternatingVW    // ADDA — V=(I+GH)^-1 e W=(I+HG)^-1 separados (2 inversões/iter)
};

// Conversões e aritmética Q-format
q_t   f2q(float x, int sh, Status* st);
float q2f(q_t x, int sh);
q_t   requant(q_t x, int shFrom, int shTo, Status* st); // converte entre shifts (ex.: Q9.22 -> Q13.18)

// qmul/qdiv em `inline` no header (não em FixedPointQ.cpp): são chamadas
// dentro do laço de eliminação de invert_q (até n*(2n) vezes por chamada) e,
// como funções externas, perderam o inlining que tinham quando o kernel
// vivia dentro do namespace anônimo de AutoLQR.cpp — parte da regressão de
// ~15% medida no SDA_FIXED após 13b33bb (ver docs/auditoria_solvers_riccati.md,
// Seção 10). `inline` no header não depende de LTO para ser recuperado.
inline q_t qmul(q_t a, q_t b, int sh, Status* st) {
    int64_t r = ((int64_t)a * (int64_t)b) >> sh;
    if (r > INT32_MAX) { st->overflow = true; return INT32_MAX; }
    if (r < INT32_MIN) { st->overflow = true; return INT32_MIN; }
    return (q_t)r;
}

inline q_t qdiv(q_t a, q_t b, int sh, Status* st) {
    if (b == 0) { st->overflow = true; return (a >= 0) ? INT32_MAX : INT32_MIN; }
    int64_t r = ((int64_t)a << sh) / b; // estoura se b minúsculo -> clamp + flag
    if (r > INT32_MAX) { st->overflow = true; return INT32_MAX; }
    if (r < INT32_MIN) { st->overflow = true; return INT32_MIN; }
    return (q_t)r;
}

// matmul_q: `inline` + `__restrict__`, mesmo tratamento que
// MatrixOperations::matrixMultiply já tem no lado float — e um atalho 6x6x6
// desenrolado, que é a forma que domina o laço de duplicação (8 das 8 matmuls
// por iteração usam n=6). A ordem de acumulação do atalho é IDÊNTICA à do
// laço genérico (mesmo `for k`, mesmo int64_t acc, mesmo `>> sh`) — resultado
// bit-a-bit igual, só sem overhead de chamada/loop.
// SEM FXQ_FAST_ATTR aqui de propósito: como é inlinado dentro de
// doubling_loop_q (que já tem IRAM_ATTR), aplicar o atributo também na
// função inlinada duplica a diretiva de seção e o linker Xtensa rejeita com
// "dangerous relocation: l32r: literal placed after use" — o código já vai
// para IRAM via o `IRAM_ATTR` da função externa que o inclui.
inline void matmul_q(const q_t* __restrict__ a, const q_t* __restrict__ b,
                      q_t* __restrict__ c, int r1, int c1, int c2, int sh, Status* st) {
    if (r1 == 6 && c1 == 6 && c2 == 6) {
        for (int i = 0; i < 6; i++) {
            const q_t* ai = a + i * 6;
            for (int j = 0; j < 6; j++) {
                int64_t acc = (int64_t)ai[0] * b[0 * 6 + j] + (int64_t)ai[1] * b[1 * 6 + j]
                             + (int64_t)ai[2] * b[2 * 6 + j] + (int64_t)ai[3] * b[3 * 6 + j]
                             + (int64_t)ai[4] * b[4 * 6 + j] + (int64_t)ai[5] * b[5 * 6 + j];
                int64_t r = acc >> sh;
                if (r > INT32_MAX)      { st->overflow = true; r = INT32_MAX; }
                else if (r < INT32_MIN) { st->overflow = true; r = INT32_MIN; }
                q_t v = (q_t)r;
                c[i * 6 + j] = v;
#ifdef FXQ_INSTRUMENT
                q_t av = (v < 0) ? -v : v;
                if (av > st->max_abs_seen) st->max_abs_seen = av;
#endif
            }
        }
        return;
    }
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            int64_t acc = 0;
            for (int k = 0; k < c1; k++) acc += (int64_t)a[i * c1 + k] * (int64_t)b[k * c2 + j];
            int64_t r = acc >> sh;
            if (r > INT32_MAX)      { st->overflow = true; r = INT32_MAX; }
            else if (r < INT32_MIN) { st->overflow = true; r = INT32_MIN; }
            q_t v = (q_t)r;
            c[i * c2 + j] = v;
#ifdef FXQ_INSTRUMENT
            q_t av = (v < 0) ? -v : v;
            if (av > st->max_abs_seen) st->max_abs_seen = av;
#endif
        }
    }
}

// (idem: sem FXQ_FAST_ATTR — inlinadas dentro de funções que já o têm)
inline void transpose_q(const q_t* __restrict__ a, q_t* __restrict__ at, int r, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) at[j * r + i] = a[i * c + j];
}

inline void add_q(const q_t* __restrict__ a, const q_t* __restrict__ b,
                   q_t* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

inline void sub_q(const q_t* __restrict__ a, const q_t* __restrict__ b,
                   q_t* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] - b[i];
}

// Inversão Gauss-Jordan com pivotamento parcial, em Q-format. n até 12 (serve
// tanto às matrizes 6x6 do laço quanto ao pencil 12x12 do setup do SDA_SS).
// Definida em FixedPointQ.cpp (grande demais para inline manual sem duplicar
// ~50 linhas por unidade de tradução) — só o atributo de flash é aplicado aqui.
FXQ_FAST_ATTR bool invert_q(const q_t* src, q_t* dst, int n, int sh, Status* st);

// Laço de duplicação estrutural compartilhado entre SDA, SDA_SS, SDA_SCALED,
// ASDA e ADDA. Ak,Gk,Hk são n×n, atualizados in-place (assume n<=6 — todos os
// métodos fixed-point desta lib são gate n==6,m==3, buffers internos
// dimensionados para 36 elementos). Retorna false em singular (não em
// overflow — checar st->overflow separadamente após a chamada, mesma
// convenção do SDA_FIXED original).
//
// cum_s_out: só usado em Variant::AdaptiveScaling — produto acumulado dos
// fatores de reescalonamento, calculado em float (o produto de ~10 fatores
// em [0,1;10] pode exceder a faixa representável em Q13.18). A extração
// P = H_final * cum_s deve ser feita pelo chamador, fora do laço.
FXQ_FAST_ATTR bool doubling_loop_q(q_t* Ak, q_t* Gk, q_t* Hk, int n, int sh, Variant variant,
                      int maxIterations, int invRelTolerance,
                      float* cum_s_out, Status* st);

} // namespace fxq

#endif
