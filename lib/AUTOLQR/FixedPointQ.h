#ifndef FIXED_POINT_Q_H
#define FIXED_POINT_Q_H

#include <stdint.h>

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

q_t qmul(q_t a, q_t b, int sh, Status* st);
q_t qdiv(q_t a, q_t b, int sh, Status* st);

void matmul_q(const q_t* a, const q_t* b, q_t* c, int r1, int c1, int c2, int sh, Status* st);
void transpose_q(const q_t* a, q_t* at, int r, int c);
void add_q(const q_t* a, const q_t* b, q_t* c, int n);
void sub_q(const q_t* a, const q_t* b, q_t* c, int n);

// Inversão Gauss-Jordan com pivotamento parcial, em Q-format. n até 12 (serve
// tanto às matrizes 6x6 do laço quanto ao pencil 12x12 do setup do SDA_SS).
bool invert_q(const q_t* src, q_t* dst, int n, int sh, Status* st);

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
bool doubling_loop_q(q_t* Ak, q_t* Gk, q_t* Hk, int n, int sh, Variant variant,
                      int maxIterations, int invRelTolerance,
                      float* cum_s_out, Status* st);

} // namespace fxq

#endif
