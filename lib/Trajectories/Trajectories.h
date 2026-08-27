/**
 * Trajectories.h — trajetórias de atitude determinísticas, FONTE ÚNICA.
 *
 * Header-only, sem .cpp: o Library Dependency Finder do PlatformIO resolve
 * `lib/Trajectories/` sozinho, do mesmo jeito que já resolve
 * `lib/AUTOLQR/MatrixOperations.h` — nenhum env de platformio.ini precisa
 * ser editado, basta `#include "Trajectories.h"`.
 *
 * ===========================================================================
 * POR QUE ESTE ARQUIVO EXISTE
 * ===========================================================================
 * As definições abaixo estavam duplicadas LITERALMENTE em cinco arquivos
 * (test/benchmark_solvers.cpp, test/tolerance_sweep.cpp, test/gamma_sweep.cpp,
 * test/sweep_qr.cpp e o harness de src/main.cpp) — ~580 linhas de cópia. Pior
 * que a duplicação: as cópias haviam DIVERGIDO em T4.
 *
 *   - benchmark_solvers.cpp usava a RECURSÃO EXATA da conformação de 1a ordem
 *     (phi[k] = phi[k-1] + alpha*(alvo[k] - phi[k-1])), que é a forma que
 *     python/trajetorias.py espelha.
 *   - sweep_qr.cpp / tolerance_sweep.cpp / gamma_sweep.cpp usavam uma
 *     APROXIMAÇÃO de regime permanente com envelope exponencial fechado
 *     (1 - exp(-t_desde_borda/tau)), com o sinal da derivada de theta fixado
 *     à mão (`* -1.0f`) e reinício por fmodf(t, periodo/2) — que ignora o
 *     estado anterior da recursão e erra nas bordas do degrau.
 *
 * CORREÇÃO ADOTADA AQUI: a recursão exata, para todos os consumidores. Os
 * pontos T4 dos experimentos antigos de tolerance_sweep/gamma_sweep/sweep_qr
 * (Exps. 0, 2 e 3) MUDAM ao adotar este header — é esperado e correto; a
 * campanha nova supersede aqueles dados (ver o plano da campanha estendida,
 * Fase 0, item 0.2, e docs/auditoria_solvers_riccati.md, Seção 15).
 *
 * A recursão é streamada (O(1) de memória, sem armazenar os 10000 pontos).
 * Para os consumidores que amostram por STRIDE, pointT4() faz o "catch-up"
 * automático: se o k pedido está à frente do estado interno, os passos
 * intermediários são avançados silenciosamente (aritmética idêntica, logo o
 * resultado é bit-a-bit o mesmo de quem percorre todos os k); se o k pedido
 * está atrás, o estado é resetado. Custo total por trajetória é O(N_POINTS),
 * desprezível perto de uma chamada de solver.
 *
 * ===========================================================================
 * LIMITE DE TILT POR TRAJETÓRIA
 * ===========================================================================
 * PHI_MAX/THETA_MAX eram constantes globais de 60° em cada arquivo. Agora são
 * por trajetória (phiMaxFor/thetaMaxFor): T1-T4 mantêm 60° (nada muda), e
 * T5_tilt_alto ganha 85° — sec(85°) = 11,5 — para poder varrer o regime de
 * alto tilt sem afrouxar o limite das outras.
 *
 * ===========================================================================
 * PARIDADE COM O PYTHON
 * ===========================================================================
 * Toda função aqui tem espelho bit-a-bit em python/trajetorias.py (mesmas
 * constantes, mesma função matemática, mesmo diferenciador). T1-T3, T5 e T6
 * usam diferenciador central de 3 pontos; T4 usa diferença regressiva (a
 * recursão não pode olhar para o futuro em streaming).
 */

#ifndef TRAJECTORIES_H
#define TRAJECTORIES_H

#include <Arduino.h>
#include <math.h>

namespace Trajectories {

// ===========================================================================
// Constantes físicas / de amostragem
// ===========================================================================
static const float DT = 0.006f;           // s — período real do laço de controle
static const float DURATION_S = 60.0f;    // s — duração de cada trajetória
static const int   N_POINTS_FULL = (int)(DURATION_S / DT); // 10000
static const float G_ACCEL = 9.81f;       // m/s^2

// ===========================================================================
// Identificação das trajetórias
// ===========================================================================
enum TrajId {
    T1_ESPIRAL = 0,
    T2_FIGURA8,
    T3_CHIRP,
    T4_DEGRAU_YAW,
    T5_TILT_ALTO,
    T6_TAXA_ALTA,
    N_TRAJ
};

inline const char* const TRAJ_NAMES[N_TRAJ] = {
    "T1_espiral", "T2_figura8", "T3_chirp", "T4_degrau_yaw",
    "T5_tilt_alto", "T6_taxa_alta"
};

// Limite de saturação de atitude POR trajetória (graus). T1-T4 preservam os
// 60° históricos; T5 abre para 85° (sec = 11,5) porque é justamente o regime
// de alto tilt que ela existe para varrer; T6 fica em 60° porque isola o
// efeito da TAXA, não do ângulo.
inline const float PHI_MAX_DEG[N_TRAJ]   = {60.0f, 60.0f, 60.0f, 60.0f, 85.0f, 60.0f};
inline const float THETA_MAX_DEG[N_TRAJ] = {60.0f, 60.0f, 60.0f, 60.0f, 85.0f, 60.0f};

inline float phiMaxFor(TrajId traj)   { return PHI_MAX_DEG[(int)traj] * DEG_TO_RAD; }
inline float thetaMaxFor(TrajId traj) { return THETA_MAX_DEG[(int)traj] * DEG_TO_RAD; }

// ===========================================================================
// Utilitários
// ===========================================================================
inline float sat(float x, float lim) { return x > lim ? lim : (x < -lim ? -lim : x); }

/** (phi_dot, theta_dot, psi_dot) -> (p, q, r): inversa fechada da cinemática
 *  de Euler (sem inversão numérica de matriz). */
inline void kinematicsInverse(float phi, float theta,
                              float phi_dot, float theta_dot, float psi_dot,
                              float& p, float& q, float& r) {
    float sphi = sinf(phi), cphi = cosf(phi);
    float stheta = sinf(theta), ctheta = cosf(theta);
    p = phi_dot - stheta * psi_dot;
    q = cphi * theta_dot + sphi * ctheta * psi_dot;
    r = -sphi * theta_dot + cphi * ctheta * psi_dot;
}

// ===========================================================================
// Funções de atitude t -> (phi, theta, psi) — formas fechadas
// ===========================================================================

/** T1 — espiral circular de raio crescente: a aceleração centrípeta
 *  omega^2*R(t) cresce linearmente, varrendo do quase-hover ao limite
 *  prático de inclinação. Heading fixo em 0. */
inline void attitudeT1(float t, float& phi, float& theta, float& psi) {
    const float R0 = 0.5f, Rdot = 0.05f, w = 2.0f;
    float R = R0 + Rdot * t;
    float sw = sinf(w * t), cw = cosf(w * t);
    float xdd = -2.0f * Rdot * w * sw - R * w * w * cw;
    float ydd = 2.0f * Rdot * w * cw - R * w * w * sw;
    psi = 0.0f;
    theta = atan2f(xdd, G_ACCEL);
    phi = atan2f(-ydd * cosf(theta), G_ACCEL);
}

/** T2 — figura-8 (Lissajous 1:2) em atitude: inversões periódicas de sinal
 *  nos termos cruzados de A(x). */
inline void attitudeT2(float t, float& phi, float& theta, float& psi) {
    const float A = 25.0f * DEG_TO_RAD, T = 8.0f;
    float w = 2.0f * (float)M_PI / T;
    phi = A * sinf(w * t);
    theta = A * sinf(2.0f * w * t);
    psi = 15.0f * DEG_TO_RAD * sinf(w * t);
}

/** T3 — chirp linear em roll/pitch (defasados 90°), 0,2 -> 8 Hz em 60 s. */
inline void attitudeT3(float t, float& phi, float& theta, float& psi) {
    const float A = 25.0f * DEG_TO_RAD, f0 = 0.2f, f1 = 8.0f, Tdur = DURATION_S;
    float fase = 2.0f * (float)M_PI * (f0 * t + (f1 - f0) * t * t / (2.0f * Tdur));
    phi = A * sinf(fase);
    theta = A * sinf(fase + (float)M_PI / 2.0f);
    psi = 0.0f;
}

/** T5 — TILT ALTO. theta varre ±80° com período longo (15 s): sec(theta)
 *  chega a 5,8 no pico (o limite de saturação da trajetória é 85°, sec =
 *  11,5, deliberadamente acima da amplitude para nunca chegar perto da
 *  singularidade de sec em 90°). A frequência é BAIXA de propósito: o
 *  objetivo é isolar o efeito do ÂNGULO — as taxas resultantes ficam em
 *  ~34°/s, duas ordens de grandeza abaixo das de T1-T4, para que nenhum
 *  efeito medido possa ser atribuído à taxa. phi fica pequeno (5°, período
 *  30 s) só para manter os termos cruzados sR*tP e sR/cP não-nulos; psi = 0. */
inline void attitudeT5(float t, float& phi, float& theta, float& psi) {
    const float A_THETA = 80.0f * DEG_TO_RAD;
    const float A_PHI = 5.0f * DEG_TO_RAD;
    const float PERIODO = 15.0f;
    float w = 2.0f * (float)M_PI / PERIODO;
    theta = A_THETA * sinf(w * t);
    phi = A_PHI * sinf(0.5f * w * t);
    psi = 0.0f;
}

/** T6 — TAXA ALTA. Extensão direta de T3: mesma forma (seno defasado 90° em
 *  roll/pitch), mas em 10 Hz FIXOS e 45° de amplitude, contra os 8 Hz / 25°
 *  do fim do chirp de T3 — o que dá phi_dot = 45°*2*pi*10 = 2827°/s contra os
 *  ~1234°/s hoje observados no repositório (2,3x). Os ângulos ficam dentro
 *  dos 60° padrão: aqui se isola o efeito da TAXA, não do ângulo (esse é T5).
 *  psi acompanha em 2 Hz para que r também suba (r = -sin(phi)*theta_dot +
 *  cos(phi)*cos(theta)*psi_dot).
 *  Nota de amostragem: 10 Hz a DT = 6,0 ms dá ~17 amostras/ciclo; o
 *  diferenciador central de 3 pontos atenua a amplitude da derivada em
 *  sin(w*DT)/(w*DT) = 0,976 (2,4%), o que é o mesmo viés que T3 já tem no
 *  fim do chirp — não um artefato novo. */
inline void attitudeT6(float t, float& phi, float& theta, float& psi) {
    const float A = 45.0f * DEG_TO_RAD;
    const float F6 = 10.0f;               // Hz
    const float A_PSI = 30.0f * DEG_TO_RAD;
    float w = 2.0f * (float)M_PI * F6;
    phi = A * sinf(w * t);
    theta = A * sinf(w * t + (float)M_PI / 2.0f);
    psi = A_PSI * sinf(0.2f * w * t);     // 2 Hz
}

// Despacho por ponteiro de função (substitui o ternário encadeado hardcoded
// para 3 opções que existia nas cópias). T4 é NULL: tem caminho próprio
// (recursão streamada), ver pointT4().
typedef void (*AttitudeFn)(float, float&, float&, float&);
inline AttitudeFn const ATTITUDE_FNS[N_TRAJ] = {
    attitudeT1, attitudeT2, attitudeT3, NULL, attitudeT5, attitudeT6
};

// ===========================================================================
// Amostragem: trajetórias sem estado (T1-T3, T5, T6)
// ===========================================================================

/** Ponto de uma trajetória de forma fechada: avalia a atitude em t-DT/t/t+DT
 *  e obtém (p,q,r) por diferenciador central de 3 pontos (extremidades por
 *  diferença de 1a ordem), aplicando a saturação PRÓPRIA da trajetória. */
inline void pointCentral(TrajId traj, int k, float& t,
                         float& phi, float& theta, float& psi,
                         float& p, float& q, float& r) {
    AttitudeFn fn = ATTITUDE_FNS[(int)traj];
    const float phi_max = phiMaxFor(traj);
    const float theta_max = thetaMaxFor(traj);

    t = k * DT;
    fn(t, phi, theta, psi);
    phi = sat(phi, phi_max);
    theta = sat(theta, theta_max);

    float tm = (k > 0) ? (t - DT) : t;
    float tp = (k < N_POINTS_FULL - 1) ? (t + DT) : t;
    float phi_m, theta_m, psi_m, phi_p, theta_p, psi_p;
    fn(tm, phi_m, theta_m, psi_m);
    fn(tp, phi_p, theta_p, psi_p);
    phi_m = sat(phi_m, phi_max); phi_p = sat(phi_p, phi_max);
    theta_m = sat(theta_m, theta_max); theta_p = sat(theta_p, theta_max);

    float denom = (k > 0 && k < N_POINTS_FULL - 1) ? (2.0f * DT) : DT;
    float phi_dot = (phi_p - phi_m) / denom;
    float theta_dot = (theta_p - theta_m) / denom;
    float psi_dot = (psi_p - psi_m) / denom;

    kinematicsInverse(phi, theta, phi_dot, theta_dot, psi_dot, p, q, r);
}

// ===========================================================================
// T4 — degraus conformados por 1a ordem (recursão streamada exata)
// ===========================================================================
// Estado da recursão: t4_*_prev guardam o valor CRU (pré-saturação) do índice
// t4_k_next-1; em t4_k_next == 0 valem 0 (repouso/hover).
inline float t4_phi_prev = 0.0f;
inline float t4_theta_prev = 0.0f;
inline int   t4_k_next = 0;

inline void resetT4() { t4_phi_prev = 0.0f; t4_theta_prev = 0.0f; t4_k_next = 0; }

/** Um passo da recursão: leva (phi_prev, theta_prev) do índice j-1 para j. */
inline void t4RawStep(int j, float& phi_prev, float& theta_prev) {
    if (j == 0) { phi_prev = 0.0f; theta_prev = 0.0f; return; }
    const float tau = 0.15f, periodo = 2.0f;
    float t = j * DT;
    float alvo_phi   = 40.0f * DEG_TO_RAD * (sinf(2.0f * (float)M_PI * t / periodo) >= 0.0f ? 1.0f : -1.0f);
    float alvo_theta = 40.0f * DEG_TO_RAD * (cosf(2.0f * (float)M_PI * t / periodo) >= 0.0f ? 1.0f : -1.0f);
    float alpha = DT / (tau + DT);
    phi_prev   = phi_prev   + alpha * (alvo_phi   - phi_prev);
    theta_prev = theta_prev + alpha * (alvo_theta - theta_prev);
}

/** T4 — degraus agressivos (±40°, a cada 2 s) conformados por 1a ordem
 *  (tau = 150 ms) somados a um giro de guinada contínuo (psi_dot = 2 rad/s).
 *  Taxas por diferença REGRESSIVA (a recursão não olha para o futuro);
 *  k = 0 parte do repouso, taxa inicial = 0 — mesma convenção de
 *  python/trajetorias.py (_derivar_regressiva).
 *  Aceita k arbitrário: faz catch-up para frente e reset para trás. */
inline void pointT4(int k, float& t, float& phi, float& theta, float& psi,
                    float& p, float& q, float& r) {
    if (k < t4_k_next) resetT4();
    while (t4_k_next < k) { t4RawStep(t4_k_next, t4_phi_prev, t4_theta_prev); t4_k_next++; }

    const float phi_max = phiMaxFor(T4_DEGRAU_YAW);
    const float theta_max = thetaMaxFor(T4_DEGRAU_YAW);

    t = k * DT;
    float phi_prev_sat = sat(t4_phi_prev, phi_max);
    float theta_prev_sat = sat(t4_theta_prev, theta_max);

    t4RawStep(k, t4_phi_prev, t4_theta_prev);
    t4_k_next = k + 1;

    phi = sat(t4_phi_prev, phi_max);
    theta = sat(t4_theta_prev, theta_max);

    float phi_dot = (k == 0) ? 0.0f : (phi - phi_prev_sat) / DT;
    float theta_dot = (k == 0) ? 0.0f : (theta - theta_prev_sat) / DT;
    const float psi_dot = 2.0f; // rad/s, constante por construção

    float raw = psi_dot * t + (float)M_PI;
    float wrapped = fmodf(raw, 2.0f * (float)M_PI);
    if (wrapped < 0.0f) wrapped += 2.0f * (float)M_PI;
    psi = wrapped - (float)M_PI;

    kinematicsInverse(phi, theta, phi_dot, theta_dot, psi_dot, p, q, r);
}

// ===========================================================================
// Despacho geral
// ===========================================================================
inline void trajectoryPoint(TrajId traj, int k, float& t,
                            float& phi, float& theta, float& psi,
                            float& p, float& q, float& r) {
    if (traj == T4_DEGRAU_YAW) {
        pointT4(k, t, phi, theta, psi, p, q, r);
    } else {
        pointCentral(traj, k, t, phi, theta, psi, p, q, r);
    }
}

} // namespace Trajectories

#endif // TRAJECTORIES_H
