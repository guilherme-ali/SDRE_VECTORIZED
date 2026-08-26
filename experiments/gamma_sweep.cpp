/**
 * VARREDURA DE γ NO SDA-SS (Exp. 3) — o SDA com shift único usa
 * γ=0.5 fixo (ponto médio de (0,1)), sem justificativa própria além de
 * "não piora nos casos testados" (ver AutoLQR.cpp, comentário corrigido em
 * 2026-08-18 e docs/auditoria_solvers_riccati.md, Seção 15). Chu, Fan & Lin
 * (2005) propõem uma busca de Fibonacci para o γ ótimo — não implementada
 * aqui. Este experimento mede, em vez de assumir, se algum γ domina.
 *
 * γ agora é configurável via AutoLQR::setSDASSGamma() (antes, constante
 * hardcoded em dois pontos do .cpp) — só para permitir esta varredura sem
 * recompilar cinco vezes.
 *
 * Varre γ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × {SDA_SS, SDA_SS_FIXED}, sob o
 * critério casado τ=1e-3/200 iterações (Frobenius relativa, Seção 14), nas
 * 6 trajetórias, 300 pontos/trajetória (STRIDE=38, igual a tolerance_sweep).
 *
 * Saída CSV pelo serial (921600 baud):
 *   EXP,3
 *   RUN,3,gamma,traj,k,metodo,time_us,iters,residuo_dare,outcome
 *   SUMMARY,3,gamma,metodo,mean_us,std_us,mean_iters,n_converged,n_budget,n_breakdown,n_total
 *   outcome: 0=converged 1=budget(censurado, NAO falha numerica) 2=breakdown
 *
 * Consumidor: python/analisa_gamma.py (a criar após esta captura).
 */

#include <Arduino.h>
#include <AutoLQR.h>
#include <math.h>
#include <esp_timer.h>
#include "MatrixOperations.h"
#include "Trajectories.h"

namespace GammaSweep {

static const int N = 6, M = 3;

// ===== Parâmetros físicos reais (idênticos a test/tolerance_sweep.cpp) =====
const float Ixx = 42.95e-6f, Iyy = 37.77e-6f, Izz = 76.15e-6f, Ir = 1.02e-7f;
const float L_ARM = 0.060f * 0.70710678f;
const float MOTOR_B = 2.98e-8f, MOTOR_D = 0.05f * MOTOR_B;
const float MAX_RPM = 26423.0f;
const float MAX_OMEGA = (MAX_RPM * 2.0f * (float)M_PI) / 60.0f;
static const float DT = Trajectories::DT; // fonte unica: lib/Trajectories/Trajectories.h

const float roll_max = 45.0f * DEG_TO_RAD, pitch_max = 45.0f * DEG_TO_RAD, yaw_max = 90.0f * DEG_TO_RAD;
const float p_max = 300.0f * DEG_TO_RAD, q_max = 300.0f * DEG_TO_RAD, r_max = 200.0f * DEG_TO_RAD;
const float Q_11 = 1.0f / (roll_max * roll_max);
const float Q_22 = 1.0f / (pitch_max * pitch_max);
const float Q_33 = 1.0f / (yaw_max * yaw_max);
const float Q_44 = 1.0f / (p_max * p_max);
const float Q_55 = 1.0f / (q_max * q_max);
const float Q_66 = 1.0f / (r_max * r_max);

const float perc_tau_max = 0.5f;
const float max_tau_roll = 2.0f * MOTOR_B * L_ARM * MAX_OMEGA * MAX_OMEGA * perc_tau_max;
const float max_tau_pitch = max_tau_roll;
const float max_tau_yaw = 4.0f * MOTOR_D * MAX_OMEGA * MAX_OMEGA * perc_tau_max;
const float R_11 = 1.0f / (max_tau_roll * max_tau_roll);
const float R_22 = 1.0f / (max_tau_pitch * max_tau_pitch);
const float R_33 = 1.0f / (max_tau_yaw * max_tau_yaw);

float Ad[N * N], Bd[N * M], Qd[N * N], Rd[M * M];

static void updateSystemMatrix(float roll, float pitch, float p, float q, float r) {
    const float omega_r = 0.0f;
    const float inv_Ixx = 1.0f / Ixx, inv_Iyy = 1.0f / Iyy, inv_Izz = 1.0f / Izz;
    const float Iyy_Izz_over_Ixx = (Iyy - Izz) / Ixx;
    const float Izz_Ixx_over_Iyy = (Izz - Ixx) / Iyy;
    const float Ixx_Iyy_over_Izz = (Ixx - Iyy) / Izz;
    const float Ir_over_Ixx = Ir / Ixx;
    const float Ir_over_Iyy = Ir / Iyy;
    const float dt = DT, dt2_2 = dt * dt * 0.5f;

    const float sR = sinf(roll), cR = cosf(roll);
    const float sP = sinf(pitch), cP = cosf(pitch);
    const float inv_cP = 1.0f / cP;
    const float tP = sP * inv_cP;

    const float A03 = 1.0f, A04 = sR * tP, A05 = cR * tP;
    const float A14 = cR, A15 = -sR;
    const float A24 = sR * inv_cP, A25 = cR * inv_cP;
    const float A34 = Iyy_Izz_over_Ixx * r - Ir_over_Ixx * omega_r;
    const float A43 = Ir_over_Iyy * omega_r;
    const float A45 = Izz_Ixx_over_Iyy * p;
    const float A54 = Ixx_Iyy_over_Izz * p;

    const float A2_03 = A04 * A43, A2_04 = A34 + A05 * A54, A2_05 = A04 * A45;
    const float A2_13 = A14 * A43, A2_14 = A15 * A54, A2_15 = A14 * A45;
    const float A2_23 = A24 * A43, A2_24 = A25 * A54, A2_25 = A24 * A45;
    const float A2_33 = A34 * A43, A2_35 = A34 * A45, A2_44 = A43 * A34 + A45 * A54;
    const float A2_53 = A54 * A43, A2_55 = A54 * A45;

    memset(Ad, 0, sizeof(Ad));
    Ad[0 * N + 0] = 1.0f; Ad[1 * N + 1] = 1.0f; Ad[2 * N + 2] = 1.0f;
    Ad[3 * N + 3] = 1.0f + A2_33 * dt2_2;
    Ad[4 * N + 4] = 1.0f + A2_44 * dt2_2;
    Ad[5 * N + 5] = 1.0f + A2_55 * dt2_2;
    Ad[0 * N + 3] = A03 * dt + A2_03 * dt2_2;
    Ad[0 * N + 4] = A04 * dt + A2_04 * dt2_2;
    Ad[0 * N + 5] = A05 * dt + A2_05 * dt2_2;
    Ad[1 * N + 3] = A2_13 * dt2_2;
    Ad[1 * N + 4] = A14 * dt + A2_14 * dt2_2;
    Ad[1 * N + 5] = A15 * dt + A2_15 * dt2_2;
    Ad[2 * N + 3] = A2_23 * dt2_2;
    Ad[2 * N + 4] = A24 * dt + A2_24 * dt2_2;
    Ad[2 * N + 5] = A25 * dt + A2_25 * dt2_2;
    Ad[3 * N + 4] = A34 * dt;
    Ad[3 * N + 5] = A2_35 * dt2_2;
    Ad[4 * N + 3] = A43 * dt;
    Ad[4 * N + 5] = A45 * dt;
    Ad[5 * N + 3] = A2_53 * dt2_2;
    Ad[5 * N + 4] = A54 * dt;

    memset(Bd, 0, sizeof(Bd));
    Bd[0 * M + 0] = A03 * inv_Ixx * dt2_2;
    Bd[0 * M + 1] = A04 * inv_Iyy * dt2_2;
    Bd[0 * M + 2] = A05 * inv_Izz * dt2_2;
    Bd[1 * M + 1] = A14 * inv_Iyy * dt2_2;
    Bd[1 * M + 2] = A15 * inv_Izz * dt2_2;
    Bd[2 * M + 1] = A24 * inv_Iyy * dt2_2;
    Bd[2 * M + 2] = A25 * inv_Izz * dt2_2;
    Bd[3 * M + 0] = inv_Ixx * dt;
    Bd[3 * M + 1] = A34 * inv_Iyy * dt2_2;
    Bd[4 * M + 0] = A43 * inv_Ixx * dt2_2;
    Bd[4 * M + 1] = inv_Iyy * dt;
    Bd[4 * M + 2] = A45 * inv_Izz * dt2_2;
    Bd[5 * M + 1] = A54 * inv_Iyy * dt2_2;
    Bd[5 * M + 2] = inv_Izz * dt;

    memset(Qd, 0, sizeof(Qd));
    Qd[0 * N + 0] = Q_11 * dt; Qd[1 * N + 1] = Q_22 * dt; Qd[2 * N + 2] = Q_33 * dt;
    Qd[3 * N + 3] = Q_44 * dt; Qd[4 * N + 4] = Q_55 * dt; Qd[5 * N + 5] = Q_66 * dt;
    const float q03 = Q_11 * A03 * dt2_2, q04 = Q_11 * A04 * dt2_2, q05 = Q_11 * A05 * dt2_2;
    const float q14 = Q_22 * A14 * dt2_2, q15 = Q_22 * A15 * dt2_2;
    const float q24 = Q_33 * A24 * dt2_2, q25 = Q_33 * A25 * dt2_2;
    const float q34 = (A43 * Q_55 + Q_44 * A34) * dt2_2;
    const float q45 = (A54 * Q_66 + Q_55 * A45) * dt2_2;
    Qd[0 * N + 3] = q03; Qd[3 * N + 0] = q03;
    Qd[0 * N + 4] = q04; Qd[4 * N + 0] = q04;
    Qd[0 * N + 5] = q05; Qd[5 * N + 0] = q05;
    Qd[1 * N + 4] = q14; Qd[4 * N + 1] = q14;
    Qd[1 * N + 5] = q15; Qd[5 * N + 1] = q15;
    Qd[2 * N + 4] = q24; Qd[4 * N + 2] = q24;
    Qd[2 * N + 5] = q25; Qd[5 * N + 2] = q25;
    Qd[3 * N + 4] = q34; Qd[4 * N + 3] = q34;
    Qd[4 * N + 5] = q45; Qd[5 * N + 4] = q45;

    const float dt3_over_3 = dt * dt * dt / 3.0f;
    memset(Rd, 0, sizeof(Rd));
    Rd[0 * M + 0] = R_11 * dt + (Q_44 * inv_Ixx * inv_Ixx) * dt3_over_3;
    Rd[1 * M + 1] = R_22 * dt + (Q_55 * inv_Iyy * inv_Iyy) * dt3_over_3;
    Rd[2 * M + 2] = R_33 * dt + (Q_66 * inv_Izz * inv_Izz) * dt3_over_3;
}

// ---------------------------------------------------------------------------
// Trajetórias — vindas de lib/Trajectories/Trajectories.h (fonte única, seis
// trajetórias, espelhada em python/trajetorias.py).
//
// MUDANÇA DE DADOS: a cópia local de T4 deste arquivo usava uma APROXIMAÇÃO
// de regime permanente; foi descartada em favor da recursão exata do header —
// os pontos T4 do Exp. 3 mudam em relação às capturas anteriores, o que é
// esperado e correto (ver o cabeçalho de lib/Trajectories/Trajectories.h e o
// plano da campanha estendida, item 0.2).
// ---------------------------------------------------------------------------
static const int N_POINTS_FULL = Trajectories::N_POINTS_FULL; // 11538
static const int N_TRAJ = Trajectories::N_TRAJ;               // 6

// ---------------------------------------------------------------------------
// Exp. 3 — γ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, τ=1e-3/200 iters (critério casado),
// 300 pts/trajetória (STRIDE=38, igual a Exp. 0a).
// ---------------------------------------------------------------------------
static const int N_GAMMA = 5;
static const float GAMMAS[N_GAMMA] = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
static const float MATCHED_TOL = 1e-3f;
static const int MATCHED_MAXITER = 200;

static const int N_METHODS = 2;
static const char* METHODS[N_METHODS] = {"SDA_SS", "SDA_SS_FIXED"};
AutoLQR lqr[N_METHODS] = {AutoLQR(N, M), AutoLQR(N, M)};

static const int STRIDE = 38; // 11538/38 ~= 300 pontos por trajetoria
static const int N_PER_TRAJ = (N_POINTS_FULL + STRIDE - 1) / STRIDE;

struct GammaStats {
    int n_total = 0, n_converged = 0, n_budget = 0, n_breakdown = 0;
    double mean_time = 0, M2_time = 0;
    double mean_iters = 0, M2_iters = 0;
    double mean_resid = 0, M2_resid = 0;
    void add(double time_us, double iters, double resid, AutoLQR::SolveOutcome outcome) {
        n_total++;
        if (outcome == AutoLQR::SolveOutcome::Budget) { n_budget++; return; }
        if (outcome == AutoLQR::SolveOutcome::Breakdown) { n_breakdown++; return; }
        n_converged++;
        auto upd = [&](double val, double& mean, double& M2) {
            double delta = val - mean; mean += delta / n_converged;
            double delta2 = val - mean; M2 += delta * delta2;
        };
        upd(time_us, mean_time, M2_time);
        upd(iters, mean_iters, M2_iters);
        upd(resid, mean_resid, M2_resid);
    }
    double std_time() const { return n_converged > 0 ? sqrt(M2_time / n_converged) : 0.0; }
};

GammaStats stats[N_GAMMA][N_METHODS];

void runExp3() {
    Serial.println("EXP,3");
    Serial.println("# SDA-SS: varredura de gamma, criterio casado tau=1e-3/200 iters, 300 pts/traj");
    Serial.println("# RUN,3,gamma,traj,k,metodo,time_us,iters,residuo_dare,outcome");
    Serial.println("# SUMMARY,3,gamma,metodo,mean_us,std_us,mean_iters,mean_resid,n_converged,n_budget,n_breakdown,n_total");

    for (int m = 0; m < N_METHODS; m++) lqr[m].setStoppingCriterion(MATCHED_TOL, MATCHED_MAXITER);

    unsigned long t0 = millis();
    for (int ig = 0; ig < N_GAMMA; ig++) {
        float gamma = GAMMAS[ig];
        for (int m = 0; m < N_METHODS; m++) lqr[m].setSDASSGamma(gamma);

        for (int traj = 0; traj < N_TRAJ; traj++) {
            for (int kk = 0; kk < N_PER_TRAJ; kk++) {
                int k = kk * STRIDE;
                if (k >= N_POINTS_FULL) break;
                float t_traj, phi, theta, psi, p, q, r;
                Trajectories::trajectoryPoint((Trajectories::TrajId)traj, k,
                                              t_traj, phi, theta, psi, p, q, r);
                updateSystemMatrix(phi, theta, p, q, r);

                for (int m = 0; m < N_METHODS; m++) {
                    lqr[m].setStateMatrix(Ad);
                    lqr[m].setInputMatrix(Bd);
                    lqr[m].setCostMatrices(Qd, Rd);

                    int64_t t_start = esp_timer_get_time();
                    lqr[m].computeGains(METHODS[m]);
                    int64_t dt_us = esp_timer_get_time() - t_start;
                    int iters = lqr[m].getLastIterations();
                    float resid = lqr[m].getLastResidual();
                    AutoLQR::SolveOutcome outcome = lqr[m].getLastOutcome();

                    stats[ig][m].add((double)dt_us, (double)iters, (double)resid, outcome);
                    Serial.printf("RUN,3,%.1f,%s,%d,%s,%lld,%d,%.6e,%d\n",
                                  gamma, Trajectories::TRAJ_NAMES[traj], k, METHODS[m],
                                  (long long)dt_us, iters, resid, (int)outcome);
                }
                yield();
            }
        }

        for (int m = 0; m < N_METHODS; m++) {
            GammaStats& s = stats[ig][m];
            Serial.printf("SUMMARY,3,%.1f,%s,%.2f,%.2f,%.3f,%.6e,%d,%d,%d,%d\n",
                          gamma, METHODS[m], s.mean_time, s.std_time(), s.mean_iters, s.mean_resid,
                          s.n_converged, s.n_budget, s.n_breakdown, s.n_total);
        }
        unsigned long elapsed_s = (millis() - t0) / 1000;
        Serial.printf("# progresso 3: gamma=%.1f (%d/%d) elapsed=%lus\n", gamma, ig + 1, N_GAMMA, elapsed_s);
    }
    Serial.printf("# FIM EXP 3 — %lu s\n", (millis() - t0) / 1000);
}

void run() {
    Serial.begin(921600);
    unsigned long t_serial = millis();
    while (!Serial && millis() - t_serial < 3000) {}
    delay(1500);

    Serial.println("# VARREDURA DE GAMMA (Exp. 3) -- ver cabecalho do arquivo");
    unsigned long global_t0 = millis();

    runExp3();

    Serial.printf("# FIM DA VARREDURA DE GAMMA — tempo total decorrido: %lu s\n",
                  (millis() - global_t0) / 1000);
}

} // namespace GammaSweep

void setup() { GammaSweep::run(); }
void loop() { delay(5000); }
