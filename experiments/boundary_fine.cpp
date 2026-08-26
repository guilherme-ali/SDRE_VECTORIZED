/**
 * MAPA FINO DAS DUAS FRONTEIRAS DE FALHA (Exp. B da campanha estendida) —
 * o Exp. A (test/tol_qr_sweep.cpp) mapeou as duas fronteiras em grade
 * grossa (13 décadas de R_scale, 1e-6 a 1e6) e confirmou o limiar analítico
 * da fronteira superior em 144,5 com precisão de uma década (0% em 1e2,
 * 100% em 1e3). Este experimento refina a grade perto das duas transições:
 * 25 pontos log-espaçados em [1e-3, 1e1] (fronteira inferior, gradual —
 * overflow de G0=B*R^-1*B^T no setup, Seção 16.2 da auditoria) e 15 pontos
 * em [50, 500] (fronteira superior, nítida — overflow de entrada de Rd,
 * limiar teórico 144,5). Objetivo: dar uma regra de projeto a priori mais
 * precisa que "década tal ou qual", e verificar se a fronteira superior é
 * de fato uma função degrau (0%→100% num intervalo estreito em torno de
 * 144,5) como o modelo analítico prevê, ou se há uma zona de transição.
 *
 * Estrutura idêntica a test/sweep_qr.cpp (mesmos 10 métodos — 5 doubling
 * float + 5 fixed; ITERATIVE/ITERATIVE_FIXED fora, não são o alvo desta
 * pergunta), τ=1e-3/200 fixo (não há eixo de tolerância aqui, só R_scale ×
 * Q_rate_scale refinados perto das fronteiras).
 *
 * Saída CSV pelo serial:
 *   RUN,r_scale,q_rate_scale,traj,k,metodo,time_us,iters,residuo_dare,outcome
 *   SUMMARY,r_scale,q_rate_scale,metodo,n_converged,n_budget,n_breakdown,count
 *
 * Consumidor: python/analisa_boundary_fine.py (a criar após esta captura).
 */

#include <Arduino.h>
#include <AutoLQR.h>
#include <math.h>
#include <esp_timer.h>
#include "MatrixOperations.h"
#include "Trajectories.h"

namespace BoundaryFine {

static const int N = 6, M = 3;

const float Ixx = 42.95e-6f, Iyy = 37.77e-6f, Izz = 76.15e-6f, Ir = 1.02e-7f;
const float L_ARM = 0.060f * 0.70710678f;
const float MOTOR_B = 2.98e-8f, MOTOR_D = 0.05f * MOTOR_B;
const float MAX_RPM = 26423.0f;
const float MAX_OMEGA = (MAX_RPM * 2.0f * (float)M_PI) / 60.0f;
static const float DT = Trajectories::DT;

const float roll_max = 45.0f * DEG_TO_RAD, pitch_max = 45.0f * DEG_TO_RAD, yaw_max = 90.0f * DEG_TO_RAD;
const float p_max = 300.0f * DEG_TO_RAD, q_max = 300.0f * DEG_TO_RAD, r_max = 200.0f * DEG_TO_RAD;
const float Q_11_nom = 1.0f / (roll_max * roll_max);
const float Q_22_nom = 1.0f / (pitch_max * pitch_max);
const float Q_33_nom = 1.0f / (yaw_max * yaw_max);
const float Q_44_nom = 1.0f / (p_max * p_max);
const float Q_55_nom = 1.0f / (q_max * q_max);
const float Q_66_nom = 1.0f / (r_max * r_max);

const float perc_tau_max = 0.5f;
const float max_tau_roll = 2.0f * MOTOR_B * L_ARM * MAX_OMEGA * MAX_OMEGA * perc_tau_max;
const float max_tau_pitch = max_tau_roll;
const float max_tau_yaw = 4.0f * MOTOR_D * MAX_OMEGA * MAX_OMEGA * perc_tau_max;
const float R_11_nom = 1.0f / (max_tau_roll * max_tau_roll);
const float R_22_nom = 1.0f / (max_tau_pitch * max_tau_pitch);
const float R_33_nom = 1.0f / (max_tau_yaw * max_tau_yaw);

float Ad[N * N], Bd[N * M], Qd[N * N], Rd[M * M];

static void updateSystemMatrix(float roll, float pitch, float p, float q, float r,
                                float r_scale, float q_rate_scale) {
    const float omega_r = 0.0f;
    const float Q_11 = Q_11_nom, Q_22 = Q_22_nom, Q_33 = Q_33_nom;
    const float Q_44 = Q_44_nom * q_rate_scale, Q_55 = Q_55_nom * q_rate_scale, Q_66 = Q_66_nom * q_rate_scale;
    const float R_11 = R_11_nom * r_scale, R_22 = R_22_nom * r_scale, R_33 = R_33_nom * r_scale;

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

static const int N_POINTS_FULL = Trajectories::N_POINTS_FULL; // 11538
static const int STRIDE = 231; // ~50 pontos/traj * 6 = ~300 pontos totais
static const int N_PER_TRAJ = (N_POINTS_FULL + STRIDE - 1) / STRIDE;
static const int N_TRAJ = Trajectories::N_TRAJ; // 6

// Grade fina: 25 pontos log-espacados em [1e-3,1e1] (fronteira inferior,
// gradual) + 15 pontos em [50,500] (fronteira superior, nitida perto de
// 144.5) -- ver docs/auditoria_solvers_riccati.md, Secoes 16.1/16.2/16.4.
static const int N_R = 40;
static const float R_SCALES[N_R] = {
    1.000000e-03f, 1.467799e-03f, 2.154435e-03f, 3.162278e-03f, 4.641589e-03f,
    6.812921e-03f, 1.000000e-02f, 1.467799e-02f, 2.154435e-02f, 3.162278e-02f,
    4.641589e-02f, 6.812921e-02f, 1.000000e-01f, 1.467799e-01f, 2.154435e-01f,
    3.162278e-01f, 4.641589e-01f, 6.812921e-01f, 1.000000e+00f, 1.467799e+00f,
    2.154435e+00f, 3.162278e+00f, 4.641589e+00f, 6.812921e+00f, 1.000000e+01f,
    5.000000e+01f, 5.893843e+01f, 6.947477e+01f, 8.189469e+01f, 9.653489e+01f,
    1.137923e+02f, 1.341348e+02f, 1.581139e+02f, 1.863797e+02f, 2.196985e+02f,
    2.589737e+02f, 3.052701e+02f, 3.598428e+02f, 4.241714e+02f, 5.000000e+02f
};
static const int N_QR = 5;
static const float QRATE_SCALES[N_QR] = {0.01f, 0.1f, 1.0f, 10.0f, 100.0f};

static const int N_METHODS = 10;
static const char* METHODS[N_METHODS] = {
    "SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA",
    "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"
};
AutoLQR lqr[N_METHODS] = {
    AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M),
    AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M),
};

#define BOUNDARY_REL_TOL 1e-3f
#define BOUNDARY_MAX_ITERS 200

struct GridStats {
    int n_converged = 0, n_budget = 0, n_breakdown = 0, count = 0;
    void reset() { n_converged = n_budget = n_breakdown = count = 0; }
};
static GridStats stats[N_METHODS];

void run() {
    Serial.begin(921600);
    unsigned long t_serial = millis();
    while (!Serial && millis() - t_serial < 3000) {}
    delay(1500);

    for (int m = 0; m < N_METHODS; m++) lqr[m].setStoppingCriterion(BOUNDARY_REL_TOL, BOUNDARY_MAX_ITERS);

    Serial.println("# MAPA FINO DAS FRONTEIRAS DE FALHA (Exp. B) -- ver cabecalho do arquivo");
    Serial.println("# RUN,r_scale,q_rate_scale,traj,k,metodo,time_us,iters,residuo_dare,outcome");
    Serial.println("# SUMMARY,r_scale,q_rate_scale,metodo,n_converged,n_budget,n_breakdown,count");
    Serial.printf("# N_R=%d N_QR=%d N_TRAJ=%d N_PER_TRAJ=%d N_METHODS=%d REL_TOL=%g MAX_ITERS=%d\n",
                  N_R, N_QR, N_TRAJ, N_PER_TRAJ, N_METHODS, (double)BOUNDARY_REL_TOL, BOUNDARY_MAX_ITERS);

    unsigned long global_t0 = millis();
    long combo = 0, total_combos = (long)N_R * N_QR;

    for (int ir = 0; ir < N_R; ir++) {
        for (int iq = 0; iq < N_QR; iq++) {
            combo++;
            float r_scale = R_SCALES[ir], q_rate_scale = QRATE_SCALES[iq];
            for (int m = 0; m < N_METHODS; m++) stats[m].reset();

            for (int traj = 0; traj < N_TRAJ; traj++) {
                for (int kk = 0; kk < N_PER_TRAJ; kk++) {
                    int k = kk * STRIDE;
                    if (k >= N_POINTS_FULL) break;
                    float t_traj, phi, theta, psi, p, q, r;
                    Trajectories::trajectoryPoint((Trajectories::TrajId)traj, k,
                                                  t_traj, phi, theta, psi, p, q, r);
                    updateSystemMatrix(phi, theta, p, q, r, r_scale, q_rate_scale);

                    for (int m = 0; m < N_METHODS; m++) {
                        lqr[m].setStateMatrix(Ad);
                        lqr[m].setInputMatrix(Bd);
                        lqr[m].setCostMatrices(Qd, Rd);

                        int64_t t0 = esp_timer_get_time();
                        lqr[m].computeGains(METHODS[m]);
                        int64_t dt_us = esp_timer_get_time() - t0;
                        int iters = lqr[m].getLastIterations();
                        float resid = lqr[m].getLastResidual();
                        AutoLQR::SolveOutcome outcome = lqr[m].getLastOutcome();

                        GridStats& s = stats[m];
                        s.count++;
                        if (outcome == AutoLQR::SolveOutcome::Budget) s.n_budget++;
                        else if (outcome == AutoLQR::SolveOutcome::Breakdown) s.n_breakdown++;
                        else s.n_converged++;

                        Serial.printf("RUN,%.4e,%.0e,%s,%d,%s,%lld,%d,%.6e,%d\n",
                                      r_scale, q_rate_scale, Trajectories::TRAJ_NAMES[traj], k, METHODS[m],
                                      (long long)dt_us, iters, resid, (int)outcome);
                    }
                    yield();
                }
            }

            for (int m = 0; m < N_METHODS; m++) {
                GridStats& s = stats[m];
                Serial.printf("SUMMARY,%.4e,%.0e,%s,%d,%d,%d,%d\n",
                              r_scale, q_rate_scale, METHODS[m],
                              s.n_converged, s.n_budget, s.n_breakdown, s.count);
            }

            unsigned long elapsed_s = (millis() - global_t0) / 1000;
            Serial.printf("# progresso combo=%ld/%ld (r_scale=%.4e q_rate_scale=%.0e) elapsed=%lus\n",
                          combo, total_combos, r_scale, q_rate_scale, elapsed_s);
        }
    }

    unsigned long total_s = (millis() - global_t0) / 1000;
    Serial.printf("# FIM DO MAPA FINO — tempo total decorrido: %lu s\n", total_s);
}

} // namespace BoundaryFine

void setup() { BoundaryFine::run(); }
void loop() { delay(5000); }
