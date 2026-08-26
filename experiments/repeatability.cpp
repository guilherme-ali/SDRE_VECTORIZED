/**
 * REPETIBILIDADE / JITTER (Exp. D da campanha estendida) — mede a variância
 * de tempo de execução no MESMO ponto de operação, repetido várias vezes
 * consecutivas. Até esta sessão, cada ponto da bateria principal e dos
 * sweeps foi medido exatamente 1 vez — a margem de tempo real alegada no
 * artigo (SDA-fx: 3,92+0,9=4,82ms médio, 4,05+0,9=4,95ms pior caso, contra
 * 5,2ms de período) assume implicitamente que o tempo de execução é
 * determinístico ponto a ponto. Este experimento testa essa suposição.
 *
 * Metodologia: ~2000 pontos de operação amostrados das 6 trajetórias de
 * lib/Trajectories/Trajectories.h (~333/traj), cada um com Ad/Bd/Qd/Rd
 * calculado UMA vez e então computeGains() chamado 20 vezes CONSECUTIVAS
 * sobre a MESMA entrada, para os 12 métodos, sob o critério casado
 * (τ=1e-3/200). Isso isola jitter de execução (cache, scheduling, contenção
 * de barramento) da variação ponto-a-ponto já caracterizada na bateria
 * principal.
 *
 * Nota sobre warm-start: para ITERATIVE/ITERATIVE_FIXED, repetir a MESMA
 * entrada 20x em sequência faz o warm-start convergir para P quase exato
 * já na 2a repetição (a 1a é a única "fria") — isso é esperado e não é um
 * artefato: mede o jitter do REGIME warm-started de fato usado em voo, não
 * o pior caso de partida fria (já caracterizado em outros experimentos).
 *
 * Saída CSV pelo serial (921600 baud):
 *   RUN,traj,k,metodo,rep,time_us,iters,residuo_dare,outcome
 *   SUMMARY,traj,k,metodo,mean_us,std_us,min_us,max_us,cv_pct,n_converged,n_total
 *   cv_pct = 100*std_us/mean_us (coeficiente de variação) -- métrica central
 *   deste experimento.
 *
 * Consumidor: python/analisa_repetibilidade.py (a criar após esta captura).
 */

#include <Arduino.h>
#include <AutoLQR.h>
#include <math.h>
#include <esp_timer.h>
#include "MatrixOperations.h"
#include "Trajectories.h"

namespace Repeatability {

static const int N = 6, M = 3;

// ===== Parâmetros físicos reais (idênticos aos demais benchmarks) =====
const float Ixx = 42.95e-6f, Iyy = 37.77e-6f, Izz = 76.15e-6f, Ir = 1.02e-7f;
const float L_ARM = 0.060f * 0.70710678f;
const float MOTOR_B = 2.98e-8f, MOTOR_D = 0.05f * MOTOR_B;
const float MAX_RPM = 26423.0f;
const float MAX_OMEGA = (MAX_RPM * 2.0f * (float)M_PI) / 60.0f;
static const float DT = Trajectories::DT;

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

static const int N_POINTS_FULL = Trajectories::N_POINTS_FULL; // 11538
static const int N_TRAJ = Trajectories::N_TRAJ;                // 6
static const int STRIDE = 35; // 11538/35 ~= 330 pontos/traj * 6 = ~1980 pontos totais
static const int N_PER_TRAJ = (N_POINTS_FULL + STRIDE - 1) / STRIDE;

static const int N_REPS = 20;
static const int N_METHODS = 12;
static const char* METHODS[N_METHODS] = {
    "SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA", "ITERATIVE",
    "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED", "ITERATIVE_FIXED"
};
AutoLQR lqr[N_METHODS] = {
    AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M),
    AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M),
};

void run() {
    Serial.begin(921600);
    unsigned long t_serial = millis();
    while (!Serial && millis() - t_serial < 3000) {}
    delay(1500);

    for (int m = 0; m < N_METHODS; m++) lqr[m].setStoppingCriterion(1e-3f, 200);

    Serial.println("# REPETIBILIDADE / JITTER (Exp. D) -- ver cabecalho do arquivo");
    Serial.println("# RUN,traj,k,metodo,rep,time_us,iters,residuo_dare,outcome");
    Serial.println("# SUMMARY,traj,k,metodo,mean_us,std_us,min_us,max_us,cv_pct,n_converged,n_total");
    Serial.printf("# N_TRAJ=%d N_PER_TRAJ=%d N_REPS=%d N_METHODS=%d REL_TOL=1e-3 MAX_ITERS=200\n",
                  N_TRAJ, N_PER_TRAJ, N_REPS, N_METHODS);

    unsigned long global_t0 = millis();
    long point_idx = 0, total_points = (long)N_TRAJ * N_PER_TRAJ;

    for (int traj = 0; traj < N_TRAJ; traj++) {
        for (int kk = 0; kk < N_PER_TRAJ; kk++) {
            int k = kk * STRIDE;
            if (k >= N_POINTS_FULL) break;
            point_idx++;
            float t_traj, phi, theta, psi, p, q, r;
            Trajectories::trajectoryPoint((Trajectories::TrajId)traj, k, t_traj, phi, theta, psi, p, q, r);
            updateSystemMatrix(phi, theta, p, q, r);

            for (int m = 0; m < N_METHODS; m++) {
                lqr[m].setStateMatrix(Ad);
                lqr[m].setInputMatrix(Bd);
                lqr[m].setCostMatrices(Qd, Rd);

                double sum_us = 0.0, sum_us2 = 0.0;
                int64_t min_us = INT64_MAX, max_us = 0;
                int n_conv = 0, n_total = 0;

                for (int rep = 0; rep < N_REPS; rep++) {
                    int64_t t0 = esp_timer_get_time();
                    lqr[m].computeGains(METHODS[m]);
                    int64_t dt_us = esp_timer_get_time() - t0;
                    int iters = lqr[m].getLastIterations();
                    float resid = lqr[m].getLastResidual();
                    AutoLQR::SolveOutcome outcome = lqr[m].getLastOutcome();

                    n_total++;
                    if (outcome == AutoLQR::SolveOutcome::Converged) {
                        n_conv++;
                        sum_us += (double)dt_us;
                        sum_us2 += (double)dt_us * (double)dt_us;
                        if (dt_us < min_us) min_us = dt_us;
                        if (dt_us > max_us) max_us = dt_us;
                    }

                    Serial.printf("RUN,%s,%d,%s,%d,%lld,%d,%.6e,%d\n",
                                  Trajectories::TRAJ_NAMES[traj], k, METHODS[m], rep,
                                  (long long)dt_us, iters, resid, (int)outcome);
                }

                double mean_us = n_conv > 0 ? sum_us / n_conv : 0.0;
                double var_us = n_conv > 1 ? (sum_us2 / n_conv - mean_us * mean_us) : 0.0;
                double std_us = var_us > 0 ? sqrt(var_us) : 0.0;
                double cv_pct = mean_us > 1e-9 ? 100.0 * std_us / mean_us : 0.0;
                Serial.printf("SUMMARY,%s,%d,%s,%.2f,%.2f,%lld,%lld,%.2f,%d,%d\n",
                              Trajectories::TRAJ_NAMES[traj], k, METHODS[m],
                              mean_us, std_us, (long long)(n_conv > 0 ? min_us : 0),
                              (long long)max_us, cv_pct, n_conv, n_total);
            }
            yield();

            if (point_idx % 200 == 0) {
                unsigned long elapsed_s = (millis() - global_t0) / 1000;
                Serial.printf("# progresso ponto=%ld/%ld traj=%s k=%d elapsed=%lus\n",
                              point_idx, total_points, Trajectories::TRAJ_NAMES[traj], k, elapsed_s);
            }
        }
    }

    unsigned long total_s = (millis() - global_t0) / 1000;
    Serial.printf("# FIM DA REPETIBILIDADE — tempo total decorrido: %lu s\n", total_s);
}

} // namespace Repeatability

void setup() { Repeatability::run(); }
void loop() { delay(5000); }
