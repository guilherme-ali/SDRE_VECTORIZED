/**
 * Verificação on-device (ESP32-S2 real) dos solvers corrigidos em lib/AUTOLQR.
 *
 * Reproduz exatamente o caso C1_hover do harness nativo
 * (test/native/verify_solvers.cpp): mesmos Ixx/Iyy/Izz/Ir, mesmo dt=5.2ms,
 * mesma regra de Bryson para Q/R, hover (roll=pitch=yaw=p=q=r=omega_r=0).
 * Roda todos os métodos, imprime K, resíduo real da DARE (getLastResidual,
 * corrigido nesta auditoria), iterações e tempo — para comparar contra
 * outputs/verify_host_baseline.csv (linha "C1_hover") e confirmar que o
 * soft-float / -O3 -ffast-math do S2 não introduz nenhuma divergência
 * qualitativa em relação ao host.
 *
 * Ver docs/auditoria_solvers_riccati.md.
 */

#include <Arduino.h>
#include <AutoLQR.h>
#include <math.h>
#include "MatrixOperations.h"

static const int N = 6, M = 3;

AutoLQR lqr(N, M);
AutoLQR lqr_iter_cold(N, M); // instância própria p/ isolar o cold-start do ITERATIVE_FIXED (P_warm zerado, sem herança da lqr acima)

float Ad[N * N], Bd[N * M], Qd[N * N], Rd[M * M];

static void buildC1Hover() {
    const float Ixx = 42.95e-6f, Iyy = 37.77e-6f, Izz = 76.15e-6f, Ir = 1.02e-7f;
    const float L_ARM = 0.060f * 0.70710678f;
    const float MOTOR_B = 2.98e-8f, MOTOR_D = 0.05f * MOTOR_B;
    const float MAX_RPM = 26423.0f;
    const float MAX_OMEGA = (MAX_RPM * 2.0f * (float)M_PI) / 60.0f;
    const float dt = 0.0052f;

    // Hover: roll=pitch=yaw=p=q=r=omega_r=0
    const float inv_Ixx = 1.0f / Ixx, inv_Iyy = 1.0f / Iyy, inv_Izz = 1.0f / Izz;

    float A[N * N] = {0};
    A[0 * N + 3] = 1.0f; // A03 = 1 (roll' = p em hover: sR*tP=0, cR*tP=0 -> só A03 sobra)
    A[1 * N + 4] = 1.0f; // A14 = cos(0) = 1
    A[2 * N + 5] = 1.0f; // A25 = cos(0)/cos(0) = 1
    // A34,A43,A45,A54 = 0 em hover (r=p=q=omega_r=0)

    float B[N * M] = {0};
    B[3 * M + 0] = inv_Ixx;
    B[4 * M + 1] = inv_Iyy;
    B[5 * M + 2] = inv_Izz;

    const float roll_max = 45.0f * DEG_TO_RAD, pitch_max = 45.0f * DEG_TO_RAD, yaw_max = 90.0f * DEG_TO_RAD;
    const float p_max = 300.0f * DEG_TO_RAD, q_max = 300.0f * DEG_TO_RAD, r_max = 200.0f * DEG_TO_RAD;

    float Qc[N * N] = {0};
    Qc[0 * N + 0] = 1.0f / (roll_max * roll_max);
    Qc[1 * N + 1] = 1.0f / (pitch_max * pitch_max);
    Qc[2 * N + 2] = 1.0f / (yaw_max * yaw_max);
    Qc[3 * N + 3] = 1.0f / (p_max * p_max);
    Qc[4 * N + 4] = 1.0f / (q_max * q_max);
    Qc[5 * N + 5] = 1.0f / (r_max * r_max);

    const float perc = 0.5f;
    const float max_tau_roll = 2.0f * MOTOR_B * L_ARM * MAX_OMEGA * MAX_OMEGA * perc;
    const float max_tau_pitch = max_tau_roll;
    const float max_tau_yaw = 4.0f * MOTOR_D * MAX_OMEGA * MAX_OMEGA * perc;

    float Rc[M * M] = {0};
    Rc[0 * M + 0] = 1.0f / (max_tau_roll * max_tau_roll);
    Rc[1 * M + 1] = 1.0f / (max_tau_pitch * max_tau_pitch);
    Rc[2 * M + 2] = 1.0f / (max_tau_yaw * max_tau_yaw);

    // Discretização Taylor 2a ordem (idêntica em forma à de verify_solvers.cpp
    // e à updateSystemMatrix() de src/main.cpp para este ponto de operação)
    float I6[N * N] = {0};
    for (int i = 0; i < N; i++) I6[i * N + i] = 1.0f;

    float A2[N * N], AA_dt2[N * N];
    MatrixOperations::matrixMultiply(A, A, A2, N, N, N);
    for (int i = 0; i < N * N; i++) Ad[i] = I6[i] + A[i] * dt + A2[i] * (dt * dt * 0.5f);

    float AB[N * M];
    MatrixOperations::matrixMultiply(A, B, AB, N, N, M);
    for (int i = 0; i < N * M; i++) Bd[i] = B[i] * dt + AB[i] * (dt * dt * 0.5f);

    float ATQ[N * N], QA[N * N], ATQ_QA[N * N];
    float AT[N * N];
    MatrixOperations::transposeMatrix(A, AT, N, N);
    MatrixOperations::matrixMultiply(AT, Qc, ATQ, N, N, N);
    MatrixOperations::matrixMultiply(Qc, A, QA, N, N, N);
    MatrixOperations::matrixAdd(ATQ, QA, ATQ_QA, N, N);
    for (int i = 0; i < N * N; i++) Qd[i] = Qc[i] * dt + ATQ_QA[i] * (dt * dt * 0.5f);

    float BTQB[M * M], BT[M * N], QB[N * M], BTQ[M * N];
    MatrixOperations::transposeMatrix(B, BT, N, M);
    MatrixOperations::matrixMultiply(Qc, B, QB, N, N, M);
    MatrixOperations::matrixMultiply(BT, QB, BTQB, M, N, M);
    for (int i = 0; i < M * M; i++) Rd[i] = Rc[i] * dt + BTQB[i] * (dt * dt * dt / 3.0f);

    (void)AA_dt2;
}

static void printMatrix(const float* m, int rows, int cols, const char* name) {
    Serial.printf("%s [%dx%d]:\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        Serial.print("  ");
        for (int j = 0; j < cols; j++) {
            Serial.printf("%12.6g ", m[i * cols + j]);
        }
        Serial.println();
    }
}

static void runMethod(AutoLQR& controller, const char* method, const char* tag = nullptr) {
    controller.setStateMatrix(Ad);
    controller.setInputMatrix(Bd);
    controller.setCostMatrices(Qd, Rd);

    unsigned long t0 = micros();
    bool ok = controller.computeGains(method);
    unsigned long t1 = micros();

    Serial.println("========================================");
    Serial.printf("METODO=%s%s%s ok=%d iters=%d time_us=%lu\n",
                   method, tag ? " " : "", tag ? tag : "",
                   ok ? 1 : 0, controller.getLastIterations(), (unsigned long)(t1 - t0));
    Serial.printf("lastResidual(DARE real)=%.6e lastStepDelta(criterio interno)=%.6e\n",
                  controller.getLastResidual(), controller.getLastStepDelta());

    float K[M * N];
    if (controller.exportGains(K)) {
        printMatrix(K, M, N, "K");
    } else {
        Serial.println("K indisponivel");
    }
}
static void runMethod(const char* method) { runMethod(lqr, method); }

void setup() {
    Serial.begin(115200);
    unsigned long t_serial = millis();
    while (!Serial && millis() - t_serial < 3000) {}
    delay(1500);

    Serial.println("### VERIFICACAO ON-DEVICE (ESP32-S2) — caso C1_hover ###");
    buildC1Hover();

    printMatrix(Ad, N, N, "Ad");
    printMatrix(Bd, N, M, "Bd");
    printMatrix(Qd, N, N, "Qd");
    printMatrix(Rd, M, M, "Rd");

    const char* methods[] = {"SDA", "SDA_FIXED", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA", "VAN_DOOREN",
                              "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"};
    for (const char* m : methods) {
        runMethod(m);
        delay(50);
    }

    // Iterativo isolado: P_warm começa zerado (só métodos Iterative* o tocam,
    // e nenhum rodou ainda nesta instância) — primeira chamada é cold-start
    // de verdade; segunda chamada reaproveita o P convergido. ITERATIVE e
    // ITERATIVE_FIXED compartilham o mesmo P_warm (mesmo P físico,
    // aritmética diferente) dentro da MESMA instância — por isso
    // ITERATIVE_FIXED já nasce warm em "lqr", herdando o resultado do
    // float. lqr_iter_cold isola o cold-start do ITERATIVE_FIXED puro,
    // numa instância própria, sem herança do float.
    Serial.println("### ITERATIVO: frio vs. quente ###");
    runMethod(lqr, "ITERATIVE", "(1/2 frio)");
    runMethod(lqr, "ITERATIVE", "(2/2 quente)");
    runMethod(lqr, "ITERATIVE_FIXED", "(1/2 quente, herdado do float acima)");
    runMethod(lqr, "ITERATIVE_FIXED", "(2/2 quente)");
    runMethod(lqr_iter_cold, "ITERATIVE_FIXED", "(instancia isolada, frio de verdade)");
    runMethod(lqr_iter_cold, "ITERATIVE_FIXED", "(instancia isolada, quente)");

    Serial.println("========================================");
    Serial.println("### FIM DA VERIFICACAO ###");
}

void loop() {
    delay(5000);
    Serial.println("(idle — verificacao concluida no setup)");
}
