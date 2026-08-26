/**
 * TESTE: Avaliação da Melhor Matriz de Execução Ad
 * 
 * Testa e compara diferentes solvers DARE usando a matriz discretizada Ad
 * para encontrar a melhor configuração de execução para o sistema de controle.
 */

#include <AutoLQR.h>
#include <math.h>
#include "MatrixOperations.h"

namespace RiccatiBenchmark {
    const int STATE_SIZE_BENCH = 6;
    const int CONTROL_SIZE_BENCH = 3;
    const int NUM_ITERATIONS = 1000;

    AutoLQR lqr_sda(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_schur(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_vd(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_iter(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);

    unsigned long times_sda[NUM_ITERATIONS];
    unsigned long times_schur[NUM_ITERATIONS];
    unsigned long times_vd[NUM_ITERATIONS];
    unsigned long times_iter[NUM_ITERATIONS];

    float A_bench[STATE_SIZE_BENCH * STATE_SIZE_BENCH];
    float Ad_bench[STATE_SIZE_BENCH * STATE_SIZE_BENCH];
    float Bd_bench[STATE_SIZE_BENCH * CONTROL_SIZE_BENCH];

    const float Ixx = 16.57e-6;
    const float Iyy = 16.57e-6;
    const float Izz = 29.80e-6;
    const float Ir = 1.0e-9;
    const float omega_r = 0;

    float B[STATE_SIZE_BENCH * CONTROL_SIZE_BENCH] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        1/Ixx, 0, 0,
        0, 1/Iyy, 0,
        0, 0, 1/Izz
    };

    float Q[STATE_SIZE_BENCH * STATE_SIZE_BENCH] = {
        100, 0, 0, 0, 0, 0,
        0, 100, 0, 0, 0, 0,
        0, 0, 100, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1
    };

    float R[CONTROL_SIZE_BENCH * CONTROL_SIZE_BENCH] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };

    void updateSystemMatrixBench(float roll, float pitch, float yaw, float p, float q, float r, const float* B_orig, float alpha_1, float alpha_2, float alpha_3) {
        float sin_roll = sin(roll);
        float cos_roll = cos(roll);
        float cos_pitch = cos(pitch);
        float tan_pitch = tan(pitch);
        float inv_cos_pitch = 1.0f / cos_pitch;
        memset(A_bench, 0, sizeof(A_bench));
        A_bench[0 * STATE_SIZE_BENCH + 3] = 1;
        A_bench[0 * STATE_SIZE_BENCH + 4] = sin_roll * tan_pitch;
        A_bench[0 * STATE_SIZE_BENCH + 5] = cos_roll * tan_pitch;
        A_bench[1 * STATE_SIZE_BENCH + 4] = cos_roll;
        A_bench[1 * STATE_SIZE_BENCH + 5] = -sin_roll;
        A_bench[2 * STATE_SIZE_BENCH + 4] = sin_roll * inv_cos_pitch;
        A_bench[2 * STATE_SIZE_BENCH + 5] = cos_roll * inv_cos_pitch;
        A_bench[3 * STATE_SIZE_BENCH + 4] = alpha_1 * ((Iyy - Izz) / Ixx) * r - Ir * omega_r / Ixx;
        A_bench[3 * STATE_SIZE_BENCH + 5] = (1 - alpha_1) * ((Iyy - Izz) / Ixx) * q;
        A_bench[4 * STATE_SIZE_BENCH + 3] = alpha_2 * ((Izz - Ixx) / Iyy) * r - Ir * omega_r / Iyy;
        A_bench[4 * STATE_SIZE_BENCH + 5] = (1 - alpha_2) * ((Izz - Ixx) / Iyy) * p;
        A_bench[5 * STATE_SIZE_BENCH + 3] = alpha_3 * ((Ixx - Iyy) / Izz) * q;
        A_bench[5 * STATE_SIZE_BENCH + 4] = (1 - alpha_3) * ((Ixx - Iyy) / Izz) * p;
        float dt = 0.012f;
        float A2[STATE_SIZE_BENCH * STATE_SIZE_BENCH] = {0};
        MatrixOperations::matrixMultiply(A_bench, A_bench, A2, STATE_SIZE_BENCH, STATE_SIZE_BENCH, STATE_SIZE_BENCH);
        for (int i = 0; i < STATE_SIZE_BENCH; i++) {
            for (int j = 0; j < STATE_SIZE_BENCH; j++) {
                if (i == j) {
                    Ad_bench[i * STATE_SIZE_BENCH + j] = 1 + A_bench[i * STATE_SIZE_BENCH + j] * dt + A2[i * STATE_SIZE_BENCH + j] * dt * dt * 0.5f;
                } else {
                    Ad_bench[i * STATE_SIZE_BENCH + j] = A_bench[i * STATE_SIZE_BENCH + j] * dt + A2[i * STATE_SIZE_BENCH + j] * dt * dt * 0.5f;
                }
            }
        }
        float AB[STATE_SIZE_BENCH * CONTROL_SIZE_BENCH] = {0};
        MatrixOperations::matrixMultiply(A_bench, B_orig, AB, STATE_SIZE_BENCH, STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
        float dt2_over_2 = dt * dt * 0.5f;
        for (int i = 0; i < STATE_SIZE_BENCH; i++) {
            for (int j = 0; j < CONTROL_SIZE_BENCH; j++) {
                int index = i * CONTROL_SIZE_BENCH + j;
                Bd_bench[index] = B_orig[index] * dt + AB[index] * dt2_over_2;
            }
        }
    }

    void run() {
        Serial.begin(115200);
        delay(2000);
        Serial.println("\n\n=========================================================");
        Serial.println("       INICIANDO OTIMIZAÇÃO DE ALPHAS (SDA)");
        Serial.println("=========================================================");
        
        lqr_sda.setCostMatrices(Q, R);
        
        // Estado fixo para comparação justa
        float roll = 0.1f, pitch = 0.1f, yaw = 0.1f;
        float p = 0.5f, q = 0.5f, r = 0.5f;
        
        float best_time = 1e9;
        float best_a1 = 0, best_a2 = 0, best_a3 = 0;
        
        int steps = 20;
        float min_val = 0.0f;
        float max_val = 1.0f;
        float step_size = (max_val - min_val) / steps;
        
        int total_iterations = (steps + 1) * (steps + 1) * (steps + 1);
        int current_iter = 0;
        
        Serial.printf("Testando intervalo [%.1f, %.1f] com %d divisões (passo %.1f)\n", min_val, max_val, steps, step_size);
        Serial.printf("Total de combinações: %d\n", total_iterations);

        for (float a1 = min_val; a1 <= max_val + 0.001f; a1 += step_size) {
            for (float a2 = min_val; a2 <= max_val + 0.001f; a2 += step_size) {
                for (float a3 = min_val; a3 <= max_val + 0.001f; a3 += step_size) {
                    
                    updateSystemMatrixBench(roll, pitch, yaw, p, q, r, B, a1, a2, a3);
                    
                    lqr_sda.setStateMatrix(Ad_bench);
                    lqr_sda.setInputMatrix(Bd_bench);
                    
                    unsigned long t0 = micros();
                    lqr_sda.computeGains("SDA");
                    unsigned long duration = micros() - t0;
                    
                    if (duration < best_time) {
                        best_time = (float)duration;
                        best_a1 = a1;
                        best_a2 = a2;
                        best_a3 = a3;
                        Serial.printf("Novo Melhor: %.2f us | Alphas: [%.1f, %.1f, %.1f]\n", best_time, a1, a2, a3);
                    }
                    
                    current_iter++;
                    if (current_iter % 500 == 0) {
                        Serial.printf("Progresso: %d/%d (%.1f%%)\n", current_iter, total_iterations, (float)current_iter*100.0/total_iterations);
                    }
                }
            }
            delay(1); 
        }

        Serial.println("\n================ RESULTADO FINAL ================");
        Serial.printf("Melhor Tempo Encontrado: %.2f us\n", best_time);
        Serial.printf("Melhor Configuração:\n");
        Serial.printf("Alpha 1: %.2f\n", best_a1);
        Serial.printf("Alpha 2: %.2f\n", best_a2);
        Serial.printf("Alpha 3: %.2f\n", best_a3);
        Serial.println("=================================================");
        
        while(1) { delay(1000); }
    }
}

void setup() {
    RiccatiBenchmark::run();
}

void loop() {}

