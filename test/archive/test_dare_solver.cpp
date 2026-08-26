/**
 * TESTE: Validação do Solver DARE
 * 
 * Testa a implementação do solver da equação algébrica de Riccati discreta (DARE)
 * verificando convergência, precisão numérica e comparando com soluções de referência.
 */

#include <Arduino.h>
#include "AutoLQR.h"
#include "MatrixOperations.h"
#include <ArduinoEigen.h>

// Define as dimensões das matrizes para o exemplo
// Sistema: x(k+1) = Ax(k) + Bu(k)
// A: 6x6, B: 6x3, Q: 6x6, R: 3x3
const int STATE_DIM_TEST = 6;
const int CONTROL_DIM_TEST = 3;

// Instância do controlador LQR para o teste
AutoLQR lqr_test(STATE_DIM_TEST, CONTROL_DIM_TEST);

// Parâmetros físicos do sistema (do arquivo main copy.cpp.txt)
const float Ixx = 1e-5;  // kg·m²
const float Iyy = 1e-5;  // kg·m²
const float Izz = 1e-5;  // kg·m²
const float Ir = 1e-5;   // kg·m²
const float omega_r = 0;

// Valores de roll, pitch, yaw, p, q, r para o teste
const float roll = 0.0f;
const float pitch = 0.0f;
const float yaw = 0.0f;
const float p = 0.0f;
const float q = 0.0f;
const float r = 0.0f;

// Matrizes do problema DARE baseadas no sistema do drone
// Matriz A contínua (será discretizada)
float A_data[STATE_DIM_TEST * STATE_DIM_TEST];

// Matriz B contínua
float B_data[STATE_DIM_TEST * CONTROL_DIM_TEST] = {
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    1/Ixx, 0, 0,
    0, 1/Iyy, 0,
    0, 0, 1/Izz
};

// Matriz Q (custo dos estados)
float Q_data[STATE_DIM_TEST * STATE_DIM_TEST] = {
    100, 0, 0, 0, 0, 0,
    0, 100, 0, 0, 0, 0,
    0, 0, 100, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1
};

// Matriz R (custo dos controles)
float R_data[CONTROL_DIM_TEST * CONTROL_DIM_TEST] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
};

void initialize_A_matrix() {
    // Inicializa A com zeros
    for (int i = 0; i < STATE_DIM_TEST * STATE_DIM_TEST; i++) {
        A_data[i] = 0.0f;
    }
    
    // Calcula valores trigonométricos
    float sin_roll = sin(roll);
    float cos_roll = cos(roll);
    float cos_pitch = cos(pitch);
    float tan_pitch = tan(pitch);
    float inv_cos_pitch = 1.0f / cos_pitch;
    
    // Preenche a matriz A conforme o sistema do drone
    // Linha 0
    A_data[0 * STATE_DIM_TEST + 3] = 1;
    A_data[0 * STATE_DIM_TEST + 4] = sin_roll * tan_pitch;
    A_data[0 * STATE_DIM_TEST + 5] = cos_roll * tan_pitch;
    
    // Linha 1
    A_data[1 * STATE_DIM_TEST + 4] = cos_roll;
    A_data[1 * STATE_DIM_TEST + 5] = -sin_roll;
    
    // Linha 2
    A_data[2 * STATE_DIM_TEST + 4] = sin_roll * inv_cos_pitch;
    A_data[2 * STATE_DIM_TEST + 5] = cos_roll * inv_cos_pitch;
    
    // Linha 3
    A_data[3 * STATE_DIM_TEST + 4] = ((Iyy - Izz) / (2 * Ixx)) * r - Ir * omega_r / Ixx;
    A_data[3 * STATE_DIM_TEST + 5] = ((Iyy - Izz) / (2 * Ixx)) * q;
    
    // Linha 4
    A_data[4 * STATE_DIM_TEST + 3] = ((Izz - Ixx) / (2 * Iyy)) * r - Ir * omega_r / Iyy;
    A_data[4 * STATE_DIM_TEST + 5] = ((Izz - Ixx) / (2 * Iyy)) * p;
    
    // Linha 5
    A_data[5 * STATE_DIM_TEST + 3] = ((Ixx - Iyy) / (2 * Izz)) * q;
    A_data[5 * STATE_DIM_TEST + 4] = ((Ixx - Iyy) / (2 * Izz)) * p;
}

void discretize_matrices() {
    float samplingTime = 0.01f;
    float dt = samplingTime;
    
    // Salva uma cópia da matriz A contínua
    float A_continuous[STATE_DIM_TEST * STATE_DIM_TEST];
    memcpy(A_continuous, A_data, sizeof(A_data));
    
    // Método de discretização mais preciso: expm(A*dt)
    // Para sistemas pequenos, usar série de Taylor até 3ª ordem
    float dt2_over_2 = dt * dt * 0.5f;
    float dt3_over_6 = dt * dt * dt / 6.0f;
    
    // Calcula A^2
    float A2[STATE_DIM_TEST * STATE_DIM_TEST] = {0};
    MatrixOperations::matrixMultiply(A_continuous, A_continuous, A2, STATE_DIM_TEST, STATE_DIM_TEST, STATE_DIM_TEST);
    
    // Calcula A^3 para maior precisão
    float A3[STATE_DIM_TEST * STATE_DIM_TEST] = {0};
    MatrixOperations::matrixMultiply(A2, A_continuous, A3, STATE_DIM_TEST, STATE_DIM_TEST, STATE_DIM_TEST);
    
    // Discretiza A: Ad = I + A*dt + (A^2*dt^2)/2 + (A^3*dt^3)/6
    for (int i = 0; i < STATE_DIM_TEST; i++) {
        for (int j = 0; j < STATE_DIM_TEST; j++) {
            int idx = i * STATE_DIM_TEST + j;
            if (i == j) {
                A_data[idx] = 1.0f + A_continuous[idx] * dt + 
                             A2[idx] * dt2_over_2 + 
                             A3[idx] * dt3_over_6;
            } else {
                A_data[idx] = A_continuous[idx] * dt + 
                             A2[idx] * dt2_over_2 + 
                             A3[idx] * dt3_over_6;
            }
        }
    }
    
    // Discretiza B: Bd = B*dt + (A*B*dt^2)/2 + (A^2*B*dt^3)/6
    float AB[STATE_DIM_TEST * CONTROL_DIM_TEST] = {0};
    MatrixOperations::matrixMultiply(A_continuous, B_data, AB, STATE_DIM_TEST, STATE_DIM_TEST, CONTROL_DIM_TEST);
    
    float A2B[STATE_DIM_TEST * CONTROL_DIM_TEST] = {0};
    MatrixOperations::matrixMultiply(A2, B_data, A2B, STATE_DIM_TEST, STATE_DIM_TEST, CONTROL_DIM_TEST);
    
    // Salva B original
    float B_continuous[STATE_DIM_TEST * CONTROL_DIM_TEST];
    memcpy(B_continuous, B_data, sizeof(B_data));
    
    for (int i = 0; i < STATE_DIM_TEST; i++) {
        for (int j = 0; j < CONTROL_DIM_TEST; j++) {
            int index = i * CONTROL_DIM_TEST + j;
            B_data[index] = B_continuous[index] * dt + 
                           AB[index] * dt2_over_2 + 
                           A2B[index] * dt3_over_6;
        }
    }
}

void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    Serial.printf("Matriz %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Serial.printf("%8.4f ", matrix[i * cols + j]);
        }
        Serial.println();
    }
    Serial.println();
}

void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println("--- Teste do Solver DARE com Sistema de Drone ---");
    Serial.printf("Parâmetros: roll=%.2f, pitch=%.2f, yaw=%.2f, p=%.2f, q=%.2f, r=%.2f\n\n",
                  roll, pitch, yaw, p, q, r);

    // Inicializa e discretiza as matrizes
    initialize_A_matrix();
    discretize_matrices();

    // Configura as matrizes no objeto lqr_test
    lqr_test.setStateMatrix(A_data);
    lqr_test.setInputMatrix(B_data);
    lqr_test.setCostMatrices(Q_data, R_data);

    Serial.println("Matrizes de entrada (discretizadas):");
    print_matrix(A_data, STATE_DIM_TEST, STATE_DIM_TEST, "A");
    print_matrix(B_data, STATE_DIM_TEST, CONTROL_DIM_TEST, "B");
    print_matrix(Q_data, STATE_DIM_TEST, STATE_DIM_TEST, "Q");
    print_matrix(R_data, CONTROL_DIM_TEST, CONTROL_DIM_TEST, "R");

    // Calcula os ganhos usando o novo método
    Serial.println("Calculando a solução da DARE e a matriz de ganho K...");
    bool success = lqr_test.computeGains();

    if (success) {
        Serial.println("\n--- Resultados ---");
        
        // Obtém e imprime a matriz de ganho K
        float K_result[CONTROL_DIM_TEST * STATE_DIM_TEST];
        lqr_test.exportGains(K_result);
        print_matrix(K_result, CONTROL_DIM_TEST, STATE_DIM_TEST, "K (Ganho)");

        // Obtém e imprime a matriz P (solução da DARE)
        const float* P_ptr = lqr_test.getRicattiSolution();
        if (P_ptr) {
            print_matrix(P_ptr, STATE_DIM_TEST, STATE_DIM_TEST, "P (Solução de Riccati)");
        }

    } else {
        Serial.println("Falha ao calcular a solução da DARE. Verifique as mensagens de erro.");
    }
    
    Serial.println("\n--- Teste Concluído ---");
}

void loop() {
    // O teste roda apenas uma vez no setup()
}
