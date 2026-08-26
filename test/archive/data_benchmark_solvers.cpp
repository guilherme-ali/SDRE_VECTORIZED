
#include <AutoLQR.h>
#include <math.h>
#include "MatrixOperations.h"

namespace RiccatiBenchmark {
    const int STATE_SIZE_BENCH = 6;
    const int CONTROL_SIZE_BENCH = 3;
    const int NUM_ITERATIONS = 10000;
    const int NUM_METHODS = 7;

    AutoLQR lqr_sda(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_sda_ss(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_asda(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_sda_scaled(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_adda(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_vd(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
    AutoLQR lqr_iter(STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);

    float A_bench[STATE_SIZE_BENCH * STATE_SIZE_BENCH];
    float Ad_bench[STATE_SIZE_BENCH * STATE_SIZE_BENCH];
    float Bd_bench[STATE_SIZE_BENCH * CONTROL_SIZE_BENCH];

    // Parâmetros de inércia (serão variados aleatoriamente)
    float Ixx = 16.57e-6f;
    float Iyy = 16.57e-6f;
    float Izz = 29.80e-6f;
    const float Ir = 1.0e-9f;
    const float omega_r = 0;

    // Matriz B será recalculada a cada iteração
    float B[STATE_SIZE_BENCH * CONTROL_SIZE_BENCH];

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

    // Função para atualizar matriz B com novos parâmetros de inércia
    void updateMatrixB() {
        memset(B, 0, sizeof(B));
        B[3 * CONTROL_SIZE_BENCH + 0] = 1.0f / Ixx;
        B[4 * CONTROL_SIZE_BENCH + 1] = 1.0f / Iyy;
        B[5 * CONTROL_SIZE_BENCH + 2] = 1.0f / Izz;
    }

    // Função para gerar parâmetros de inércia aleatórios
    void randomizeInertia() {
        // Faixa de inércia: Mini drones (~1e-6 kg*m²) até drones agrícolas grandes (~1e-1 kg*m²)
        float log_min = log10f(1e-6f);
        float log_max = log10f(1e-1f);
        
        // Gera uma escala base de magnitude para o drone atual
        float log_scale = log_min + (log_max - log_min) * random(0, 1000) / 1000.0f;
        float I_base = powf(10.0f, log_scale);
        
        // Aplica simetria aproximada: Ixx e Iyy variam pouco entre si (±10% da base)
        // Isso reflete a construção física onde os braços são simétricos
        Ixx = I_base * (0.9f + 0.2f * (random(0, 1000) / 1000.0f));
        Iyy = I_base * (0.9f + 0.2f * (random(0, 1000) / 1000.0f));
        
        // Izz geralmente é maior que Ixx/Iyy (para estruturas planas, Izz ~ Ixx + Iyy)
        // Vamos definir Izz proporcional à média das inércias laterais
        float I_lat_avg = (Ixx + Iyy) / 2.0f;
        Izz = I_lat_avg * (1.5f + 0.5f * (random(0, 1000) / 1000.0f)); 
        
        updateMatrixB();
    }

    // Função para calcular número de condicionamento da matriz Hamiltoniana H
    // H = [ A        -B*R^(-1)*B^T ]
    //     [ -Q       -A^T          ]
    // O condicionamento de H indica a dificuldade do problema de Riccati
    float calculateConditionNumber() {
        const int n = STATE_SIZE_BENCH;
        const int m = CONTROL_SIZE_BENCH;
        const int n2 = 2 * n;  // Hamiltoniana é 2n x 2n
        
        // Construir matriz Hamiltoniana H (2n x 2n)
        float H[n2 * n2];
        memset(H, 0, sizeof(H));
        
        // Calcular G = Bd * R^(-1) * Bd^T (como R = I, G = Bd * Bd^T)
        float G[n * n];
        memset(G, 0, sizeof(G));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += Bd_bench[i * m + k] * Bd_bench[j * m + k];
                }
                G[i * n + j] = sum;
            }
        }
        
        // Preencher H:
        // Bloco superior esquerdo: A (usa Ad_bench)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[i * n2 + j] = Ad_bench[i * n + j];
            }
        }
        
        // Bloco superior direito: -G = -B*R^(-1)*B^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[i * n2 + (n + j)] = -G[i * n + j];
            }
        }
        
        // Bloco inferior esquerdo: -Q
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[(n + i) * n2 + j] = -Q[i * n + j];
            }
        }
        
        // Bloco inferior direito: -A^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[(n + i) * n2 + (n + j)] = -Ad_bench[j * n + i];  // Transposta
            }
        }
        
        // Calcular norma de Frobenius de H
        float norm_H = 0;
        for (int i = 0; i < n2 * n2; i++) {
            norm_H += H[i] * H[i];
        }
        norm_H = sqrtf(norm_H);
        
        // Calcular inversa de H
        float H_inv[n2 * n2];
        MatrixOperations::matrixCopy(H, H_inv, n2 * n2);
        
        if (!MatrixOperations::invertMatrix(H_inv, H_inv, n2)) {
            return 1e10f; // Matriz singular
        }
        
        // Calcular norma de Frobenius de H^(-1)
        float norm_H_inv = 0;
        for (int i = 0; i < n2 * n2; i++) {
            norm_H_inv += H_inv[i] * H_inv[i];
        }
        norm_H_inv = sqrtf(norm_H_inv);
        
        // Número de condicionamento: ||H|| * ||H^(-1)||
        return norm_H * norm_H_inv;
    }

    void updateSystemMatrixBench(float roll, float pitch, float yaw, float p, float q, float r) {
        float alpha_1 = 0.5f;
        float alpha_2 = 0.5f;
        float alpha_3 = 0.5f;
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
        MatrixOperations::matrixMultiply(A_bench, B, AB, STATE_SIZE_BENCH, STATE_SIZE_BENCH, CONTROL_SIZE_BENCH);
        float dt2_over_2 = dt * dt * 0.5f;
        for (int i = 0; i < STATE_SIZE_BENCH; i++) {
            for (int j = 0; j < CONTROL_SIZE_BENCH; j++) {
                int index = i * CONTROL_SIZE_BENCH + j;
                Bd_bench[index] = B[index] * dt + AB[index] * dt2_over_2;
            }
        }
    }

    // Função para imprimir resultado em formato CSV
    void printResult(float cond, unsigned long time, int iters, const char* method) {
        char csvLine[100];
        char condStr[20];
        
        // Formatar CondNumber com vírgula decimal
        snprintf(condStr, sizeof(condStr), "%.4f", cond);
        // Substituir ponto por vírgula
        for (int j = 0; condStr[j] != '\0'; j++) {
            if (condStr[j] == '.') condStr[j] = ',';
        }
        
        snprintf(csvLine, sizeof(csvLine), "%s;%lu;%d;%s", 
                 condStr,
                 time,
                 iters,
                 method);
        Serial.println(csvLine);
    }

    void run() {
        Serial.begin(115200);
        delay(2000);
        
        // Inicializar matriz B
        updateMatrixB();
        
        lqr_sda.setCostMatrices(Q, R);
        lqr_sda_ss.setCostMatrices(Q, R);
        lqr_asda.setCostMatrices(Q, R);
        lqr_sda_scaled.setCostMatrices(Q, R);
        lqr_adda.setCostMatrices(Q, R);
        lqr_vd.setCostMatrices(Q, R);
        lqr_iter.setCostMatrices(Q, R);
        
        float roll = 0, pitch = 0, yaw = 0;
        float p = 0, q = 0, r_rate = 0;
        
        // Warm-up do método iterativo
        updateSystemMatrixBench(roll, pitch, yaw, p, q, r_rate);
        lqr_iter.setStateMatrix(Ad_bench);
        lqr_iter.setInputMatrix(Bd_bench);
        lqr_iter.computeGains("ITERATIVE");
        
        // Imprimir cabeçalho CSV
        Serial.println("CondNumber;Tempo_us;Iteracoes;Metodo");
        
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            // Randomizar parâmetros de inércia
            randomizeInertia();
            
            // Randomizar estado
            roll  = (random(-314, 314) / 1000.0f);  // -0.314 a 0.314 rad (~18°)
            pitch = (random(-314, 314) / 1000.0f);
            yaw   = (random(-314, 314) / 1000.0f);
            p     = (random(-100, 100) / 100.0f);   // -1 a 1 rad/s
            q     = (random(-100, 100) / 100.0f);
            r_rate = (random(-100, 100) / 100.0f);
            
            updateSystemMatrixBench(roll, pitch, yaw, p, q, r_rate);
            
            // Calcular número de condicionamento
            float condNum = calculateConditionNumber();
            
            unsigned long t0;
            
            // SDA Original
            lqr_sda.setStateMatrix(Ad_bench);
            lqr_sda.setInputMatrix(Bd_bench);
            t0 = micros();
            lqr_sda.computeGains("SDA");
            printResult(condNum, micros() - t0, lqr_sda.getLastIterations(), "SDA");
            
            // SDA-SS
            lqr_sda_ss.setStateMatrix(Ad_bench);
            lqr_sda_ss.setInputMatrix(Bd_bench);
            t0 = micros();
            lqr_sda_ss.computeGains("SDA_SS");
            printResult(condNum, micros() - t0, lqr_sda_ss.getLastIterations(), "SDA-SS");
            
            // ASDA
            lqr_asda.setStateMatrix(Ad_bench);
            lqr_asda.setInputMatrix(Bd_bench);
            t0 = micros();
            lqr_asda.computeGains("ASDA");
            printResult(condNum, micros() - t0, lqr_asda.getLastIterations(), "ASDA");
            
            // SDA Scaled
            lqr_sda_scaled.setStateMatrix(Ad_bench);
            lqr_sda_scaled.setInputMatrix(Bd_bench);
            t0 = micros();
            lqr_sda_scaled.computeGains("SDA_SCALED");
            printResult(condNum, micros() - t0, lqr_sda_scaled.getLastIterations(), "SDA_SCALED");
            
            // ADDA
            lqr_adda.setStateMatrix(Ad_bench);
            lqr_adda.setInputMatrix(Bd_bench);
            t0 = micros();
            lqr_adda.computeGains("ADDA");
            printResult(condNum, micros() - t0, lqr_adda.getLastIterations(), "ADDA");
            
            // VAN DOOREN
            lqr_vd.setStateMatrix(Ad_bench);
            lqr_vd.setInputMatrix(Bd_bench);
            t0 = micros();
            lqr_vd.computeGains("VAN_DOOREN");
            printResult(condNum, micros() - t0, lqr_vd.getLastIterations(), "VAN_DOOREN");
            
            // ITERATIVE
            lqr_iter.setStateMatrix(Ad_bench);
            lqr_iter.setInputMatrix(Bd_bench);
            t0 = micros();
            lqr_iter.computeGains("ITERATIVE");
            printResult(condNum, micros() - t0, lqr_iter.getLastIterations(), "ITERATIVE");
            
            delay(10);
        }
    }
}

void setup() {
    RiccatiBenchmark::run();
}

void loop() {}

