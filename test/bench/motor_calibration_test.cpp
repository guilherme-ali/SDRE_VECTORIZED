/**
 * ========================================================================
 * TESTE DE CALIBRAÇÃO DO COEFICIENTE MOTOR_B_COEFF
 * ========================================================================
 * 
 * OBJETIVO: Medir a força real produzida pelos motores e calcular o 
 *           coeficiente de empuxo (MOTOR_B_COEFF)
 * 
 * SETUP NECESSÁRIO:
 * 1. Coloque o drone sobre uma balança digital (precisão de 0.1g ou melhor)
 * 2. Fixe bem o drone para que não voe (use fita ou suporte)
 * 3. Zere a balança com o drone em cima (tara)
 * 4. Conecte via Serial Monitor (115200 baud)
 * 
 * PROCEDIMENTO:
 * 1. O programa irá varrer throttle de 0% a 100% em incrementos
 * 2. Para cada nível, você deve ANOTAR o valor da balança em gramas
 * 3. Digite o valor no Serial Monitor e pressione Enter
 * 4. O programa calculará automaticamente o MOTOR_B_COEFF
 * 
 * FÓRMULA:
 *   F_thrust = MOTOR_B_COEFF × (ω₁² + ω₂² + ω₃² + ω₄²)
 *   
 *   Onde:
 *   - F_thrust está em Newtons (N)
 *   - ω está em rad/s (velocidade angular)
 *   - MOTOR_B_COEFF está em N/(rad/s)²
 * 
 * CÁLCULO:
 *   1. Força medida na balança (gramas) → Newtons: F_N = (gramas/1000) × 9.81
 *   2. MOTOR_B_COEFF = F_N / (4 × ω²) (considerando 4 motores iguais)
 * 
 * ========================================================================
 */

#include <Arduino.h>
#include "MotorControl.h"

// ===== CONFIGURAÇÃO DO TESTE =====
const int THROTTLE_START = 0;      // Throttle inicial (%)
const int THROTTLE_END = 100;      // Throttle final (%)
const int THROTTLE_STEP = 5;       // Incremento de throttle (%)
const bool REVERSE_ORDER = false
;  // true = fim->início, false = início->fim
const int STABILIZATION_TIME = 3000; // Tempo de estabilização em ms

const float GRAVITY = 9.81f;       // Aceleração da gravidade (m/s²)

// Parâmetros do motor 716 Coreless
// Especificações típicas para motor 716 (7x16mm) coreless
const float KV_RATING = 14000.0f;  // KV do motor 716 coreless (RPM/Volt)
const float VOLTAGE = 3.7f;        // Tensão da bateria 1S LiPo (V)
const float MAX_RPM = KV_RATING * VOLTAGE; // RPM máximo (~52,000 RPM)
const float MAX_OMEGA = (MAX_RPM * 2.0f * PI) / 60.0f; // rad/s máximo (~5445 rad/s)

// ===== VARIÁVEIS GLOBAIS =====
MotorControl motors;
int current_throttle;
bool waiting_for_input = false;

struct CalibrationData {
    int throttle_percent;
    float force_grams;
    float force_newtons;
    float omega_sq_total;
    float calculated_b_coeff;
};

const int MAX_MEASUREMENTS = 30;
CalibrationData measurements[MAX_MEASUREMENTS];
int measurement_count = 0;

// ===== FUNÇÕES AUXILIARES =====

float throttleToOmegaSq(int throttle_percent) {
    /**
     * Converte throttle (%) para ω² (velocidade angular ao quadrado)
     * 
     * Assumindo relação linear: throttle % → RPM → ω
     * ω = (throttle/100) × MAX_OMEGA
     * ω² = [(throttle/100) × MAX_OMEGA]²
     */
    float throttle_fraction = throttle_percent / 100.0f;
    float omega = throttle_fraction * MAX_OMEGA;
    return omega * omega;
}

void printTestHeader() {
    Serial.println("\n");
    Serial.println("╔═══════════════════════════════════════════════════════════════╗");
    Serial.println("║    CALIBRAÇÃO DO COEFICIENTE DE EMPUXO (MOTOR_B_COEFF)       ║");
    Serial.println("╚═══════════════════════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("IMPORTANTE:");
    Serial.println("  1. Drone deve estar FIXO sobre a balança");
    Serial.println("  2. Balança deve estar ZERADA (tara) com drone em cima");
    Serial.println("  3. Anote o valor da balança e digite quando solicitado");
    Serial.println("  4. NÃO toque no drone durante as medições!");
    Serial.println();
    Serial.printf("Configuração do teste:\n");
    Serial.printf("  - Throttle: %d%% a %d%% em passos de %d%%\n", 
                  THROTTLE_START, THROTTLE_END, THROTTLE_STEP);
    Serial.printf("  - Ordem: %s\n", REVERSE_ORDER ? "DECRESCENTE (fim->início)" : "CRESCENTE (início->fim)");
    Serial.printf("  - Tempo de estabilização: %d ms\n", STABILIZATION_TIME);
    Serial.printf("  - KV do motor: %.0f RPM/V\n", KV_RATING);
    Serial.printf("  - Tensão: %.1f V\n", VOLTAGE);
    Serial.printf("  - RPM máximo: %.0f\n", MAX_RPM);
    Serial.printf("  - ω máximo: %.1f rad/s\n", MAX_OMEGA);
    Serial.println();
    Serial.println("Pressione ENTER para iniciar o teste...");
}

void printMeasurementPrompt(int throttle) {
    Serial.println("\n╔════════════════════════════════════════════════════════╗");
    Serial.printf("║  MEDIÇÃO - Throttle: %3d%%                             ║\n", throttle);
    Serial.println("╚════════════════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("  ► Motores acelerando...");
}

void printStabilized() {
    Serial.println("  ► Estabilizado!");
    Serial.println();
    Serial.println("  📊 LEIA O VALOR DA BALANÇA (em gramas)");
    Serial.println("     Digite o valor e pressione ENTER:");
    Serial.print("     > ");
}

void processMeasurement(float force_grams) {
    if (measurement_count >= MAX_MEASUREMENTS) {
        Serial.println("❌ Limite de medições atingido!");
        return;
    }
    
    // Converte gramas para Newtons
    float force_newtons = (force_grams / 1000.0f) * GRAVITY;
    
    // Calcula ω² total (4 motores com mesmo throttle)
    float omega_sq_single = throttleToOmegaSq(current_throttle);
    float omega_sq_total = 4.0f * omega_sq_single;
    
    // Calcula MOTOR_B_COEFF para esta medição
    float b_coeff = 0;
    if (omega_sq_total > 0.001f) { // Evita divisão por zero
        b_coeff = force_newtons / omega_sq_total;
    }
    
    // Armazena medição
    measurements[measurement_count].throttle_percent = current_throttle;
    measurements[measurement_count].force_grams = force_grams;
    measurements[measurement_count].force_newtons = force_newtons;
    measurements[measurement_count].omega_sq_total = omega_sq_total;
    measurements[measurement_count].calculated_b_coeff = b_coeff;
    measurement_count++;
    
    // Mostra resultado da medição
    Serial.println();
    Serial.println("  ✅ Medição registrada:");
    Serial.printf("     Força: %.1f g = %.4f N\n", force_grams, force_newtons);
    Serial.printf("     ω² total: %.2f (rad/s)²\n", omega_sq_total);
    Serial.printf("     ω por motor: %.1f rad/s\n", sqrt(omega_sq_single));
    Serial.printf("     B_coeff calculado: %.8f N/(rad/s)²\n", b_coeff);
    Serial.println();
}

void exportToCSV() {
    Serial.println("\n\n");
    Serial.println("╔═══════════════════════════════════════════════════════════════╗");
    Serial.println("║              EXPORTAÇÃO PARA EXCEL (CSV)                      ║");
    Serial.println("╚═══════════════════════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("📋 Copie o conteúdo abaixo e cole no Excel:");
    Serial.println("   1. Selecione todo o texto entre as linhas de '==='");
    Serial.println("   2. Copie (Ctrl+C)");
    Serial.println("   3. Abra Excel → Cole (Ctrl+V)");
    Serial.println("   4. Excel → Dados → Texto para Colunas → Delimitado → Vírgula");
    Serial.println();
    Serial.println("===== INÍCIO DOS DADOS CSV =====");
    
    // Cabeçalho CSV
    Serial.println("Throttle(%),Força(g),Força(N),Omega²_Total(rad/s)²,B_coeff(N/(rad/s)²),Omega_Motor(rad/s),RPM_Motor");
    
    // Dados
    for (int i = 0; i < measurement_count; i++) {
        float omega_single = sqrt(measurements[i].omega_sq_total / 4.0f);
        float rpm_single = (omega_single * 60.0f) / (2.0f * PI);
        
        Serial.printf("%d,%.2f,%.6f,%.2f,%.10f,%.2f,%.0f\n",
                      measurements[i].throttle_percent,
                      measurements[i].force_grams,
                      measurements[i].force_newtons,
                      measurements[i].omega_sq_total,
                      measurements[i].calculated_b_coeff,
                      omega_single,
                      rpm_single);
    }
    
    Serial.println("===== FIM DOS DADOS CSV =====");
    Serial.println();
}

void printFinalResults() {
    Serial.println("\n\n");
    Serial.println("╔═══════════════════════════════════════════════════════════════╗");
    Serial.println("║                    RESULTADOS FINAIS                          ║");
    Serial.println("╚═══════════════════════════════════════════════════════════════╝");
    Serial.println();
    
    // Tabela de medições
    Serial.println("┌─────────┬───────────┬────────────┬─────────────┬──────────────────┐");
    Serial.println("│Throttle │  Força    │   Força    │   ω² total  │   B_coeff        │");
    Serial.println("│   (%)   │   (g)     │    (N)     │  (rad/s)²   │  N/(rad/s)²      │");
    Serial.println("├─────────┼───────────┼────────────┼─────────────┼──────────────────┤");
    
    float sum_b_coeff = 0;
    int valid_measurements = 0;
    
    for (int i = 0; i < measurement_count; i++) {
        Serial.printf("│  %3d    │  %7.1f  │  %8.4f  │  %10.2f │  %.10f  │\n",
                      measurements[i].throttle_percent,
                      measurements[i].force_grams,
                      measurements[i].force_newtons,
                      measurements[i].omega_sq_total,
                      measurements[i].calculated_b_coeff);
        
        // Soma apenas medições válidas (throttle > 0)
        if (measurements[i].throttle_percent > 0) {
            sum_b_coeff += measurements[i].calculated_b_coeff;
            valid_measurements++;
        }
    }
    
    Serial.println("└─────────┴───────────┴────────────┴─────────────┴──────────────────┘");
    Serial.println();
    
    // Calcula média do coeficiente
    if (valid_measurements > 0) {
        float avg_b_coeff = sum_b_coeff / valid_measurements;
        
        Serial.println("╔═══════════════════════════════════════════════════════════════╗");
        Serial.println("║              VALOR RECOMENDADO PARA MOTOR_B_COEFF             ║");
        Serial.println("╚═══════════════════════════════════════════════════════════════╝");
        Serial.println();
        Serial.printf("  📌 MOTOR_B_COEFF = %.10f\n", avg_b_coeff);
        Serial.println();
        Serial.printf("     (Média de %d medições válidas)\n", valid_measurements);
        Serial.println();
        Serial.println("  💡 COMO USAR:");
        Serial.println("     1. Copie o valor acima");
        Serial.println("     2. Cole em main.cpp na linha:");
        Serial.println("        const float MOTOR_B_COEFF = [VALOR];");
        Serial.println();
        
        // Calcula desvio padrão
        float variance = 0;
        for (int i = 0; i < measurement_count; i++) {
            if (measurements[i].throttle_percent > 0) {
                float diff = measurements[i].calculated_b_coeff - avg_b_coeff;
                variance += diff * diff;
            }
        }
        float std_dev = sqrt(variance / valid_measurements);
        float cv = (std_dev / avg_b_coeff) * 100.0f; // Coeficiente de variação
        
        Serial.printf("  📊 Estatísticas:\n");
        Serial.printf("     Desvio padrão: %.10f\n", std_dev);
        Serial.printf("     Coef. variação: %.2f%%\n", cv);
        
        if (cv < 5.0f) {
            Serial.println("     ✅ Excelente consistência!");
        } else if (cv < 10.0f) {
            Serial.println("     ✔️  Boa consistência");
        } else {
            Serial.println("     ⚠️  Alta variação - verifique setup experimental");
        }
        
        Serial.println();
    } else {
        Serial.println("❌ Nenhuma medição válida para calcular média!");
    }
    
    Serial.println("╔═══════════════════════════════════════════════════════════════╗");
    Serial.println("║                          GRÁFICO                              ║");
    Serial.println("╚═══════════════════════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("Força (N) vs Throttle (%):");
    Serial.println();
    
    // Gráfico ASCII simples
    for (int i = 0; i < measurement_count; i++) {
        int bar_length = (int)((measurements[i].force_newtons / 
                               measurements[measurement_count-1].force_newtons) * 50);
        Serial.printf("%3d%% │", measurements[i].throttle_percent);
        for (int j = 0; j < bar_length; j++) Serial.print("█");
        Serial.printf(" %.3f N\n", measurements[i].force_newtons);
    }
    
    Serial.println();
    Serial.println("═══════════════════════════════════════════════════════════════");
    Serial.println("Teste concluído! Pressione RESET para executar novamente.");
    Serial.println("═══════════════════════════════════════════════════════════════");
    
    // Exporta dados em formato CSV
    exportToCSV();
}

// ===== SETUP =====
void setup() {
    Serial.begin(115200);
    delay(2000);
    
    // Inicializa motores
    motors.begin();
    motors.setThrottleLimits(0, 100);
    
    // Calcula omega_sq máximo baseado no motor 716
    // MAX_OMEGA = ~5445 rad/s, então MAX_OMEGA² = ~29,647,025
    float max_omega_sq_motor = MAX_OMEGA * MAX_OMEGA;
    motors.setOmegaSqLimits(0, max_omega_sq_motor);
    
    Serial.printf("\n📊 Configuração de limites:\n");
    Serial.printf("   Throttle: 0%% - 100%%\n");
    Serial.printf("   Omega máximo: %.1f rad/s\n", MAX_OMEGA);
    Serial.printf("   Omega² máximo: %.0f (rad/s)²\n\n", max_omega_sq_motor);
    
    // IMPORTANTE: Motores já devem estar calibrados!
    // Se não estiverem, descomente a linha abaixo:
    // motors.calibrateESCs();
    
    motors.armMotors();
    delay(1000);
    
    printTestHeader();
    
    // Inicializa throttle baseado na ordem do teste
    current_throttle = REVERSE_ORDER ? THROTTLE_END : THROTTLE_START;
    
    // Aguarda usuário pressionar Enter
    while (!Serial.available()) {
        delay(100);
    }
    while (Serial.available()) Serial.read(); // Limpa buffer
    
    Serial.println("\n🚀 Iniciando teste de calibração...\n");
    delay(1000);
}

// ===== LOOP =====
void loop() {
    if (!waiting_for_input) {
        // Verifica se ainda há medições a fazer
        bool has_more_measurements = REVERSE_ORDER ? 
            (current_throttle >= THROTTLE_START) : 
            (current_throttle <= THROTTLE_END);
            
        // Inicia nova medição
        if (has_more_measurements) {
            printMeasurementPrompt(current_throttle);
            
            // Define throttle nos 4 motores
            motors.setMotor1(current_throttle);
            motors.setMotor2(current_throttle);
            motors.setMotor3(current_throttle);
            motors.setMotor4(current_throttle);
            
            // Debug: mostra status dos motores
            Serial.print("  ► Status: ");
            motors.printMotorValues();
            
            // Aguarda estabilização
            delay(STABILIZATION_TIME);
            
            printStabilized();
            waiting_for_input = true;
        } else {
            // Teste concluído
            motors.stopAllMotors();
            delay(500);
            motors.disarmMotors();
            
            printFinalResults();
            
            // Para o loop
            while (true) {
                delay(1000);
            }
        }
    } else {
        // Aguardando entrada do usuário
        if (Serial.available() > 0) {
            String input = Serial.readStringUntil('\n');
            input.trim();
            
            float force_grams = input.toFloat();
            
            if (force_grams >= 0) {
                processMeasurement(force_grams);
                
                // Para os motores antes da próxima medição (mas mantém armados!)
                motors.setAllMotors(0);
                delay(100);
                
                // Próximo throttle (crescente ou decrescente)
                if (REVERSE_ORDER) {
                    current_throttle -= THROTTLE_STEP;
                } else {
                    current_throttle += THROTTLE_STEP;
                }
                waiting_for_input = false;
            } else {
                Serial.println("❌ Valor inválido! Digite novamente:");
                Serial.print("     > ");
            }
        }
    }
}
