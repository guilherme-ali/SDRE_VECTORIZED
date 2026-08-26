/**
 * ============================================================================
 * CALIBRAÇÃO DO MAGNETÔMETRO QMC5883L
 * ============================================================================
 * 
 * Este script realiza a calibração do magnetômetro QMC5883L para corrigir:
 * 1. Hard-Iron Distortion (offset) - Campos magnéticos fixos do hardware
 * 2. Soft-Iron Distortion (escala) - Distorções causadas por materiais próximos
 * 
 * COMO USAR:
 * ----------
 * 1. Descomente este arquivo no platformio.ini (ou renomeie main.cpp temporariamente)
 * 2. Compile e faça upload
 * 3. Abra o Serial Monitor (115200 baud)
 * 4. Quando aparecer "INICIANDO CALIBRAÇÃO", gire o sensor LENTAMENTE em todas
 *    as direções por 30 segundos, formando um "8" no ar
 * 5. Copie os valores de calibração exibidos no final
 * 6. Cole os valores no main.cpp nas variáveis de calibração
 * 
 * CONEXÕES:
 * ---------
 * QMC5883L SDA -> GPIO40
 * QMC5883L SCL -> GPIO41
 * QMC5883L VCC -> 3.3V
 * QMC5883L GND -> GND
 * 
 * ============================================================================
 */

#include <Arduino.h>
#include <Wire.h>

// ===== CONFIGURAÇÃO =====
#define CALIBRATION_TIME_MS 30000  // Tempo de calibração (30 segundos)
#define SAMPLE_DELAY_MS 20         // Delay entre amostras (50Hz)

// ===== DEFINIÇÕES DO QMC5883L =====
#define QMC5883L_ADDR 0x0D

// Registros
#define QMC5883L_REG_DATA       0x00
#define QMC5883L_REG_STATUS     0x06
#define QMC5883L_REG_CONTROL1   0x09
#define QMC5883L_REG_CONTROL2   0x0A
#define QMC5883L_REG_SET_RESET  0x0B

// Segundo barramento I2C (já definido pela biblioteca Wire do ESP32)
// extern TwoWire Wire1; // Removido, usando Wire padrão

// ===== VARIÁVEIS DE CALIBRAÇÃO =====
int16_t x_min = 32767, x_max = -32768;
int16_t y_min = 32767, y_max = -32768;
int16_t z_min = 32767, z_max = -32768;

// Valores calculados
float offset_x, offset_y, offset_z;
float scale_x, scale_y, scale_z;

// ===== FUNÇÕES DO QMC5883L =====
void writeReg(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(QMC5883L_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

bool initQMC5883L() {
    // Inicializa I2C1 nos pinos corretos (MPU6050 já faz Wire.begin em outros arquivos,
    // mas aqui estamos rodando sozinhos)
    Wire.begin(11, 10);  // SDA = GPIO11, SCL = GPIO10
    Wire.setClock(400000);
    
    // Verifica se o sensor está presente
    Wire.beginTransmission(QMC5883L_ADDR);
    uint8_t error = Wire.endTransmission();
    
    if (error != 0) {
        Serial.println("❌ ERRO: QMC5883L não encontrado!");
        Serial.println("   Verifique as conexões:");
        Serial.println("   - SDA -> GPIO11");
        Serial.println("   - SCL -> GPIO10");
        Serial.println("   - VCC -> 3.3V");
        Serial.println("   - GND -> GND");
        return false;
    }
    
    // Soft Reset
    writeReg(QMC5883L_REG_CONTROL2, 0x80);
    delay(10);
    
    // Define período SET/RESET
    writeReg(QMC5883L_REG_SET_RESET, 0x01);
    
    // Configura: Modo Contínuo, ODR 200Hz, Range 8G, OSR 512
    writeReg(QMC5883L_REG_CONTROL1, 0x01 | 0x0C | 0x10 | 0x00);
    
    Serial.println("✅ QMC5883L inicializado com sucesso!");
    return true;
}

bool readRawQMC5883L(int16_t& x, int16_t& y, int16_t& z) {
    Wire.beginTransmission(QMC5883L_ADDR);
    Wire.write(QMC5883L_REG_DATA);
    Wire.endTransmission();
    
    Wire.requestFrom((uint8_t)QMC5883L_ADDR, (uint8_t)6);
    
    if (Wire.available() >= 6) {
        x = Wire.read() | (Wire.read() << 8);
        y = Wire.read() | (Wire.read() << 8);
        z = Wire.read() | (Wire.read() << 8);
        return true;
    }
    return false;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println();
    Serial.println("╔══════════════════════════════════════════════════════════════╗");
    Serial.println("║        CALIBRAÇÃO DO MAGNETÔMETRO QMC5883L                   ║");
    Serial.println("╚══════════════════════════════════════════════════════════════╝");
    Serial.println();
    
    if (!initQMC5883L()) {
        Serial.println("❌ Falha na inicialização. Reinicie o dispositivo.");
        while(1) delay(1000);
    }
    
    Serial.println();
    Serial.println("╔══════════════════════════════════════════════════════════════╗");
    Serial.println("║                    INSTRUÇÕES DE CALIBRAÇÃO                  ║");
    Serial.println("╠══════════════════════════════════════════════════════════════╣");
    Serial.println("║  1. Segure o drone/sensor firmemente                         ║");
    Serial.println("║  2. Quando começar, gire em TODAS as direções:               ║");
    Serial.println("║     - Rotacione no eixo X (roll)                             ║");
    Serial.println("║     - Rotacione no eixo Y (pitch)                            ║");
    Serial.println("║     - Rotacione no eixo Z (yaw)                              ║");
    Serial.println("║  3. Faça movimentos em forma de '8' no ar                    ║");
    Serial.println("║  4. Continue girando por 30 segundos                         ║");
    Serial.println("╚══════════════════════════════════════════════════════════════╝");
    Serial.println();
    
    // Countdown
    for (int i = 5; i > 0; i--) {
        Serial.printf("   Iniciando em %d segundos...\n", i);
        delay(1000);
    }
    
    Serial.println();
    Serial.println("🔄 ═══════════════════════════════════════════════════════════");
    Serial.println("🔄       INICIANDO CALIBRAÇÃO - GIRE O SENSOR AGORA!         ");
    Serial.println("🔄 ═══════════════════════════════════════════════════════════");
    Serial.println();
    
    unsigned long startTime = millis();
    unsigned long lastPrint = 0;
    int sampleCount = 0;
    
    while (millis() - startTime < CALIBRATION_TIME_MS) {
        int16_t x, y, z;
        
        if (readRawQMC5883L(x, y, z)) {
            // Atualiza mínimos e máximos
            if (x < x_min) x_min = x;
            if (x > x_max) x_max = x;
            if (y < y_min) y_min = y;
            if (y > y_max) y_max = y;
            if (z < z_min) z_min = z;
            if (z > z_max) z_max = z;
            
            sampleCount++;
            
            // Print de progresso a cada segundo
            if (millis() - lastPrint >= 1000) {
                int remaining = (CALIBRATION_TIME_MS - (millis() - startTime)) / 1000;
                Serial.printf("⏱️  Tempo restante: %2d s | Amostras: %4d | ", remaining, sampleCount);
                Serial.printf("X:[%+6d,%+6d] Y:[%+6d,%+6d] Z:[%+6d,%+6d]\n",
                             x_min, x_max, y_min, y_max, z_min, z_max);
                lastPrint = millis();
            }
        }
        
        delay(SAMPLE_DELAY_MS);
    }
    
    Serial.println();
    Serial.println("✅ ═══════════════════════════════════════════════════════════");
    Serial.println("✅            CALIBRAÇÃO CONCLUÍDA!                           ");
    Serial.println("✅ ═══════════════════════════════════════════════════════════");
    Serial.println();
    
    // Calcula os offsets (Hard-Iron correction)
    offset_x = (x_max + x_min) / 2.0f;
    offset_y = (y_max + y_min) / 2.0f;
    offset_z = (z_max + z_min) / 2.0f;
    
    // Calcula as escalas (Soft-Iron correction)
    float avg_delta_x = (x_max - x_min) / 2.0f;
    float avg_delta_y = (y_max - y_min) / 2.0f;
    float avg_delta_z = (z_max - z_min) / 2.0f;
    float avg_delta = (avg_delta_x + avg_delta_y + avg_delta_z) / 3.0f;
    
    scale_x = avg_delta / avg_delta_x;
    scale_y = avg_delta / avg_delta_y;
    scale_z = avg_delta / avg_delta_z;
    
    // Exibe resultados
    Serial.println("📊 VALORES BRUTOS COLETADOS:");
    Serial.printf("   X: min=%+6d  max=%+6d  (range=%d)\n", x_min, x_max, x_max - x_min);
    Serial.printf("   Y: min=%+6d  max=%+6d  (range=%d)\n", y_min, y_max, y_max - y_min);
    Serial.printf("   Z: min=%+6d  max=%+6d  (range=%d)\n", z_min, z_max, z_max - z_min);
    Serial.printf("   Total de amostras: %d\n", sampleCount);
    Serial.println();
    
    Serial.println("🧲 VALORES DE CALIBRAÇÃO CALCULADOS:");
    Serial.println();
    Serial.println("   Hard-Iron Offsets (para subtrair dos valores brutos):");
    Serial.printf("   offset_x = %.1f\n", offset_x);
    Serial.printf("   offset_y = %.1f\n", offset_y);
    Serial.printf("   offset_z = %.1f\n", offset_z);
    Serial.println();
    Serial.println("   Soft-Iron Scales (para multiplicar após subtrair offset):");
    Serial.printf("   scale_x = %.4f\n", scale_x);
    Serial.printf("   scale_y = %.4f\n", scale_y);
    Serial.printf("   scale_z = %.4f\n", scale_z);
    Serial.println();
    
    // Código pronto para copiar
    Serial.println("╔══════════════════════════════════════════════════════════════╗");
    Serial.println("║     COPIE O CÓDIGO ABAIXO PARA O ARQUIVO main.cpp            ║");
    Serial.println("╚══════════════════════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("// ===== CALIBRAÇÃO DO MAGNETÔMETRO QMC5883L =====");
    Serial.printf("const float MAG_OFFSET_X = %.1ff;\n", offset_x);
    Serial.printf("const float MAG_OFFSET_Y = %.1ff;\n", offset_y);
    Serial.printf("const float MAG_OFFSET_Z = %.1ff;\n", offset_z);
    Serial.printf("const float MAG_SCALE_X = %.4ff;\n", scale_x);
    Serial.printf("const float MAG_SCALE_Y = %.4ff;\n", scale_y);
    Serial.printf("const float MAG_SCALE_Z = %.4ff;\n", scale_z);
    Serial.println("// ===============================================");
    Serial.println();
    
    Serial.println("╔══════════════════════════════════════════════════════════════╗");
    Serial.println("║     ADICIONE ESTA FUNÇÃO OU MODIFIQUE read_QMC5883L()        ║");
    Serial.println("╚══════════════════════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("// No arquivo utils.cpp, modifique a função read_QMC5883L:");
    Serial.println("// Após ler x_raw, y_raw, z_raw, aplique a calibração:");
    Serial.println("//");
    Serial.println("//   // Aplica calibração Hard-Iron e Soft-Iron");
    Serial.printf("//   float x_cal = (x_raw - %.1ff) * %.4ff;\n", offset_x, scale_x);
    Serial.printf("//   float y_cal = (y_raw - %.1ff) * %.4ff;\n", offset_y, scale_y);
    Serial.printf("//   float z_cal = (z_raw - %.1ff) * %.4ff;\n", offset_z, scale_z);
    Serial.println("//");
    Serial.println("//   // Converte para µT");
    Serial.println("//   const float SCALE_FACTOR = 1.0f / 30.0f;");
    Serial.println("//   mx = x_cal * SCALE_FACTOR;");
    Serial.println("//   my = y_cal * SCALE_FACTOR;");
    Serial.println("//   mz = z_cal * SCALE_FACTOR;");
    Serial.println();
    
    // Verifica qualidade da calibração
    Serial.println("📋 VERIFICAÇÃO DA QUALIDADE:");
    float range_ratio_xy = (float)(x_max - x_min) / (y_max - y_min);
    float range_ratio_xz = (float)(x_max - x_min) / (z_max - z_min);
    float range_ratio_yz = (float)(y_max - y_min) / (z_max - z_min);
    
    bool good_cal = true;
    if (range_ratio_xy < 0.7 || range_ratio_xy > 1.4) {
        Serial.println("   ⚠️  Proporção X/Y fora do ideal. Tente girar mais no eixo Z.");
        good_cal = false;
    }
    if (range_ratio_xz < 0.7 || range_ratio_xz > 1.4) {
        Serial.println("   ⚠️  Proporção X/Z fora do ideal. Tente girar mais no eixo Y.");
        good_cal = false;
    }
    if (range_ratio_yz < 0.7 || range_ratio_yz > 1.4) {
        Serial.println("   ⚠️  Proporção Y/Z fora do ideal. Tente girar mais no eixo X.");
        good_cal = false;
    }
    if ((x_max - x_min) < 500 || (y_max - y_min) < 500 || (z_max - z_min) < 500) {
        Serial.println("   ⚠️  Range muito pequeno. Gire o sensor mais amplamente.");
        good_cal = false;
    }
    
    if (good_cal) {
        Serial.println("   ✅ Calibração parece boa! Os ranges estão proporcionais.");
    }
    Serial.println();
    
    Serial.println("═══════════════════════════════════════════════════════════════");
    Serial.println("  Agora o programa mostrará leituras calibradas em tempo real  ");
    Serial.println("  Verifique se o Azimute aponta corretamente para o Norte      ");
    Serial.println("═══════════════════════════════════════════════════════════════");
    Serial.println();
}

void loop() {
    int16_t x_raw, y_raw, z_raw;
    
    if (readRawQMC5883L(x_raw, y_raw, z_raw)) {
        // Aplica calibração
        float x_cal = (x_raw - offset_x) * scale_x;
        float y_cal = (y_raw - offset_y) * scale_y;
        float z_cal = (z_raw - offset_z) * scale_z;
        
        // Converte para µT
        const float SCALE_FACTOR = 1.0f / 30.0f;
        float mx = x_cal * SCALE_FACTOR;
        float my = y_cal * SCALE_FACTOR;
        float mz = z_cal * SCALE_FACTOR;
        
        // Calcula azimute (heading)
        float heading = atan2(my, mx) * RAD_TO_DEG;
        if (heading < 0) heading += 360.0f;
        
        // Intensidade total
        float intensity = sqrt(mx*mx + my*my + mz*mz);
        
        // Exibe valores
        Serial.printf("🧭 Mag: X=%+7.2f Y=%+7.2f Z=%+7.2f µT | Azimute: %6.1f° | Intensidade: %5.1f µT\n",
                     mx, my, mz, heading, intensity);
    }
    
    delay(200);
}
