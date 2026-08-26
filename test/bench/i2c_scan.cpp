// Scanner I2C minimo para diagnosticar o MPU6050 (ver test/main_backup.cpp /
// lib/utils/utils.cpp:19 -- "Falha ao inicializar MPU6050!"). Sem depender de
// nenhuma biblioteca de sensor: varre 0x03-0x77 nos pinos SDA=11/SCL=10 (os
// mesmos do firmware de voo) e imprime quem responde.
#include <Arduino.h>
#include <Wire.h>

void setup() {
    Serial.begin(115200);
    unsigned long t0 = millis();
    while (!Serial && millis() - t0 < 3000) {}
    delay(1000);

    Serial.println("=== Scanner I2C (SDA=GPIO11, SCL=GPIO10) ===");
    Wire.begin(11, 10);
    Wire.setClock(100000); // 100 kHz p/ diagnostico -- mais tolerante a fiacao/pull-ups fracos
    Wire.setTimeOut(50);   // ms -- evita travar para sempre se o barramento estiver preso

    int found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Serial.printf("  testando 0x%02X...\n", addr);
        Serial.flush();
        Wire.beginTransmission(addr);
        uint8_t err = Wire.endTransmission();
        if (err == 0) {
            Serial.printf("  >>> Dispositivo encontrado em 0x%02X\n", addr);
            found++;
        }
        delay(5);
    }
    Serial.printf("=== Fim do scan: %d dispositivo(s) encontrado(s) ===\n", found);
    if (found == 0) {
        Serial.println("Nenhum dispositivo respondeu -- verificar fiacao SDA/SCL, pull-ups e alimentacao 3V3 do MPU6050.");
    }
}

void loop() {
    delay(5000);
}
