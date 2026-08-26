#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <Arduino.h>
#include <LittleFS.h>

/**
 * Buffer circular em RAM para telemetria de voo.
 * Captura amostras a cada chamada de log() sem alocacao dinamica.
 * Dump em CSV via Serial sob demanda (apos pouso, drone parado).
 *
 * Persistencia em flash (LittleFS): saveToFile() chamado ao desarmar.
 * loadFromFile() no setup() restaura buffer apos reboot/reset do ESP32
 * (ESP32 reseta quando Serial Monitor abre - sem isso o buffer some).
 *
 * Tamanho: CAPACITY * sizeof(Sample) bytes.
 * Default: 800 amostras x 76 bytes (68 B de estado + 6 B de custo/desfecho do
 * solver + 2 B de padding de alinhamento) = ~60,8 KB em RAM (ver a nota sobre
 * o teto de dram0_0_seg em CAPACITY).
 *
 * Custo por log(): apenas escritas em RAM (~1 us). Zero printf, zero I/O.
 * Custo de saveToFile(): ~100-300 ms - so chamado apos pouso (motor parado).
 */
class Telemetry {
public:
    struct Sample {
        uint32_t t_ms;
        float roll, pitch, yaw;           // rad (medido)
        float roll_ref, pitch_ref, yaw_ref; // rad (referencia)
        float p, q, r;                    // rad/s (taxas no corpo)
        float u0, u1, u2;                 // torques SDRE [N·m]: roll, pitch, yaw
        float w1_sq, w2_sq, w3_sq, w4_sq; // rad^2/s^2
        // ---- Custo do ciclo e desfecho do solver (Exp. E do plano) ----
        uint16_t processingTime;          // us: inicio do ciclo -> fim do trabalho util
                                          //     (ANTES do padding de periodo; satura em 65535)
        uint16_t t_lqr;                   // us: updateSystemMatrix + computeGains no modo
                                          //     sincrono; 0 no async (Riccati roda na SDRETask,
                                          //     logo FORA de processingTime)
        uint8_t  iters;                   // iteracoes da ultima chamada a computeGains()
        uint8_t  outcome;                 // AutoLQR::SolveOutcome: 0=Converged 1=Budget 2=Breakdown
    };

    // CAPACITY x sizeof(Sample) e' o maior bloco estatico do firmware e o que
    // define a folga de dram0_0_seg. Medido no link do env esp32-s2-saola-1: o
    // teto para ESTE buffer e' 68.144 B (com 1000 x 76 B = 76.000 B o linker
    // acusa "dram0_0_seg overflowed by 7856 bytes"). Ou seja, a versao antiga
    // (1000 x 68 B = 68.000 B) cabia com apenas 144 B de sobra — nao havia
    // espaco para os 6 B/amostra de custo de ciclo sem reduzir a capacidade.
    // 800 x 76 B = 60.800 B: cabe com ~7,3 KB a mais de DRAM livre que o
    // firmware anterior, ao custo de 800*5*5,2ms = 20,8 s de janela de voo
    // gravada (era 26 s) — ver TELEMETRY_DECIMATION_CYCLES em src/main.cpp.
    static constexpr size_t CAPACITY = 800;

    Telemetry() : head(0), count(0) {}

    inline void log(uint32_t t_ms,
                    float roll, float pitch, float yaw,
                    float roll_ref, float pitch_ref, float yaw_ref,
                    float p, float q, float r,
                    float u0, float u1, float u2,
                    float w1_sq, float w2_sq, float w3_sq, float w4_sq,
                    uint16_t processingTime = 0, uint16_t t_lqr = 0,
                    uint8_t iters = 0, uint8_t outcome = 0) {
        Sample &s = buf[head];
        s.t_ms = t_ms;
        s.roll = roll;  s.pitch = pitch;  s.yaw = yaw;
        s.roll_ref = roll_ref; s.pitch_ref = pitch_ref; s.yaw_ref = yaw_ref;
        s.p = p;        s.q = q;          s.r = r;
        s.u0 = u0;      s.u1 = u1;        s.u2 = u2;
        s.w1_sq = w1_sq; s.w2_sq = w2_sq; s.w3_sq = w3_sq; s.w4_sq = w4_sq;
        s.processingTime = processingTime; s.t_lqr = t_lqr;
        s.iters = iters; s.outcome = outcome;
        head = (head + 1) % CAPACITY;
        if (count < CAPACITY) count++;
    }

    void reset() {
        head = 0;
        count = 0;
    }

    size_t size() const { return count; }

    /**
     * Salva buffer em LittleFS (binario, raw struct dump).
     * Chamar APENAS com motores parados (operacao lenta, ~100-300 ms).
     * Retorna true em sucesso.
     */
    bool saveToFile(const char* path = "/telem.bin") {
        File f = LittleFS.open(path, "w");
        if (!f) return false;
        // v4 struct: acrescenta processingTime/t_lqr/iters/outcome. Magic
        // diferente do v3 ("TELR") de proposito — um telem.bin gravado pelo
        // firmware antigo tem layout incompativel e deve ser rejeitado em vez
        // de lido como lixo.
        uint32_t magic = 0x54454C53; // "TELS" - v4 struct
        f.write((uint8_t*)&magic, sizeof(magic));
        uint32_t cap = CAPACITY;
        f.write((uint8_t*)&cap, sizeof(cap));
        uint32_t h = (uint32_t)head;
        uint32_t c = (uint32_t)count;
        f.write((uint8_t*)&h, sizeof(h));
        f.write((uint8_t*)&c, sizeof(c));
        f.write((uint8_t*)buf, sizeof(Sample) * CAPACITY);
        f.close();
        return true;
    }

    /**
     * Carrega buffer de LittleFS. Retorna true se arquivo existe e e valido.
     * Chamar no setup() apos LittleFS.begin().
     */
    bool loadFromFile(const char* path = "/telem.bin") {
        if (!LittleFS.exists(path)) return false;
        File f = LittleFS.open(path, "r");
        if (!f) return false;
        uint32_t magic = 0, cap = 0, h = 0, c = 0;
        f.read((uint8_t*)&magic, sizeof(magic));
        f.read((uint8_t*)&cap, sizeof(cap));
        if (magic != 0x54454C53 || cap != CAPACITY) {
            f.close();
            return false;
        }
        f.read((uint8_t*)&h, sizeof(h));
        f.read((uint8_t*)&c, sizeof(c));
        f.read((uint8_t*)buf, sizeof(Sample) * CAPACITY);
        f.close();
        head = (size_t)h;
        count = (size_t)c;
        return true;
    }

    void dumpCSV(Stream &out) {
        out.println();
        out.println("=== TELEMETRY DUMP START ===");
        out.print("Samples: "); out.println((unsigned long)count);
        out.println("t_ms,roll_deg,pitch_deg,yaw_deg,roll_ref,pitch_ref,yaw_ref,p_dps,q_dps,r_dps,u_roll,u_pitch,u_yaw,w1_sq,w2_sq,w3_sq,w4_sq,processing_us,t_lqr_us,iters,outcome");
        // outcome: 0=Converged 1=Budget 2=Breakdown (AutoLQR::SolveOutcome);
        // iters=255 => telemetria indisponivel (sentinela -1 do solver).

        const float RAD2DEG = 57.29578f;
        size_t start = (count < CAPACITY) ? 0 : head;
        for (size_t i = 0; i < count; i++) {
            const Sample &s = buf[(start + i) % CAPACITY];
            out.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.1f,%.1f,%.1f,%.1f,%u,%u,%u,%u\n",
                       (unsigned long)s.t_ms,
                       s.roll * RAD2DEG, s.pitch * RAD2DEG, s.yaw * RAD2DEG,
                       s.roll_ref * RAD2DEG, s.pitch_ref * RAD2DEG, s.yaw_ref * RAD2DEG,
                       s.p * RAD2DEG, s.q * RAD2DEG, s.r * RAD2DEG,
                       s.u0, s.u1, s.u2,
                       s.w1_sq, s.w2_sq, s.w3_sq, s.w4_sq,
                       (unsigned)s.processingTime, (unsigned)s.t_lqr,
                       (unsigned)s.iters, (unsigned)s.outcome);
        }
        out.println("=== TELEMETRY DUMP END ===");
    }

private:
    Sample buf[CAPACITY];
    size_t head;
    size_t count;
};

#endif
