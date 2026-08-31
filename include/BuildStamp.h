#ifndef BUILD_STAMP_H
#define BUILD_STAMP_H

#include <Arduino.h>

// ---------------------------------------------------------------------------
// Carimbo de procedencia emitido no cabecalho de toda captura serial.
//
// Sem isto, um arquivo em outputs/ nao diz de qual firmware veio: foi assim que
// uma captura anterior a otimizacao push-through do ADDA sobreviveu em duas
// versoes do artigo, reportando 12.17 ms onde o codigo atual mede 9.62 ms. A
// linha STAMP amarra cada CSV a um commit, a um estado de arvore (dirty ou nao),
// a um instante de build e ao chip/clock que de fato executou.
//
// GIT_REV/GIT_DIRTY/BUILD_EPOCH vem de scripts/build_stamp.py (extra_scripts).
// Os defaults abaixo mantem o header compilavel fora do PlatformIO.
// ---------------------------------------------------------------------------
#ifndef GIT_REV
#define GIT_REV "unknown"
#endif
#ifndef GIT_DIRTY
#define GIT_DIRTY -1
#endif
#ifndef BUILD_EPOCH
#define BUILD_EPOCH 0
#endif

namespace buildstamp {

/**
 * Imprime uma linha CSV auto-identificavel:
 *   STAMP,<git_rev>,<dirty>,<build_epoch>,<compilado>,<chip>,<revisao>,<nucleos>,<mhz>
 * Chamar como PRIMEIRA saida de qualquer firmware de experimento, antes do
 * cabecalho de colunas — um consumidor que nao a encontre deve tratar a captura
 * como de procedencia desconhecida.
 */
inline void print() {
    // Emitida DUAS vezes, cada uma precedida de nova linha e seguida de flush.
    // O host abre a porta com a placa ja' transmitindo, entao os primeiros bytes
    // sao rotineiramente perdidos ou colados a lixo de sincronizacao: na
    // primeira campanha carimbada a linha saiu truncada em "STAMP,ab678c8-di".
    // Uma segunda copia, ja' com o fluxo sincronizado, garante ao menos um
    // carimbo integro; o consumidor (python/verifica_procedencia.py) aceita
    // qualquer ocorrencia bem formada em qualquer posicao do arquivo.
    for (int i = 0; i < 2; i++) {
        Serial.printf("\nSTAMP,%s,%d,%lu,%s %s,%s,%d,%d,%u\n",
                      GIT_REV, (int)GIT_DIRTY, (unsigned long)BUILD_EPOCH,
                      __DATE__, __TIME__,
                      ESP.getChipModel(), ESP.getChipRevision(), ESP.getChipCores(),
                      (unsigned)getCpuFrequencyMhz());
        Serial.flush();
        delay(120);
    }
}

} // namespace buildstamp

#endif
