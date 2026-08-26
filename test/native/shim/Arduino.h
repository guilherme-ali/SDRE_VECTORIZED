// Shim mínimo de Arduino.h para compilar lib/AUTOLQR no host (g++ nativo).
// Cobre só o que AutoLQR.cpp / MatrixOperations.cpp realmente usam:
// Serial.print/println (com overload de precisão float) e a macro F().
#ifndef ARDUINO_SHIM_H
#define ARDUINO_SHIM_H

#include <cstdio>
#include <cstring>
#include <cstdint>

#define F(x) (x)

// No host, ESP32 nao esta definido, entao lib/AUTOLQR nunca inclui esp_attr.h
// (que definiria IRAM_ATTR de verdade). As funcoes de MatrixOperations.cpp usam
// IRAM_ATTR incondicionalmente (nao via MATRIX_FAST_ATTR), entao precisa existir
// aqui como no-op para compilar sem tocar na lib.
#ifndef IRAM_ATTR
#define IRAM_ATTR
#endif

struct SerialShim {
    void print(const char* s) const { std::fputs(s, stdout); }
    void print(int v) const { std::printf("%d", v); }
    void print(float v, int decimals) const { std::printf("%.*f", decimals, v); }
    void println(const char* s) const { std::fputs(s, stdout); std::fputc('\n', stdout); }
    void println() const { std::fputc('\n', stdout); }
};
static SerialShim Serial;

using String = const char*;
using byte = uint8_t;
using boolean = bool;

#endif // ARDUINO_SHIM_H
