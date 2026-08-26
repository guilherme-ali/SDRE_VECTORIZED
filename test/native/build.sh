#!/usr/bin/env bash
# Compila lib/AUTOLQR no host (g++), sem tocar em nenhuma linha da lib,
# usando shims de Arduino.h/ArduinoEigen.h e o Eigen já vendorizado pelo
# PlatformIO. Objetivo: isolar erro algébrico de erro de precisão do alvo.
set -euo pipefail
cd "$(dirname "$0")/../.."

EIGEN_DIR=".pio/libdeps/esp32-s2-saola-1/ArduinoEigen/ArduinoEigen"
if [ ! -d "$EIGEN_DIR" ]; then
    echo "Eigen vendorizado não encontrado em $EIGEN_DIR" >&2
    echo "Rode 'pio pkg install' no env esp32-s2-saola-1 primeiro." >&2
    exit 1
fi

mkdir -p build

g++ -std=gnu++17 -O2 -Wall -Wextra -Wno-unused-parameter \
    -static-libgcc -static-libstdc++ -static \
    -I test/native/shim \
    -I "$EIGEN_DIR" \
    -I lib/AUTOLQR \
    lib/AUTOLQR/AutoLQR.cpp \
    lib/AUTOLQR/MatrixOperations.cpp \
    lib/AUTOLQR/FixedPointQ.cpp \
    test/native/verify_solvers.cpp \
    -o build/verify_solvers.exe

# Link estático: em Git Bash, com PATH que mistura toolchains MinGW distintos
# (ex.: Strawberry Perl's g++ + runtime DLLs do MSYS2 mingw64 achados antes na
# PATH), um binário dinâmico pode carregar libstdc++/libgcc de uma build
# incompatível e falhar silenciosamente com exit 127, sem imprimir nada —
# visto na prática ao rodar esta harness (ver docs/auditoria_solvers_riccati.md,
# Seção 13). Estático elimina a dependência de runtime em tempo de execução.

echo "OK: build/verify_solvers.exe"
