// Shim de ArduinoEigen.h para o host: aponta direto para o Eigen puro
// já vendorizado pelo PlatformIO em .pio/libdeps/esp32-s2-saola-1/ArduinoEigen.
#ifndef ARDUINO_EIGEN_SHIM_H
#define ARDUINO_EIGEN_SHIM_H

#define EIGEN_MPL2_ONLY
#include <Eigen/Eigen>

#endif // ARDUINO_EIGEN_SHIM_H
