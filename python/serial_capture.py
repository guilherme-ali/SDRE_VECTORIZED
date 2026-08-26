"""
Biblioteca de captura serial para os experimentos do ESP32-S2 (SDRE_VECTORIZED).

Resgatada e generalizada a partir dos scripts ad-hoc usados durante a campanha de
medicao (capture_generic.py, capture_115200.py) que viviam fora do repositorio.
Usada por python/run_experiments.py; pode tambem ser chamada isoladamente:

    python -m python.serial_capture COM3 outputs/serial_gamma_sweep.txt 400 \
        "FIM DA VARREDURA DE GAMMA"
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyserial nao encontrado. Instale com: pip install -r requirements.txt"
    ) from exc


# Identificadores USB tipicamente usados pelas pontes serie do ESP32-S2 Saola
# (CP210x da Silicon Labs) e placas correlatas (CH340). O ESP32-S2 tambem pode
# aparecer como USB-CDC nativo (VID 0x303A, Espressif).
_KNOWN_VID_PID = {
    (0x10C4, 0xEA60),  # Silicon Labs CP2102/CP2104
    (0x1A86, 0x7523),  # QinHeng CH340
    (0x303A, None),     # Espressif (USB-CDC nativo) - qualquer PID
}


@dataclass
class CaptureResult:
    elapsed_s: float
    n_bytes: int
    done_flag: bool
    outfile: str


def list_serial_ports() -> List["serial.tools.list_ports_common.ListPortInfo"]:
    return list(serial.tools.list_ports.comports())


def find_esp32_port(preferred: Optional[str] = None) -> str:
    """Autodetecta a porta serial do ESP32-S2.

    Se `preferred` for passado, apenas valida que a porta existe e a retorna.
    Caso contrario, procura por VID/PID conhecidos (CP210x, CH340, Espressif).
    Levanta SystemExit com mensagem acionavel se nao encontrar exatamente uma.
    """
    ports = list_serial_ports()

    if preferred:
        for p in ports:
            if p.device.upper() == preferred.upper():
                return p.device
        available = ", ".join(p.device for p in ports) or "(nenhuma)"
        raise SystemExit(
            f"Porta '{preferred}' nao encontrada. Portas disponiveis: {available}"
        )

    candidates = []
    for p in ports:
        if p.vid is None:
            continue
        for vid, pid in _KNOWN_VID_PID:
            if p.vid == vid and (pid is None or p.pid == pid):
                candidates.append(p)
                break

    if len(candidates) == 1:
        return candidates[0].device

    if not candidates:
        available = "\n".join(f"  {p.device}  {p.description}" for p in ports) or "  (nenhuma porta serial detectada)"
        raise SystemExit(
            "Nao foi possivel autodetectar o ESP32-S2 (nenhum VID/PID conhecido "
            "encontrado — CP210x, CH340 ou Espressif USB-CDC).\n"
            f"Portas vistas pelo sistema:\n{available}\n"
            "Passe a porta explicitamente com --port COMx (Windows) ou --port /dev/ttyUSBx (Linux)."
        )

    listing = "\n".join(f"  {p.device}  {p.description}" for p in candidates)
    raise SystemExit(
        f"Mais de uma porta candidata a ESP32-S2 encontrada:\n{listing}\n"
        "Especifique qual usar com --port COMx."
    )


def reset_board(ser: "serial.Serial") -> None:
    """Forca reset via o circuito auto-reset do CP210x/CH340 (toggle DTR/RTS)."""
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    time.sleep(0.1)
    ser.dtr = True


def capture(
    port: str,
    baud: int,
    outfile: str,
    markers: Optional[Iterable[str]] = None,
    timeout_s: float = 3600.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    echo_prefixes: Iterable[str] = ("#", "EXP,"),
    grace_read_s: float = 3.0,
) -> CaptureResult:
    """Captura dados seriais do ESP32-S2 para `outfile`, com escrita incremental.

    - Se `markers` for uma lista nao-vazia: para assim que uma linha contendo
      qualquer um dos marcadores for vista (mais uma leitura de folga de
      `grace_read_s` segundos para capturar o que ainda estiver em transito),
      ou quando `timeout_s` for atingido (o que ocorrer primeiro).
    - Se `markers` for None/vazio: captura por uma janela de tempo fixa de
      `timeout_s` segundos (usado no experimento de voo, que nao emite marcador
      de fim porque roda em loop continuo).

    Escreve em disco a cada chunk recebido (flush explicito) — capturas longas
    (horas) nao podem viver so em memoria.
    """
    marker_list = list(markers) if markers else []
    echo_tuple = tuple(echo_prefixes)

    ser = serial.Serial(port, baud, timeout=1.0)
    reset_board(ser)

    start = time.time()
    buf = b""
    done = False
    n_bytes = 0

    with open(outfile, "wb") as f:
        while time.time() - start < timeout_s:
            chunk = ser.read(8192)
            if chunk:
                f.write(chunk)
                f.flush()
                n_bytes += len(chunk)

                if marker_list or progress_cb:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode("utf-8", errors="replace").rstrip("\r")
                        if progress_cb and text.startswith(echo_tuple):
                            progress_cb(text)
                        for marker in marker_list:
                            if marker in text:
                                done = True
            if done:
                time.sleep(grace_read_s)
                extra = ser.read(200000)
                if extra:
                    f.write(extra)
                    n_bytes += len(extra)
                break

    ser.close()
    elapsed = time.time() - start
    return CaptureResult(elapsed_s=elapsed, n_bytes=n_bytes, done_flag=done, outfile=outfile)


def _main() -> None:
    if len(sys.argv) < 3:
        print(
            "uso: python -m python.serial_capture PORTA ARQUIVO_SAIDA "
            "[TIMEOUT_S] [MARCADOR ...]"
        )
        raise SystemExit(2)

    port = sys.argv[1]
    outfile = sys.argv[2]
    timeout_s = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
    markers = sys.argv[4:] if len(sys.argv) > 4 else None
    baud = 921600 if markers else 115200

    result = capture(
        port,
        baud,
        outfile,
        markers=markers,
        timeout_s=timeout_s,
        progress_cb=lambda line: print(line, flush=True),
    )
    print(
        f"CAPTURE_DONE elapsed={result.elapsed_s:.1f}s done_flag={result.done_flag} "
        f"bytes={result.n_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    _main()
