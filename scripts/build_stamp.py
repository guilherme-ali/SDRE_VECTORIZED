"""
Injeta procedencia no firmware em tempo de compilacao (PlatformIO extra_script).

Motivacao: ate aqui nenhuma captura serial dizia de qual build ela veio. Foi
assim que uma captura da bateria anterior a otimizacao push-through do ADDA
sobreviveu em duas versoes do artigo reportando 12.17 ms onde o codigo do
repositorio produz 9.62 ms. Sem carimbo, um .txt em outputs/ e' indistinguivel
de um gerado seis meses antes por outro codigo.

Define tres macros, consumidas por experiments/*.cpp no cabecalho do CSV:
  GIT_REV     hash curto do commit (+ "-dirty" se ha alteracao nao commitada)
  GIT_DIRTY   1 se a arvore tem modificacoes, 0 caso contrario
  BUILD_EPOCH segundos desde a epoca no momento do link

Uso no platformio.ini de cada env de experimento:
    extra_scripts = pre:scripts/build_stamp.py
"""
import subprocess
import time

Import("env")


def _git(*args):
    try:
        return subprocess.check_output(["git"] + list(args), stderr=subprocess.DEVNULL,
                                        text=True).strip()
    except Exception:
        return ""


rev = _git("rev-parse", "--short", "HEAD") or "unknown"
dirty = 1 if _git("status", "--porcelain") else 0
if dirty:
    rev += "-dirty"

env.Append(CPPDEFINES=[
    ("GIT_REV", env.StringifyMacro(rev)),
    ("GIT_DIRTY", str(dirty)),
    ("BUILD_EPOCH", str(int(time.time()))),
])

print("build_stamp: GIT_REV=%s GIT_DIRTY=%d" % (rev, dirty))
