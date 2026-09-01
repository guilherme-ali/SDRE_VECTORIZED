# -*- coding: utf-8 -*-
"""
Consolida N execucoes do ciclo de voo completo e reporta a FAIXA de cada
estatistica, nao um valor unico.

Por que: ao contrario dos benchmarks de solver, que sao deterministicos (mesmas
iteracoes e residuos bit-a-bit entre execucoes), o laco de voo e' um sistema em
tempo real com I2C, WiFi e blocos de impressao — ele varia entre execucoes. Duas
capturas do MESMO firmware deram 14 e 1 ciclos acima do periodo (0.029% e
0.002%). Reportar um numero unico sugere um determinismo que o dado nao tem.

Uso:
    python python/analisa_voo.py <captura1.txt> [captura2.txt ...]
    python python/analisa_voo.py --dir <pasta com voo_run*.txt>
"""
import argparse
import glob
import os
import re
import statistics as st

import numpy as np

# limiar de estouro: bin 120 do histograma de 50 us = 6.00 ms
BIN_US = 50
PERIODO_MS = 6.0
BIN_PERIODO = int(PERIODO_MS * 1000 / BIN_US)

ESTAGIOS = [(r"LQR .Ganhos.", "DARE solve"), ("Leitura MPU", "IMU read"),
            ("Matriz Sistema", "SDC matrix"), (r"C.lc. .ngulos", "Euler")]


def analisa(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocos = txt.split("STATUS DO SISTEMA")
    # Uma captura invalida NAO pode sumir em silencio: quatro janelas desta
    # campanha morreram com "Falha ao inicializar MPU6050!" (o firmware para em
    # while(1)) e teriam encolhido a consolidacao de 10 para 6 execucoes sem
    # nenhuma linha de aviso. Aqui cada descarte diz o motivo.
    if len(blocos) < 2:
        motivo = "sem blocos de status"
        if "Falha ao inicializar MPU6050" in txt:
            motivo = "firmware travou: IMU nao inicializou"
        print("  [DESCARTADA] %s -- %s" % (os.path.basename(path), motivo))
        return None
    last = blocos[-1]
    m = re.search(r"HIST_PROC_50US:([0-9,]+)", last)
    if not m:
        print("  [DESCARTADA] %s -- sem HIST_PROC_50US no ultimo bloco" % os.path.basename(path))
        return None
    h = [int(x) for x in m.group(1).split(",") if x]
    tot = sum(h)
    if tot == 0:
        print("  [DESCARTADA] %s -- histograma vazio" % os.path.basename(path))
        return None
    cdf = 100.0 * np.cumsum(h) / tot
    q = lambda p: (np.searchsorted(cdf, p) * BIN_US) / 1000.0

    mx = re.search(r"Processamento_Maximo:\s*(\d+)", last)
    ml = re.search(r"Tempo_Medio:\s*([\d.]+)", last)
    mp = re.search(r"Processamento_Medio:\s*([\d.]+)", last)
    # o firmware imprime o custo do bloco de diagnostico anterior; e' o termo que
    # fecha o balanco da janela de captura (ciclos x periodo + prints + boot)
    prints = [int(x) for x in re.findall(r"Prints \(ant\.\):\s*(\d+)", txt)]

    est = {}
    for pat, lbl in ESTAGIOS:
        v = []
        for b in blocos[1:]:
            mm = re.search(pat + r":\s*(\d+)\s*.s", b)
            if mm:
                v.append(int(mm.group(1)))
        if v:
            est[lbl] = v

    over = sum(h[BIN_PERIODO:])
    carimbo = re.search(r"STAMP,([^,\s]+),(\d+),", txt)
    return {
        "arquivo": os.path.basename(path),
        "ciclos": tot,
        "mediana": q(50), "p99": q(99), "p999": q(99.9),
        "max": int(mx.group(1)) / 1e3 if mx else float("nan"),
        "periodo": float(ml.group(1)) / 1e3 if ml else float("nan"),
        "estouros": over, "estouros_pct": 100.0 * over / tot,
        "n_prints": len(prints), "t_prints": sum(prints) / 1e6,
        "estagios": est,
        "hist": h,
        "media": float(mp.group(1)) / 1e3 if mp else float("nan"),
        "commit": carimbo.group(1) if carimbo else None,
        "dirty": carimbo.group(2) if carimbo else None,
    }


def faixa(vals, fmt="%.2f"):
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return fmt % lo
    return (fmt + " a " + fmt) % (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capturas", nargs="*")
    ap.add_argument("--dir", help="pasta com voo_run*.txt")
    a = ap.parse_args()

    paths = list(a.capturas)
    if a.dir:
        paths += sorted(glob.glob(os.path.join(a.dir, "voo_run*.txt")))
    if not paths:
        raise SystemExit("informe as capturas ou --dir")

    rs = [r for r in (analisa(p) for p in paths) if r]
    if not rs:
        raise SystemExit("nenhuma captura valida")
    if len(rs) < len(paths):
        print("  -> %d de %d capturas aproveitadas" % (len(rs), len(paths)))

    print("%-16s %8s %8s %7s %7s %7s %9s %10s" %
          ("execucao", "ciclos", "mediana", "p99", "p99.9", "max", "periodo", "estouros"))
    print("-" * 82)
    for r in rs:
        print("%-16s %8d %8.2f %7.2f %7.2f %7.2f %9.4f %5d (%.3f%%)" %
              (r["arquivo"], r["ciclos"], r["mediana"], r["p99"], r["p999"],
               r["max"], r["periodo"], r["estouros"], r["estouros_pct"]))
    print("-" * 82)

    print("\n%d execucoes do mesmo firmware:" % len(rs))
    print("  ciclos          : %s" % faixa([r["ciclos"] for r in rs], "%d"))
    print("  mediana (ms)    : %s" % faixa([r["mediana"] for r in rs]))
    print("  p99 (ms)        : %s" % faixa([r["p99"] for r in rs]))
    print("  p99.9 (ms)      : %s" % faixa([r["p999"] for r in rs]))
    print("  maximo (ms)     : %s" % faixa([r["max"] for r in rs]))
    print("  periodo (ms)    : %s" % faixa([r["periodo"] for r in rs], "%.4f"))
    print("  estouros        : %s  (%s)" %
          (faixa([r["estouros"] for r in rs], "%d"),
           faixa([r["estouros_pct"] for r in rs], "%.3f%%")))
    tot_c = sum(r["ciclos"] for r in rs)
    tot_o = sum(r["estouros"] for r in rs)
    print("  AGREGADO        : %d de %d ciclos acima do periodo (%.3f%%)" %
          (tot_o, tot_c, 100.0 * tot_o / tot_c))

    print("\n  estagios (mediana de cada execucao, ms):")
    for lbl in [l for _, l in ESTAGIOS]:
        v = [st.median(r["estagios"][lbl]) / 1e3 for r in rs if lbl in r["estagios"]]
        if v:
            print("    %-12s %s" % (lbl, faixa(v, "%.3f")))

    # ------------------------------------------------------------------
    # Populacao agregada: somar os histogramas de 50 us e' exato (os bins
    # sao os mesmos em toda execucao), entao os percentis abaixo sao os do
    # conjunto de ciclos -- nao a media dos percentis por execucao. E' esta
    # a estatistica que o artigo cita; a faixa acima diz quanto ela balanca.
    # ------------------------------------------------------------------
    n = max(len(r["hist"]) for r in rs)
    hp = [0] * n
    for r in rs:
        for i, c in enumerate(r["hist"]):
            hp[i] += c
    tot = sum(hp)
    cdf = 100.0 * np.cumsum(hp) / tot
    qp = lambda p: (np.searchsorted(cdf, p) * BIN_US) / 1000.0
    med_pond = sum(r["media"] * r["ciclos"] for r in rs) / tot

    print("\n  AGREGADO sobre %d ciclos (%d execucoes):" % (tot, len(rs)))
    print("    mediana         : %.2f ms (bin [%.2f, %.2f) ms)"
          % (qp(50), qp(50), qp(50) + BIN_US / 1000.0))
    print("    media           : %.2f ms" % med_pond)
    print("    p99             : %.2f ms" % qp(99))
    print("    p99.9           : %.2f ms" % qp(99.9))
    print("    p99.99          : %.2f ms" % qp(99.99))
    print("    maximo por exec.: %s ms" % faixa([r["max"] for r in rs]))
    tot_o = sum(r["estouros"] for r in rs)
    print("    acima de %.1f ms : %d de %d (%.3f%%)"
          % (PERIODO_MS, tot_o, tot, 100.0 * tot_o / tot))

    print("\n  estagios agregados (todas as amostras, ms):")
    for lbl in [l for _, l in ESTAGIOS]:
        v = [x for r in rs for x in r["estagios"].get(lbl, [])]
        if v:
            print("    %-12s n=%-6d mediana %.3f | p99 %.3f | max %.3f"
                  % (lbl, len(v), st.median(v) / 1e3,
                     np.percentile(v, 99) / 1e3, max(v) / 1e3))

    print("\n  balanco da janela de captura (por execucao):")
    for r in rs:
        ciclos_s = r["ciclos"] * r["periodo"] / 1e3
        print("    %-14s %7.2f s de ciclos + %6.2f s em %d blocos de print = %7.2f s"
              % (r["arquivo"], ciclos_s, r["t_prints"], r["n_prints"],
                 ciclos_s + r["t_prints"]))

    commits = {r["commit"] for r in rs}
    print("\n  procedencia     : %s" %
          (", ".join(str(c) for c in commits) if commits != {None} else "SEM CARIMBO"))


if __name__ == "__main__":
    main()
