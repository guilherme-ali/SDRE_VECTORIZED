# -*- coding: utf-8 -*-
"""
Recalcula, a partir das capturas correntes em outputs/, os numeros que o artigo
cita, e imprime-os no formato em que aparecem no .tex. Serve para dois fins:
conferir o artigo contra o dado, e reescrever as tabelas apos uma recaptura sem
digitar nada a mao.

Uso:  python python/numeros_artigo.py [--tabelas]
"""
import argparse
import collections
import os
import statistics as st
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
sys.path.insert(0, os.path.join(REPO, "python"))

DBL = ["SDA", "SDA_SS", "ADDA", "SDA_SCALED", "ASDA"]
LBL = {"SDA": "SDA", "SDA_SS": "SDA-SS", "ADDA": "ADDA", "SDA_SCALED": "SDA-Scaled",
       "ASDA": "ASDA", "ITERATIVE": "Value iteration"}
ORDER1 = ["SDA", "SDA_SS", "ADDA", "SDA_SCALED", "ASDA", "ITERATIVE",
          "SDA_FIXED", "SDA_SCALED_FIXED", "SDA_SS_FIXED", "ASDA_FIXED",
          "ADDA_FIXED", "ITERATIVE_FIXED"]
LBL1 = dict(LBL)
LBL1.update({"SDA_FIXED": "SDA-fx", "SDA_SCALED_FIXED": "SDA-Scaled-fx",
             "SDA_SS_FIXED": "SDA-SS-fx", "ASDA_FIXED": "ASDA-fx",
             "ADDA_FIXED": "ADDA-fx", "ITERATIVE_FIXED": "Value iteration-fx"})


def captura_completa(path, n_pontos_esperado=60000):
    """True so se a captura chegou ao fim: marcador de conclusao presente E
    todo metodo com n_total == n_pontos_esperado. Sem isto, uma captura
    interrompida (placa desconectada, timeout do monitor) produz medianas
    calculadas sobre as primeiras trajetorias apenas — numeros plausiveis e
    silenciosamente errados, que foi exatamente o que aconteceu na primeira
    tentativa de recaptura do ESP32-S3."""
    if not os.path.isfile(path):
        return False, "arquivo ausente"
    txt = open(path, encoding="utf-8", errors="replace").read()
    if "FIM DO BENCHMARK" not in txt:
        return False, "sem marcador de conclusao (captura interrompida)"
    tot = {}
    for line in txt.splitlines():
        if line.startswith("SUMMARY,ALL,"):
            p = line.split(",")
            if len(p) >= 14:
                tot[p[2]] = int(p[13])
    if not tot:
        return False, "sem linhas SUMMARY,ALL"
    ruins = {m: n for m, n in tot.items() if n != n_pontos_esperado}
    if ruins:
        return False, "n_total != %d em %s" % (n_pontos_esperado, ruins)
    return True, "ok (%d metodos, %d pontos cada)" % (len(tot), n_pontos_esperado)


def load_runs(path):
    t = collections.defaultdict(list)
    it = collections.defaultdict(list)
    res = collections.defaultdict(list)
    conv = collections.Counter()
    tot = collections.Counter()
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("RUN,"):
            p = line.rstrip().split(",")
            if len(p) < 8:
                continue
            m = p[3]
            t[m].append(int(p[4]))
            it[m].append(int(p[5]))
            res[m].append(float(p[6]))
        elif line.startswith("SUMMARY,ALL,"):
            # SUMMARY,ALL,metodo,mean_us,std_us,max_us,mean_iters,max_iters,
            #   mean_res,max_res,n_converged,n_budget,n_breakdown,n_total
            p = line.strip().split(",")
            if len(p) < 14:
                continue
            conv[p[2]] = int(p[10])
            tot[p[2]] = int(p[13])
    return t, it, res, conv, tot


def gain_errors():
    """e_K = ||K_dev - K_ref||_F / ||K_ref||_F por metodo, das linhas GAIN.

    A referencia e' resolvida NO PONTO QUE O DISPOSITIVO REPORTOU (linha PT),
    nao num ponto regenerado em Python. Regenerar introduzia descasamento de
    modelo em T4 — onde o alvo do degrau vem do sinal de um seno e a grade de
    6 ms cai sobre os zeros — que aparecia como se fosse erro de solver: 4.18e-4
    contra 1.5e-6 no ponto real, inflando a coluna float da Tabela 1 de 1.9e-6
    para 3.7e-6. Ver python/pontos_dispositivo.py.
    """
    import pontos_dispositivo as pd_
    from bench_trajetorias import build_Ad_Bd_Qd_Rd, k_from_P
    from scipy.linalg import solve_discrete_are

    dados = pd_.carregar()
    gains = collections.defaultdict(list)
    path = os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt")
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("GAIN,"):
            continue
        p = line.rstrip().split(",")
        if len(p) < 22:
            continue
        gains[(p[1], int(p[2]))].append((p[3], p[4:22]))

    eK = collections.defaultdict(list)
    for (traj, k), lst in gains.items():
        d = dados.get(traj)
        if d is None or k >= len(d["t"]):
            continue
        Ad, Bd, Qd, Rd = build_Ad_Bd_Qd_Rd(d["phi"][k], d["theta"][k],
                                            d["p"][k], d["q"][k], d["r"][k])
        try:
            P = solve_discrete_are(Ad, Bd, Qd, Rd)
        except Exception:
            continue
        Kref = k_from_P(Ad, Bd, Qd, Rd, P)
        nref = np.linalg.norm(Kref)
        if nref <= 0:
            continue
        for m, vals in lst:
            Kdev = np.array([float(x) for x in vals]).reshape(3, 6)
            eK[m].append(float(np.linalg.norm(Kdev - Kref) / nref))
    return eK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabelas", action="store_true", help="emite os tabulars LaTeX")
    args = ap.parse_args()

    s2f = os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt")
    ok2, msg2 = captura_completa(s2f)
    print("captura S2: %s - %s" % ("COMPLETA" if ok2 else "INCOMPLETA", msg2))
    if not ok2:
        raise SystemExit("Abortado: a bateria S2 nao esta completa, medianas seriam parciais.")
    t2, i2, r2, c2, n2 = load_runs(s2f)

    s3f = os.path.join(OUT, "s3", "serial_capture_bateria_s3.txt")
    have_s3, msg3 = captura_completa(s3f)
    print("captura S3: %s - %s\n" % ("COMPLETA" if have_s3 else "INCOMPLETA", msg3))
    t3 = {}
    if have_s3:
        t3, i3, r3, _, _ = load_runs(s3f)

    eK = gain_errors()

    print("=" * 94)
    print("TABELA 1  (bateria S2, tau=1e-3)")
    print("=" * 94)
    print("%-20s %8s %10s %11s %10s %10s  %s" %
          ("Method", "t50(ms)", "t99.9(ms)", "Iterations", "Residual", "e_K", "Converged"))
    for m in ORDER1:
        if m not in t2:
            print("  %-18s AUSENTE" % m)
            continue
        e = st.median(eK[m]) if eK.get(m) else float("nan")
        print("%-20s %8.2f %10.2f %11.2f %10.1e %10.1e  %d/%d" %
              (LBL1[m], st.median(t2[m]) / 1e3, np.percentile(t2[m], 99.9) / 1e3,
               st.mean(i2[m]), st.median(r2[m]), e, c2.get(m, 0), n2.get(m, 0)))

    sp = [st.median(t2[m]) / st.median(t2[m + "_FIXED"]) for m in DBL]
    fxs = [st.median(t2[m + "_FIXED"]) / 1e3 for m in DBL]
    fls = [st.median(t2[m]) / 1e3 for m in DBL]
    print("\n-- derivados da Tabela 1 --")
    print("  fixed doubling            : %.2f a %.2f ms" % (min(fxs), max(fxs)))
    print("  float doubling            : %.2f a %.2f ms" % (min(fls), max(fls)))
    print("  speedup float/fixed       : %.2f a %.2f x" % (min(sp), max(sp)))
    print("  spread entre variantes fx : %.2f x" % (max(fxs) / min(fxs)))
    if all(eK.get(m + "_FIXED") for m in DBL):
        print("  e_K fixed doubling        : %.1e a %.1e" %
              (min(st.median(eK[m + "_FIXED"]) for m in DBL),
               max(st.median(eK[m + "_FIXED"]) for m in DBL)))
        print("  e_K float doubling        : %.1e a %.1e" %
              (min(st.median(eK[m]) for m in DBL), max(st.median(eK[m]) for m in DBL)))
        print("  ASDA-fx x mais exato      : %.2f x" %
              (st.median(eK["SDA_FIXED"]) / st.median(eK["ASDA_FIXED"])))
    print("  ASDA-fx custo extra       : %.2f ms" %
          ((st.median(t2["ASDA_FIXED"]) - st.median(t2["SDA_FIXED"])) / 1e3))
    print("  VI ganho da aritmetica    : %.2f x" %
          (st.median(t2["ITERATIVE"]) / st.median(t2["ITERATIVE_FIXED"])))

    if have_s3:
        print("\n" + "=" * 94)
        print("TABELA 2  (S2 vs S3)")
        print("=" * 94)
        print("%-18s %9s %7s %9s %7s %13s %15s" %
              ("Method", "S2-float", "S2-fx", "S3-float", "S3-fx",
               "S2-fx/S3-fx", "S3-fx/S3-float"))
        for m in DBL + ["ITERATIVE"]:
            fx = m + "_FIXED"
            if m not in t3:
                continue
            print("%-18s %9.2f %7.2f %9.2f %7.2f %13.2f %15.2f" %
                  (LBL[m], st.median(t2[m]) / 1e3, st.median(t2[fx]) / 1e3,
                   st.median(t3[m]) / 1e3, st.median(t3[fx]) / 1e3,
                   st.median(t2[fx]) / st.median(t3[fx]),
                   st.median(t3[fx]) / st.median(t3[m])))
        pl = [st.median(t2[m + "_FIXED"]) / st.median(t3[m + "_FIXED"]) for m in DBL]
        ar = [st.median(t3[m + "_FIXED"]) / st.median(t3[m]) for m in DBL]
        flr = [st.median(t2[m]) / st.median(t3[m]) for m in DBL]
        print("\n-- derivados da Tabela 2 --")
        print("  S3-float vs S2-float : %.1f a %.1f x" % (min(flr), max(flr)))
        print("  S2-fx/S3-fx          : %.2f a %.2f x" % (min(pl), max(pl)))
        print("  S3-fx/S3-float       : %.2f a %.2f x" % (min(ar), max(ar)))
        print("  VI: S2-fx/S3-fx=%.2f  S3-fx/S3-float=%.2f" %
              (st.median(t2["ITERATIVE_FIXED"]) / st.median(t3["ITERATIVE_FIXED"]),
               st.median(t3["ITERATIVE_FIXED"]) / st.median(t3["ITERATIVE"])))
    else:
        print("\n[S3 ainda nao capturado/incompleto - Tabela 2 pendente]")

    if args.tabelas:
        print("\n" + "=" * 94)
        print("LaTeX - Tabela 1")
        print("=" * 94)
        for m in ORDER1:
            if m not in t2:
                continue
            print("%-20s & %5.2f  & %5.2f  & %.2f & %.1e & %.1e & %d/%d \\\\" %
                  (LBL1[m], st.median(t2[m]) / 1e3, np.percentile(t2[m], 99.9) / 1e3,
                   st.mean(i2[m]), st.median(r2[m]), st.median(eK[m]),
                   c2.get(m, 0), n2.get(m, 0)))
        if have_s3:
            print("\nLaTeX - Tabela 2")
            for m in DBL + ["ITERATIVE"]:
                fx = m + "_FIXED"
                if m not in t3:
                    continue
                print("%-16s & %5.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\" %
                      (LBL[m], st.median(t2[m]) / 1e3, st.median(t2[fx]) / 1e3,
                       st.median(t3[m]) / 1e3, st.median(t3[fx]) / 1e3,
                       st.median(t2[fx]) / st.median(t3[fx]),
                       st.median(t3[fx]) / st.median(t3[m])))


if __name__ == "__main__":
    main()
