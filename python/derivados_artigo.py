# -*- coding: utf-8 -*-
"""Números que o artigo cita e que se derivam dos pontos de operação.

Reúne o que eram dois scripts de uma métrica cada (`residuo_referencia.py` e
`fidelidade_discretizacao.py`, fundidos aqui em 2026-09-01): ambos percorrem os
mesmos 60000 pontos que o dispositivo reportou e produzem um número que aparece
no texto e não saía de nenhuma captura serial.

  referencia    Resíduo DARE da referência em dupla precisão (scipy), nos pontos
                do dispositivo. O artigo afirma "entre 4.8e-15 e 3.0e-13, mediana
                6.3e-14, precisão de máquina para este tamanho". Sem isto, a
                única alegação do artigo que um revisor não conseguia refazer.

  fidelidade    ||A_c*T_s||_2 por trajetória e o primeiro termo desprezado da
                expansão de Taylor de 2a ordem, ||A_c*T_s||^3/6. Sustentam a
                escolha de ancorar a previsibilidade em T3 e tratar T6 como
                limite superior de demanda.

Os pontos vêm das linhas PT da captura (`pontos_dispositivo.py`), não de uma
regeneração em Python — a mesma razão documentada em `numeros_artigo.gain_errors()`:
regenerar introduzia descasamento de modelo em T4 que aparecia como se fosse erro
de solver.

Uso:
    python python/derivados_artigo.py              # ambos
    python python/derivados_artigo.py referencia   # só um
    python python/derivados_artigo.py fidelidade --decimacao 20
"""
import argparse
import collections
import csv
import math
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)
OUT = os.path.join(REPO, "outputs")
sys.path.insert(0, AQUI)

TS = 0.006  # periodo de amostragem do firmware de voo (167 Hz)
COBERTURA = os.path.join(OUT, "cobertura_full_v5_6traj.csv")


# ---------------------------------------------------------------------------
# referencia
# ---------------------------------------------------------------------------
def residuo_dare(Ad, Bd, Qd, Rd, P):
    """||R(P)||_F / ||Q||_F -- identica a verifica_solvers.dare_residual, que e'
    a normalizacao usada em toda a campanha (e na coluna de residuo da Tab. 1)."""
    S = Rd + Bd.T @ P @ Bd
    Res = Ad.T @ P @ Ad - P - Ad.T @ P @ Bd @ np.linalg.solve(S, Bd.T @ P @ Ad) + Qd
    return float(np.linalg.norm(Res) / max(np.linalg.norm(Qd), 1e-30))


def referencia(args):
    import pontos_dispositivo as pd_
    from bench_trajetorias import build_Ad_Bd_Qd_Rd
    from scipy.linalg import solve_discrete_are

    dados = pd_.carregar()
    linhas, falhas = [], 0
    for traj in sorted(dados):
        d = dados[traj]
        for k in range(0, len(d["t"]), args.decimacao):
            Ad, Bd, Qd, Rd = build_Ad_Bd_Qd_Rd(d["phi"][k], d["theta"][k],
                                               d["p"][k], d["q"][k], d["r"][k])
            try:
                P = solve_discrete_are(Ad, Bd, Qd, Rd)
            except Exception:
                falhas += 1
                continue
            linhas.append((traj, k, residuo_dare(Ad, Bd, Qd, Rd, P),
                           float(np.linalg.norm(P, "fro"))))
    if not linhas:
        raise SystemExit("nenhum ponto resolvido")

    r = np.array([x[2] for x in linhas])
    print("referencia float64 (scipy) sobre %d pontos (%d falhas)" % (len(r), falhas))
    print("  residuo relativo : min %.2e | mediana %.2e | max %.2e"
          % (r.min(), float(np.median(r)), r.max()))
    print("  artigo afirma    : entre 4.8e-15 e 3.0e-13, mediana 6.3e-14")

    destino = os.path.join(OUT, "v8", "residuo_referencia.csv")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="") as f:
        f.write("traj,k,dare_residual_rel,normP_F\n")
        for t, k, res, npf in linhas:
            f.write("%s,%d,%.6e,%.6f\n" % (t, k, res, npf))
    print("  CSV: %s" % os.path.relpath(destino, REPO))


# ---------------------------------------------------------------------------
# fidelidade
# ---------------------------------------------------------------------------
def matriz_Ac(phi, theta, p, q, r, IXX, IYY, IZZ):
    """A_c(x) da fatoracao SDC, identica a montada em build_Ad_Bd_Qd_Rd()
    (bench_trajetorias.py:66-88) com a velocidade do rotor omega_r = 0."""
    A = np.zeros((6, 6))
    sR, cR = math.sin(phi), math.cos(phi)
    sP, cP = math.sin(theta), math.cos(theta)
    tP = sP / cP
    A[0, 3] = 1.0
    A[0, 4] = sR * tP
    A[0, 5] = cR * tP
    A[1, 4] = cR
    A[1, 5] = -sR
    A[2, 4] = sR / cP
    A[2, 5] = cR / cP
    A[3, 4] = (IYY - IZZ) / IXX * r
    A[4, 5] = (IZZ - IXX) / IYY * p
    A[5, 4] = (IXX - IYY) / IZZ * p
    return A


def fidelidade(args):
    import bench_trajetorias as B

    pior = collections.defaultdict(float)
    n = collections.Counter()
    with open(COBERTURA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            A = matriz_Ac(float(row["phi"]), float(row["theta"]),
                          float(row["p"]), float(row["q"]), float(row["r"]),
                          B.IXX, B.IYY, B.IZZ)
            nrm = float(np.linalg.norm(A * TS, 2))
            n[row["traj"]] += 1
            if nrm > pior[row["traj"]]:
                pior[row["traj"]] = nrm

    print("%-16s %7s %14s %16s" % ("trajetoria", "n", "||Ac*Ts||_2", "||Ac*Ts||^3/6"))
    print("-" * 58)
    for t in sorted(pior):
        print("%-16s %7d %14.4f %16.2e" % (t, n[t], pior[t], pior[t] ** 3 / 6))
    print("-" * 58)

    destino = os.path.join(OUT, "v8", "fidelidade_discretizacao.csv")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj", "n_pontos", "norm_Ac_Ts_2", "termo_desprezado_rel"])
        for t in sorted(pior):
            w.writerow([t, n[t], "%.6f" % pior[t], "%.6e" % (pior[t] ** 3 / 6)])
    print("CSV: %s" % os.path.relpath(destino, REPO))


METRICAS = {"referencia": referencia, "fidelidade": fidelidade}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("metrica", nargs="?", choices=sorted(METRICAS),
                    help="omitir roda todas")
    ap.add_argument("--decimacao", type=int, default=1,
                    help="usa 1 de cada N pontos na metrica 'referencia' (1 = todos)")
    args = ap.parse_args()

    alvos = [args.metrica] if args.metrica else sorted(METRICAS)
    for nome in alvos:
        print("=== %s ===" % nome)
        METRICAS[nome](args)
        print()


if __name__ == "__main__":
    main()
