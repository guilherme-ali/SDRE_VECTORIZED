"""Verificacao final: cada numero afirmado no diname2027_v5.tex contra os dados brutos.

Uso: python verifica_numeros_artigo.py
     python verifica_numeros_artigo.py --v8   # tambem confere os numeros novos da v8
     (Tabela 2 com Value iteration, fracao de passos bit-exatos, piso de quantizacao)
"""
import argparse
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
TEX = r"G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\artigo_diname\diname2027_v5.tex"

_ap = argparse.ArgumentParser()
_ap.add_argument("--v8", action="store_true", help="tambem confere os numeros novos da v8")
_args, _ = _ap.parse_known_args()

ok = []
bad = []


def check(label, claimed, measured, tol=0.02):
    """tol relativa; measured==None => nao verificavel automaticamente."""
    if measured is None:
        return
    rel = abs(claimed - measured) / abs(measured) if measured else abs(claimed - measured)
    (ok if rel <= tol else bad).append(
        "%-52s artigo=%-12s dados=%-12s (dif %.1f%%)" % (label, claimed, round(measured, 6), 100 * rel)
    )


def load_runs(path):
    t, it, res = defaultdict(list), defaultdict(list), defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip().split(",")
            if len(p) < 8:
                continue
            t[p[3]].append(int(p[4]))
            it[p[3]].append(int(p[5]))
            res[p[3]].append(float(p[6]))
    return t, it, res


t2, i2, r2 = load_runs(os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt"))
t3, i3, _ = load_runs(os.path.join(OUT, "s3", "serial_capture_bateria_s3.txt"))

# ---- Tabela 1 (S2) ----
TAB1 = {  # metodo: (t50, t999, iters)
    "SDA": (8.87, 8.92, 9.00), "SDA_SS": (9.37, 9.43, 7.00), "ADDA": (9.58, 9.64, 9.00),
    "SDA_SCALED": (9.71, 9.77, 9.00), "ASDA": (10.19, 10.25, 9.00), "ITERATIVE": (1.03, 19.28, 8.60),
    "SDA_FIXED": (3.67, 3.69, 9.00), "SDA_SCALED_FIXED": (3.80, 3.83, 9.00),
    "SDA_SS_FIXED": (3.91, 3.95, 7.00), "ASDA_FIXED": (4.20, 4.23, 9.00),
    "ADDA_FIXED": (4.99, 5.03, 9.00), "ITERATIVE_FIXED": (0.95, 14.66, 8.62),
}
for m, (a, b, c) in TAB1.items():
    check("Tab1 %s t50" % m, a, st.median(t2[m]) / 1e3)
    check("Tab1 %s t99.9" % m, b, np.percentile(t2[m], 99.9) / 1e3)
    check("Tab1 %s iters" % m, c, st.mean(i2[m]))

# ---- Tabela 2 (S3) ----
TAB2 = {"SDA": (1.06, 2.76), "SDA_SS": (1.18, 3.06), "ADDA": (1.04, 3.96),
        "SDA_SCALED": (1.08, 2.82), "ASDA": (1.10, 3.13)}
for m, (fl, fx) in TAB2.items():
    check("Tab2 %s S3-float" % m, fl, st.median(t3[m]) / 1e3)
    check("Tab2 %s S3-fx" % m, fx, st.median(t3[m + "_FIXED"]) / 1e3)

# ---- razoes ----
DBL = ["SDA", "SDA_SS", "ADDA", "SDA_SCALED", "ASDA"]
sp = [st.median(t2[m]) / st.median(t2[m + "_FIXED"]) for m in DBL]
check("speedup S2 minimo (1.92)", 1.92, min(sp))
check("speedup S2 maximo (2.55)", 2.55, max(sp))
s3r = [st.median(t3[m + "_FIXED"]) / st.median(t3[m]) for m in DBL]
check("S3 float mais rapido, min (2.6)", 2.6, min(s3r))
check("S3 float mais rapido, max (3.8)", 3.8, max(s3r))
plat = [st.median(t2[m + "_FIXED"]) / st.median(t3[m + "_FIXED"]) for m in DBL]
check("S2-fx/S3-fx min (1.26)", 1.26, min(plat))
check("S2-fx/S3-fx max (1.35)", 1.35, max(plat))
fl = [st.median(t2[m]) / st.median(t3[m]) for m in DBL]
check("S3-float/S2-float min (7.9)", 7.9, min(fl))
check("S3-float/S2-float max (9.3)", 9.3, max(fl))
check("VI ganho fx (1.09)", 1.09, st.median(t2["ITERATIVE"]) / st.median(t2["ITERATIVE_FIXED"]))
check("VI vs SDA-fx mediana (3.9)", 3.9, st.median(t2["SDA_FIXED"]) / st.median(t2["ITERATIVE_FIXED"]))

# ---- condicionamento ----
import csv
cond, normP = [], []
with open(os.path.join(OUT, "cobertura_full_v5_6traj.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cond.append(float(row["cond_IGP"]))
        normP.append(float(row["normP_F"]))
check("cond(I+GP) min (5.54)", 5.54, min(cond))
check("cond(I+GP) max (7.07)", 7.07, max(cond))
check("||P||_F min (0.375)", 0.375, min(normP))
check("||P||_F max (0.487)", 0.487, max(normP))
check("piso min 4.7e-5", 4.7e-5, 6 * 2 ** -18 / max(normP))
check("piso max 6.1e-5", 6.1e-5, 6 * 2 ** -18 / min(normP))
check("n de pontos (60000)", 60000, len(cond))

# ---- voo ----
flight_file = os.path.join(OUT, "serial_flightloop_E.txt")
if os.path.exists(flight_file):
    txt = open(flight_file, encoding="utf-8", errors="replace").read()
    blocks = txt.split("STATUS DO SISTEMA")
    last = blocks[-1]
    hist = [int(x) for x in re.search(r"HIST_PROC_50US:([0-9,]+)", last).group(1).split(",") if x]
    tot = sum(hist)
    cum = np.cumsum(hist)
    cdf = 100.0 * cum / tot
    q = lambda p: (np.searchsorted(cdf, p) * 50) / 1000.0
    check("voo: n de ciclos (47802)", 47802, tot)
    check("voo: mediana 4.70 ms", 4.70, q(50), tol=0.03)
    check("voo: p99 5.20 ms", 5.20, q(99), tol=0.03)
    check("voo: p99.9 5.55 ms", 5.55, q(99.9), tol=0.03)
    check("voo: estouros 0.025%", 0.025, 100.0 * sum(hist[120:]) / tot, tol=0.05)
    check("voo: maximo 6.45 ms", 6.45, int(re.search(r"Processamento_Maximo:\s*(\d+)", last).group(1)) / 1e3, tol=0.03)
    mean_loop = float(re.search(r"Tempo_Medio:\s*([\d.]+)", last).group(1))
    check("voo: periodo medio 6.0023 ms", 6.0023, mean_loop / 1e3, tol=0.01)
    prints = [int(m) for m in re.findall(r"Tempo dos Prints:\s*(\d+)", txt)]
    check("voo: n de blocos de print (290)", 290, len(prints))
    check("voo: tempo de prints 65.37 s", 65.37, sum(prints) / 1e6, tol=0.01)
    check("voo: tempo de laco 286.92 s", 286.92, tot * mean_loop / 1e6, tol=0.01)
    check("voo: init 8.70 s", 8.70, 361 - tot * mean_loop / 1e6 - sum(prints) / 1e6, tol=0.05)
else:
    print(f"[INFO] Arquivo de voo {flight_file} ainda nao capturado nesta maquina (pode ser executado com python run_experiments.py --exp voo).")

# ---- v8: Tabela 2 com Value iteration, fracao bit-exata, e nota de divergencia ----
if _args.v8:
    check("Tab2-v8 VI S2-fx/S3-fx (1.69)", 1.69,
          st.median(t2["ITERATIVE_FIXED"]) / st.median(t3["ITERATIVE_FIXED"]))
    check("Tab2-v8 VI S3-fx/S3-float (1.97)", 1.97,
          st.median(t3["ITERATIVE_FIXED"]) / st.median(t3["ITERATIVE"]))

    tolsweep_file = os.path.join(OUT, "serial_tolerance_sweep_frobenius.txt")
    if os.path.exists(tolsweep_file):
        n_be, n_tot = 0, 0
        with open(tolsweep_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("RUN,0a,1e-05,"):
                    continue
                p = line.rstrip("\n").split(",")
                if len(p) < 6 or p[5] != "SDA_FIXED":
                    continue
                n_tot += 1
                if len(p) >= 12 and p[11] == "1":
                    n_be += 1
                elif len(p) >= 11 and float(p[10]) == 0.0:  # fallback: captura pre-bit_exact
                    n_be += 1
        if n_tot:
            check("v8: fracao bit-exata SDA_FIXED @tau=1e-5 (1.0 = 1824/1824)",
                  1.0, n_be / n_tot)
        else:
            print("[INFO] serial_tolerance_sweep_frobenius.txt nao tem linhas RUN,0a,1e-05,...,SDA_FIXED "
                  "com rel_step (recapturar com o firmware instrumentado).")
    else:
        print(f"[INFO] {tolsweep_file} nao encontrado.")

    # Divergencia encontrada entre o texto do v7 e os dados brutos, registrada
    # para a rodada de redacao da v8 (nao corrigida aqui — fora do escopo desta
    # auditoria, que so confere numeros, nao edita o .tex).
    try:
        tol_qr_file = os.path.join(OUT, "serial_tol_qr_sweep_A.txt")
        cell = defaultdict(dict)
        with open(tol_qr_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("SUMMARY,"):
                    continue
                p = line.strip().split(",")
                if len(p) != 11:
                    continue
                _, tau, rs, qr, m, nc, nb, nbk, cnt, mus, mres = p
                if rs == "1e+00" and qr == "1e+00" and m in ("SDA_FIXED", "SDA_SCALED_FIXED"):
                    cell[m][tau] = float(mres)
        rows_div = []
        for m in ("SDA_FIXED", "SDA_SCALED_FIXED"):
            if "1e-02" in cell[m] and "3e-05" in cell[m]:
                delta_pct = 100.0 * (cell[m]["3e-05"] / cell[m]["1e-02"] - 1.0)
                rows_div.append((m, delta_pct))
        div_path = os.path.join(OUT, "v8", "divergencias_v7.md")
        os.makedirs(os.path.dirname(div_path), exist_ok=True)
        with open(div_path, "w", encoding="utf-8") as f:
            f.write("# Divergencias entre diname2027_v7.tex e os dados brutos (achadas na auditoria v8)\n\n")
            f.write("Nao corrigidas neste arquivo — fora do escopo da preparacao de dados/scripts da v8. "
                    "Registradas aqui para a proxima rodada de redacao.\n\n")
            f.write("## Variacao do residuo atingido ao apertar tau (1e-2 -> 3e-5), pesos nominais\n\n")
            f.write("O v7 (secao 'Tolerance, achieved accuracy...') afirma "
                    "\"+0.3% para SDA-fx e +3.2% para SDA-Scaled-fx\". "
                    "Os dados brutos (outputs/serial_tol_qr_sweep_A.txt, celula R_scale=1, "
                    "Q_rate_scale=1) e outputs/serial_tolerance_sweep_frobenius.txt "
                    "(medianas sobre 1824 pontos) concordam entre si e discordam do texto:\n\n")
            f.write("| Metodo | v7 (texto) | tol_qr_sweep (celula nominal) |\n|---|---|---|\n")
            claimed = {"SDA_FIXED": 0.3, "SDA_SCALED_FIXED": 3.2}
            for m, delta_pct in rows_div:
                f.write("| %s | %+.1f%% | %+.2f%% |\n" % (m, claimed.get(m, float("nan")), delta_pct))
            f.write("\nVer python/verifica_numeros_artigo.py --v8 para reproduzir.\n")
        if rows_div:
            print("\n[AVISO] divergencia texto x dados encontrada e registrada em %s" %
                  os.path.relpath(div_path, REPO))
            for m, delta_pct in rows_div:
                print("        %s: v7 afirma %+.1f%%, dado bruto mostra %+.2f%%" %
                      (m, claimed.get(m, float("nan")), delta_pct))
    except FileNotFoundError:
        pass

print("=" * 96)
print("CONFEREM (%d):" % len(ok))
for s in ok:
    print("  OK  " + s)
print()
print("DIVERGEM (%d):" % len(bad))
for s in bad:
    print("  XX  " + s)
print("=" * 96)
