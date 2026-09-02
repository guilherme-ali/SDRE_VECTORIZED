"""Verificacao final: cada numero afirmado no diname2027_v5.tex contra os dados brutos.

Uso: python verifica_numeros_artigo.py
     python verifica_numeros_artigo.py --v8   # tambem confere os numeros novos da v8
     (Tabela 2 com Value iteration, fracao de passos bit-exatos, piso de quantizacao)
"""
import argparse
import glob
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
# Artigo cujas afirmacoes as constantes deste script reproduzem. O .tex nao e' lido:
# os numeros abaixo sao transcritos do texto de proposito, para que uma mudanca no
# dado bruto apareca como divergencia em vez de ser absorvida em silencio.
TEX = r"G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\artigo_diname\diname2027_v8.tex"

_ap = argparse.ArgumentParser()
_ap.add_argument("--v8", action="store_true", help="tambem confere os numeros novos da v8")
_args, _ = _ap.parse_known_args()

ok = []
bad = []


def _carrega_tolerance_sweep():
    """Reaproveita o parser da varredura de tolerancia de figuras_artigo_final,
    para que figura e auditoria leiam a captura pelo mesmo codigo."""
    try:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from figuras_artigo_final import load_tolerance_sweep

        return load_tolerance_sweep()
    except Exception as e:
        print("[INFO] varredura de tolerancia indisponivel: %r" % (e,))
        return None


def check(label, claimed, measured, tol=0.02):
    """tol relativa; measured==None => nao verificavel automaticamente."""
    if measured is None:
        return
    rel = abs(claimed - measured) / abs(measured) if measured else abs(claimed - measured)
    # %g em vez de round(x, 6): um passo de 5.44e-07 virava "1e-06" na coluna
    # de dados, escondendo justamente a ordem de grandeza que se quer conferir.
    fmt = (lambda v: "%.6g" % v) if isinstance(measured, float) else str
    (ok if rel <= tol else bad).append(
        "%-52s artigo=%-12s dados=%-12s (dif %.1f%%)" % (label, claimed, fmt(measured), 100 * rel)
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
# Valores EXATAMENTE como a Tabela 1 do v8 os imprime. Ate 2026-09-01 este
# dicionario trazia os numeros do v5 (SDA 8.87/8.92 contra os 8.92/8.97 do v8) e
# passava assim mesmo, porque a tolerancia de 2% engolia a diferenca de 0.6%: a
# tabela era conferida contra o dado, nunca contra o que o artigo publica.
TAB1 = {  # metodo: (t50, t999, iters)
    "SDA": (8.92, 8.97, 9.00), "SDA_SS": (9.40, 9.46, 7.00), "ADDA": (9.61, 9.67, 9.00),
    "SDA_SCALED": (9.76, 9.82, 9.00), "ASDA": (10.21, 10.27, 9.00), "ITERATIVE": (1.03, 19.50, 8.60),
    "SDA_FIXED": (3.15, 3.17, 9.00), "SDA_SCALED_FIXED": (3.29, 3.31, 9.00),
    "SDA_SS_FIXED": (3.50, 3.53, 7.00), "ASDA_FIXED": (3.66, 3.69, 9.00),
    "ADDA_FIXED": (4.47, 4.50, 9.00), "ITERATIVE_FIXED": (0.96, 14.71, 8.63),
}
TOL_TAB1 = 0.005  # 0.5%: com os valores certos, nao ha mais folga a dar
for m, (a, b, c) in TAB1.items():
    check("Tab1 %s t50" % m, a, st.median(t2[m]) / 1e3,
          tol=TOL_TAB1 if a > 1.0 else 0.01)  # 0.955 -> 0.95 com 2 casas
    check("Tab1 %s t99.9" % m, b, np.percentile(t2[m], 99.9) / 1e3, tol=TOL_TAB1)
    check("Tab1 %s iters" % m, c, st.mean(i2[m]), tol=TOL_TAB1)

# ---- Tabela 2 (S3) ----
TAB2 = {"SDA": (1.06, 2.83), "SDA_SS": (1.19, 3.13), "ADDA": (1.04, 4.01),
        "SDA_SCALED": (1.10, 2.88), "ASDA": (1.12, 3.18)}
for m, (fl, fx) in TAB2.items():
    check("Tab2 %s S3-float" % m, fl, st.median(t3[m]) / 1e3)
    check("Tab2 %s S3-fx" % m, fx, st.median(t3[m + "_FIXED"]) / 1e3)

# ---- razoes ----
DBL = ["SDA", "SDA_SS", "ADDA", "SDA_SCALED", "ASDA"]
sp = [st.median(t2[m]) / st.median(t2[m + "_FIXED"]) for m in DBL]
check("speedup S2 minimo (2.15)", 2.15, min(sp))
check("speedup S2 maximo (2.97)", 2.97, max(sp))
s3r = [st.median(t3[m + "_FIXED"]) / st.median(t3[m]) for m in DBL]
check("S3 float mais rapido, min (2.60)", 2.60, min(s3r))
check("S3 float mais rapido, max (3.87)", 3.87, max(s3r))
plat = [st.median(t2[m + "_FIXED"]) / st.median(t3[m + "_FIXED"]) for m in DBL]
check("S2-fx/S3-fx min (1.11)", 1.11, min(plat))
check("S2-fx/S3-fx max (1.15)", 1.15, max(plat))
fl = [st.median(t2[m]) / st.median(t3[m]) for m in DBL]
check("S3-float/S2-float min (7.9)", 7.9, min(fl))
check("S3-float/S2-float max (9.3)", 9.3, max(fl))
check("VI ganho fx (1.08)", 1.08, st.median(t2["ITERATIVE"]) / st.median(t2["ITERATIVE_FIXED"]))
# derivado, nao citado no artigo (que da os dois tempos, 0.96 e 3.68 ms)
check("VI vs SDA-fx mediana (3.30, derivado)", 3.30, st.median(t2["SDA_FIXED"]) / st.median(t2["ITERATIVE_FIXED"]))

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
# Fonte: as N janelas de outputs/voo/, nao a captura unica. O laco de voo e' o
# unico experimento nao deterministico da campanha -- as mesmas 360 s do mesmo
# binario deram de 0 a 6 ciclos acima do periodo -- entao conferir o artigo
# contra uma janela so' confere contra a janela que calhou de ser gravada.
# Somar os histogramas de 50 us e' exato: os bins sao os mesmos em toda janela.
VOO_ESPERADO = {
    "execucoes": 10,
    "ciclos": 475444,
    "mediana_ms": 4.20,
    "media_ms": 4.25,
    "p99_ms": 4.85,
    "p999_ms": 4.90,
    "estouros_pct": 0.0004,
    "max_min_ms": 5.29,
    "max_max_ms": 6.32,
    "dare_mediana_ms": 3.162,
    "imu_mediana_ms": 0.624,
}

voo_dir = os.path.join(OUT, "voo")
voo_files = sorted(glob.glob(os.path.join(voo_dir, "voo_run*.txt")),
                   key=lambda p: int(re.search(r"voo_run(\d+)", p).group(1)))
if not voo_files:
    single = os.path.join(OUT, "serial_flightloop_E.txt")
    voo_files = [single] if os.path.exists(single) else []

if voo_files:
    hist_total, maximos, dare, imu, medias, ciclos_por_run = None, [], [], [], [], []
    for path in voo_files:
        txt = open(path, encoding="utf-8", errors="replace").read()
        blocks = txt.split("STATUS DO SISTEMA")
        if len(blocks) < 2:
            continue
        mh = re.search(r"HIST_PROC_50US:([0-9,]+)", blocks[-1])
        if not mh:
            continue
        h = [int(x) for x in mh.group(1).split(",") if x]
        if hist_total is None:
            hist_total = list(h)
        else:
            if len(h) > len(hist_total):
                hist_total += [0] * (len(h) - len(hist_total))
            for i, c in enumerate(h):
                hist_total[i] += c
        ciclos_por_run.append(sum(h))
        mm = re.search(r"Processamento_Maximo:\s*(\d+)", blocks[-1])
        if mm:
            maximos.append(int(mm.group(1)) / 1e3)
        md = re.search(r"Processamento_Medio:\s*([\d.]+)", blocks[-1])
        if md:
            medias.append(float(md.group(1)) / 1e3)
        for b in blocks[1:]:
            m1 = re.search(r"LQR .Ganhos.:\s*(\d+)\s*.s", b)
            if m1:
                dare.append(int(m1.group(1)) / 1e3)
            m2 = re.search(r"Leitura MPU:\s*(\d+)\s*.s", b)
            if m2:
                imu.append(int(m2.group(1)) / 1e3)

    tot = sum(hist_total)
    cdf = 100.0 * np.cumsum(hist_total) / tot
    q = lambda p: (np.searchsorted(cdf, p) * 50) / 1000.0
    media_pond = sum(m * c for m, c in zip(medias, ciclos_por_run)) / tot

    check("voo: n de execucoes", VOO_ESPERADO["execucoes"], len(ciclos_por_run), tol=0.001)
    check("voo: ciclos agregados", VOO_ESPERADO["ciclos"], tot, tol=0.001)
    check("voo: mediana (ms)", VOO_ESPERADO["mediana_ms"], q(50), tol=0.01)
    check("voo: media (ms)", VOO_ESPERADO["media_ms"], media_pond, tol=0.01)
    check("voo: p99 (ms)", VOO_ESPERADO["p99_ms"], q(99), tol=0.01)
    check("voo: p99.9 (ms)", VOO_ESPERADO["p999_ms"], q(99.9), tol=0.01)
    check("voo: estouros (%)", VOO_ESPERADO["estouros_pct"],
          100.0 * sum(hist_total[120:]) / tot, tol=0.15)
    check("voo: menor maximo por janela (ms)", VOO_ESPERADO["max_min_ms"], min(maximos), tol=0.01)
    check("voo: maior maximo por janela (ms)", VOO_ESPERADO["max_max_ms"], max(maximos), tol=0.01)
    check("voo: DARE mediana (ms)", VOO_ESPERADO["dare_mediana_ms"], st.median(dare), tol=0.01)
    check("voo: I2C mediana (ms)", VOO_ESPERADO["imu_mediana_ms"], st.median(imu), tol=0.01)
else:
    print("[INFO] Nenhuma captura de voo nesta maquina "
          "(rode: python python/run_experiments.py --only voo --repeat 10).")

# ---- v8: Tabela 2 com Value iteration, fracao bit-exata, e nota de divergencia ----
if _args.v8:
    check("Tab2-v8 VI S2-fx/S3-fx (1.65)", 1.65,
          st.median(t2["ITERATIVE_FIXED"]) / st.median(t3["ITERATIVE_FIXED"]))
    check("Tab2-v8 VI S3-fx/S3-float (1.91)", 1.91,
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

# ---- numeros do corpo do artigo (fora das tabelas) ----
# Ate 2026-09-01 estes eram conferidos a mao, quando eram; ver o docstring de
# python/checagens_prosa.py para o que isso custou nas versoes anteriores.
try:
    import checagens_prosa

    checagens_prosa.todas(
        check,
        os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt"),
        _carrega_tolerance_sweep,
        t2, t3, r2,
    )
except Exception as e:  # pragma: no cover
    print("[AVISO] bloco de checagens de prosa falhou: %r" % (e,))

print("=" * 96)
print("CONFEREM (%d):" % len(ok))
for s in ok:
    print("  OK  " + s)
print()
print("DIVERGEM (%d):" % len(bad))
for s in bad:
    print("  XX  " + s)
print("=" * 96)
