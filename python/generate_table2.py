"""
Gera a Tabela 2 (mediana de tempo de solve, S2-float/S2-fx/S3-float/S3-fx) INCLUINDO
o Value iteration, que o v7 do artigo deixava de fora da tabela apesar de a captura do
ESP32-S3 já trazê-lo (N_METHODS=14 na captura, ver outputs/s3/serial_capture_bateria_s3.txt).
Nenhum firmware precisa rodar de novo para isto — os dados já existem nas duas capturas.

Calcula também os dois ratios pedidos:
  S2-fx / S3-fx      — razão de PLATAFORMA mantendo a aritmética fixa (ponto fixo).
  S3-fx / S3-float   — razão de ARITMÉTICA mantendo a plataforma fixa (S3).

Fontes (mesmas de python/figuras_artigo_final.py):
  outputs/serial_capture_bateria_v5_6traj.txt   ESP32-S2 (sem FPU)
  outputs/s3/serial_capture_bateria_s3.txt      ESP32-S3 (com FPU)

Uso:
    python python/generate_table2.py
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")

S2_FILE = os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt")
S3_FILE = os.path.join(OUT, "s3", "serial_capture_bateria_s3.txt")

# (metodo_float, metodo_fixo, rotulo) — mesma ordem/rotulos da Tab. 2 do v7,
# com Value iteration acrescentado ao final.
METHODS = [
    ("SDA", "SDA_FIXED", "SDA"),
    ("SDA_SS", "SDA_SS_FIXED", "SDA-SS"),
    ("ADDA", "ADDA_FIXED", "ADDA"),
    ("SDA_SCALED", "SDA_SCALED_FIXED", "SDA-Scaled"),
    ("ASDA", "ASDA_FIXED", "ASDA"),
    ("ITERATIVE", "ITERATIVE_FIXED", "Value iteration"),
]


def load_times(path):
    """RUN,traj,k,metodo,time_us,iters,residuo,outcome[,rel_step,bit_exact] -> {metodo: [time_us]}."""
    t = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 8:
                continue
            try:
                metodo, us = p[3], int(p[4])
            except ValueError:
                continue
            t[metodo].append(us)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default=os.path.join(OUT, "v8", "tabela2_v8.csv"))
    ap.add_argument("--out-tex", default=os.path.join(OUT, "v8", "tabela2_v8.tex"))
    args = ap.parse_args()

    if not os.path.isfile(S2_FILE):
        raise SystemExit("Captura S2 nao encontrada: %s (rodar a campanha primeiro)" % S2_FILE)
    if not os.path.isfile(S3_FILE):
        raise SystemExit("Captura S3 nao encontrada: %s (env 'benchmark_s3', ver experiments/README.md)" % S3_FILE)

    t2 = load_times(S2_FILE)
    t3 = load_times(S3_FILE)

    rows = []
    for m_float, m_fixed, label in METHODS:
        missing = [m for m in (m_float, m_fixed) if m not in t2 or m not in t3]
        if missing:
            print("aviso: %s ausente em uma das capturas (%s) — pulando %s" %
                  (missing, "S2" if any(m not in t2 for m in missing) else "S3", label))
            continue
        s2_float = st.median(t2[m_float]) / 1000.0
        s2_fx = st.median(t2[m_fixed]) / 1000.0
        s3_float = st.median(t3[m_float]) / 1000.0
        s3_fx = st.median(t3[m_fixed]) / 1000.0
        ratio_platform_fx = s2_fx / s3_fx        # S2-fx / S3-fx
        ratio_arith_s3 = s3_fx / s3_float         # S3-fx / S3-float
        rows.append({
            "metodo": label,
            "s2_float_ms": s2_float, "s2_fx_ms": s2_fx,
            "s3_float_ms": s3_float, "s3_fx_ms": s3_fx,
            "ratio_s2fx_s3fx": ratio_platform_fx,
            "ratio_s3fx_s3float": ratio_arith_s3,
            "n_s2_float": len(t2[m_float]), "n_s2_fx": len(t2[m_fixed]),
            "n_s3_float": len(t3[m_float]), "n_s3_fx": len(t3[m_fixed]),
        })

    print("%-16s %9s %9s %9s %9s %14s %16s" %
          ("Method", "S2-float", "S2-fx", "S3-float", "S3-fx", "S2-fx/S3-fx", "S3-fx/S3-float"))
    for r in rows:
        print("%-16s %9.2f %9.2f %9.2f %9.2f %14.2f %16.2f" % (
            r["metodo"], r["s2_float_ms"], r["s2_fx_ms"], r["s3_float_ms"], r["s3_fx_ms"],
            r["ratio_s2fx_s3fx"], r["ratio_s3fx_s3float"]))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nCSV escrito em %s" % args.out_csv)

    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("%% gerado por python/generate_table2.py a partir de\n")
        f.write("%%   %s\n%%   %s\n" % (
            os.path.relpath(S2_FILE, REPO), os.path.relpath(S3_FILE, REPO)))
        f.write("%% nao editar a mao\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("Method & S2-float & S2-fx & S3-float & S3-fx & S2-fx/S3-fx & S3-fx/S3-float \\\\\n\\hline\n")
        for r in rows:
            f.write("%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\\n" % (
                r["metodo"], r["s2_float_ms"], r["s2_fx_ms"], r["s3_float_ms"], r["s3_fx_ms"],
                r["ratio_s2fx_s3fx"], r["ratio_s3fx_s3float"]))
        f.write("\\hline\n\\end{tabular}\n")
    print("LaTeX escrito em %s" % args.out_tex)

    vi = next((r for r in rows if r["metodo"] == "Value iteration"), None)
    doubling = [r for r in rows if r["metodo"] != "Value iteration"]
    if vi and doubling:
        lo_p = min(r["ratio_s2fx_s3fx"] for r in doubling)
        hi_p = max(r["ratio_s2fx_s3fx"] for r in doubling)
        lo_a = min(r["ratio_s3fx_s3float"] for r in doubling)
        hi_a = max(r["ratio_s3fx_s3float"] for r in doubling)
        print("\nFamilia doubling: S2-fx/S3-fx em [%.2f, %.2f], S3-fx/S3-float em [%.2f, %.2f]" %
              (lo_p, hi_p, lo_a, hi_a))
        print("Value iteration:  S2-fx/S3-fx = %.2f, S3-fx/S3-float = %.2f (fora da faixa "
              "doubling nas duas — o teste de convergencia em float emulado domina a iteracao "
              "mais barata, ver DISCUSSION do artigo)" %
              (vi["ratio_s2fx_s3fx"], vi["ratio_s3fx_s3float"]))


if __name__ == "__main__":
    main()
