"""
Analisa a varredura de tolerância (Exp. 0a/0b, test/tolerance_sweep.cpp).

Objetivo: escolher a "tolerância casada" para a bateria principal (Exp. 1) e
a varredura Q/R (Exp. 2) — a tolerância mais apertada que TODOS os métodos
(float e _FIXED) da família de duplicação atingem sem breakdown e com
convergência efetiva (não apenas dentro do orçamento por acaso), e reporta
separadamente o que a iteração de valor (Exp. 0b) precisa para as mesmas
tolerâncias, já que ela usa um orçamento muito maior (2000 x 200).

Não inventa nada: todo número vem das linhas SUMMARY do arquivo de captura
serial (outputs/serial_tolerance_sweep.txt por padrão).

Uso:
    python python/analisa_tolerancia.py [--device outputs/serial_tolerance_sweep_frobenius.txt]
"""
import argparse
import csv
import io
from collections import defaultdict

DEFAULT_DEVICE = "outputs/serial_tolerance_sweep_frobenius.txt"


def parse_summary_lines(path):
    """Retorna duas listas de dicts: (linhas_0a, linhas_0b)."""
    rows_0a, rows_0b = [], []
    cols_0a = ["exp", "tol", "metodo", "mean_us", "std_us", "mean_iters",
               "n_converged", "n_budget", "n_breakdown", "n_total"]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("SUMMARY,"):
                continue
            parts = line.split(",")
            if len(parts) < 11:
                continue
            _, exp, tol, metodo, mean_us, std_us, mean_iters, n_conv, n_bud, n_brk, n_tot = parts[:11]
            row = {
                "tol": float(tol),
                "metodo": metodo,
                "mean_us": float(mean_us),
                "std_us": float(std_us),
                "mean_iters": float(mean_iters),
                "n_converged": int(n_conv),
                "n_budget": int(n_bud),
                "n_breakdown": int(n_brk),
                "n_total": int(n_tot),
            }
            if exp == "0a":
                rows_0a.append(row)
            elif exp == "0b":
                rows_0b.append(row)
    return rows_0a, rows_0b


def choose_matched_tolerance(rows_0a, min_converged_frac=0.95):
    """
    Entre as tolerâncias testadas em 0a, escolhe a mais apertada (menor valor)
    em que TODOS os métodos (float e _FIXED) têm zero breakdown e pelo menos
    min_converged_frac de convergência dentro do orçamento de 200 iterações.
    """
    by_tol = defaultdict(list)
    for r in rows_0a:
        by_tol[r["tol"]].append(r)

    candidates = []
    for tol, rows in by_tol.items():
        ok = True
        worst_frac = 1.0
        for r in rows:
            if r["n_breakdown"] > 0:
                ok = False
            frac = r["n_converged"] / r["n_total"] if r["n_total"] > 0 else 0.0
            worst_frac = min(worst_frac, frac)
        if worst_frac < min_converged_frac:
            ok = False
        if ok:
            candidates.append(tol)

    if not candidates:
        return None
    return min(candidates)  # a mais apertada (numericamente menor) que sobrevive


def report(rows_0a, rows_0b):
    print("=" * 78)
    print("EXP. 0a — família de duplicação, orçamento 200 iterações")
    print("=" * 78)
    by_tol = defaultdict(list)
    for r in rows_0a:
        by_tol[r["tol"]].append(r)
    for tol in sorted(by_tol.keys(), reverse=True):
        print(f"\n--- tolerância {tol:.0e} ---")
        print(f"{'metodo':20s} {'mean_us':>10s} {'mean_iters':>11s} "
              f"{'conv':>6s} {'budget':>7s} {'breakdown':>10s} {'total':>6s}")
        for r in sorted(by_tol[tol], key=lambda x: x["metodo"]):
            print(f"{r['metodo']:20s} {r['mean_us']:10.1f} {r['mean_iters']:11.2f} "
                  f"{r['n_converged']:6d} {r['n_budget']:7d} {r['n_breakdown']:10d} {r['n_total']:6d}")

    matched = choose_matched_tolerance(rows_0a)
    print("\n" + "=" * 78)
    if matched is not None:
        print(f"TOLERÂNCIA CASADA ESCOLHIDA (Exp. 0a): {matched:.0e}")
    else:
        print("NENHUMA tolerância testada satisfaz o critério (breakdown=0 em todos "
              "os métodos, >=95% convergência) — reportar as curvas, não um ponto único.")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("EXP. 0b — iteração de valor, orçamento 2000 iterações")
    print("=" * 78)
    by_tol_b = defaultdict(list)
    for r in rows_0b:
        by_tol_b[r["tol"]].append(r)
    for tol in sorted(by_tol_b.keys(), reverse=True):
        print(f"\n--- tolerância {tol:.0e} ---")
        print(f"{'metodo':20s} {'mean_us':>10s} {'mean_iters':>11s} "
              f"{'conv':>6s} {'budget':>7s} {'breakdown':>10s} {'total':>6s}")
        for r in sorted(by_tol_b[tol], key=lambda x: x["metodo"]):
            print(f"{r['metodo']:20s} {r['mean_us']:10.1f} {r['mean_iters']:11.2f} "
                  f"{r['n_converged']:6d} {r['n_budget']:7d} {r['n_breakdown']:10d} {r['n_total']:6d}")

    if matched is not None:
        print(f"\nAplicando a tolerância casada ({matched:.0e}) ao caso ITERATIVE/ITERATIVE_FIXED:")
        for r in by_tol_b.get(matched, []):
            frac = r["n_converged"] / r["n_total"] if r["n_total"] > 0 else 0.0
            print(f"  {r['metodo']:20s} converge em {frac*100:5.1f}% dos pontos "
                  f"(mean_iters={r['mean_iters']:.1f}, mean_us={r['mean_us']:.0f})")

    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    args = ap.parse_args()

    rows_0a, rows_0b = parse_summary_lines(args.device)
    print(f"Lidas {len(rows_0a)} linhas SUMMARY de 0a e {len(rows_0b)} de 0b de {args.device}")
    if not rows_0a and not rows_0b:
        print("Nenhuma linha SUMMARY encontrada — captura incompleta ou arquivo errado.")
        return
    report(rows_0a, rows_0b)


if __name__ == "__main__":
    main()
