"""
Analisa a varredura de γ do SDA-SS (Exp. 3, test/gamma_sweep.cpp).

Objetivo: γ estava fixo em 0,5 (ponto médio de (0,1)) sem justificativa
própria. Este script mede, a partir do dado bruto capturado no hardware, se
algum γ ∈ {0,1; 0,3; 0,5; 0,7; 0,9} domina os demais (mais rápido E mais
preciso, nas duas aritméticas) — em vez de assumir o ponto médio ou inventar
uma conclusão. Ver docs/auditoria_solvers_riccati.md, Seção 15.3.

Critério de dominância usado: γ_a domina γ_b se γ_a é mais rápido (menos
tempo médio) E tem resíduo igual ou melhor que γ_b, na MESMA aritmética.
Se um γ é mais rápido mas com resíduo pior, não há dominância — é troca de
velocidade por acurácia, reportada como tal, não escondida.

Não inventa nada: todo número vem das linhas SUMMARY,3,... da captura serial
(outputs/serial_gamma_sweep.txt por padrão).

Uso:
    python python/analisa_gamma.py [--device outputs/serial_gamma_sweep.txt]
"""
import argparse

DEFAULT_DEVICE = "outputs/serial_gamma_sweep.txt"
COLS = ["exp", "gamma", "metodo", "mean_us", "std_us", "mean_iters", "mean_resid",
        "n_converged", "n_budget", "n_breakdown", "n_total"]


def parse_summary_lines(path):
    rows = []
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
            if not line.startswith("SUMMARY,3,"):
                continue
            parts = line.split(",")
            if len(parts) < 12:
                continue
            _, exp, gamma, metodo, mean_us, std_us, mean_iters, mean_resid, n_conv, n_bud, n_brk, n_tot = parts[:12]
            rows.append({
                "gamma": float(gamma), "metodo": metodo,
                "mean_us": float(mean_us), "std_us": float(std_us),
                "mean_iters": float(mean_iters), "mean_resid": float(mean_resid),
                "n_converged": int(n_conv), "n_budget": int(n_bud),
                "n_breakdown": int(n_brk), "n_total": int(n_tot),
            })
    return rows


def print_table(rows, metodo):
    sub = sorted([r for r in rows if r["metodo"] == metodo], key=lambda r: r["gamma"])
    print("\n--- %s ---" % metodo)
    print("%-6s %8s %8s %10s %12s %10s" % ("gamma", "iters", "us", "resid", "convergiu", "n_total"))
    for r in sub:
        conv_pct = 100.0 * r["n_converged"] / max(r["n_total"], 1)
        print("%-6.1f %8.3f %8.1f %12.4e %9.1f%% %10d" % (
            r["gamma"], r["mean_iters"], r["mean_us"], r["mean_resid"], conv_pct, r["n_total"]))
    return sub


def analisa_dominancia(sub, nome_aritmetica):
    """Para cada par (γ_a, γ_b) com a mais rapido, reporta se domina (resid igual/melhor)
    ou se e' troca de velocidade por acuracia (resid pior)."""
    print("\n  dominancia (%s): comparando cada gamma contra o default antigo (0.5)" % nome_aritmetica)
    baseline = next((r for r in sub if r["gamma"] == 0.5), None)
    if baseline is None:
        print("  (sem gamma=0.5 nos dados -- pulando comparacao com baseline)")
        return
    for r in sub:
        if r["gamma"] == 0.5:
            continue
        mais_rapido = r["mean_us"] < baseline["mean_us"]
        resid_melhor_ou_igual = r["mean_resid"] <= baseline["mean_resid"]
        speedup = (baseline["mean_us"] - r["mean_us"]) / baseline["mean_us"] * 100.0
        resid_delta = (r["mean_resid"] - baseline["mean_resid"]) / baseline["mean_resid"] * 100.0
        if mais_rapido and resid_melhor_ou_igual:
            veredito = "DOMINA o default (0.5): mais rapido E resid igual/melhor"
        elif mais_rapido and not resid_melhor_ou_igual:
            veredito = "mais rapido, mas troca velocidade por acuracia (resid pior)"
        elif not mais_rapido and resid_melhor_ou_igual:
            veredito = "mais lento, mas resid melhor -- dominado pelo default"
        else:
            veredito = "pior nos dois eixos -- dominado pelo default"
        print("  gamma=%.1f: %+.1f%% tempo, %+.1f%% resid -> %s" % (
            r["gamma"], -speedup, resid_delta, veredito))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    args = ap.parse_args()

    rows = parse_summary_lines(args.device)
    if not rows:
        print("nenhuma linha SUMMARY,3,... encontrada em %s -- nada a reportar" % args.device)
        return
    print("linhas SUMMARY do Exp. 3 carregadas: %d" % len(rows))

    for metodo, nome in [("SDA_SS", "float"), ("SDA_SS_FIXED", "fixed")]:
        sub = print_table(rows, metodo)
        if sub:
            analisa_dominancia(sub, nome)

    print("\n=== Resumo ===")
    print("Grade grosseira de 5 pontos (nao e a busca de Fibonacci que Chu, Fan & Lin (2005)")
    print("propoem para o gamma otimo teorico) -- suficiente para descartar o ponto medio")
    print("gamma=0.5 (sem justificativa propria) e substitui-lo por um valor medido.")
    print("Busca fina fica registrada como trabalho futuro (docs/auditoria_solvers_riccati.md,")
    print("Secao 15.3).")


if __name__ == "__main__":
    main()
