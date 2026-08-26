"""
Q4 do plano de revisão dos dados: o comportamento dos solvers depende de Q/R?
Consome o CSV serial de test/sweep_qr.cpp e produz o mapa de segurança
(escala de R x escala do bloco de taxas de Q) por método _FIXED — taxa de
falha e margem até o teto ±8192 do Q13.18.

Uso:
    python analisa_sweep_qr.py outputs/serial_sweep_qr_v4.txt

v4 (2026-08-18): re-varredura pos-fix do bug de lastOutcome em SDA_Fixed
(AutoLQR.cpp:237) e sob tau=1e-3 (era 1e-6, abaixo do piso de quantizacao do
Q13.18 -- ver docs/auditoria_solvers_riccati.md Secao 15). v3 e anteriores
tem o SDA_FIXED contaminado em r_scale>=1e3 (telemetria congelada da chamada
anterior) e nao usam v4.
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS_FIXED = ["SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"]
METHODS_FLOAT = ["SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA"]
CEILING = 8192.0  # teto real do Q13.18 (2^13)


def load_summaries(path):
    """
    Suporta os dois formatos de SUMMARY do test/sweep_qr.cpp:
      - antigo (7 campos): _,r_scale,q_rate_scale,metodo,failures,count,max_abs
      - novo   (9 campos): _,r_scale,q_rate_scale,metodo,n_converged,n_budget,
                            n_breakdown,count,max_abs
    O novo separa censura por orcamento (n_budget, NAO e falha numerica) de
    falha real (n_breakdown: overflow/singular) — ver AutoLQR::SolveOutcome e
    docs/auditoria_solvers_riccati.md, Secao 13. "failures" abaixo, quando
    lido do formato novo, e SEMPRE n_breakdown (a definicao antiga, "failures",
    e mantida como alias de breakdown para nao quebrar o resto do script).
    """
    rows = []
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("SUMMARY,"):
                continue
            parts = line.split(",")
            if len(parts) == 9:
                _, r_scale, q_rate_scale, metodo, n_conv, n_bud, n_brk, count, max_abs = parts
                rows.append({
                    "r_scale": float(r_scale), "q_rate_scale": float(q_rate_scale),
                    "metodo": metodo,
                    "n_converged": int(n_conv), "n_budget": int(n_bud),
                    "n_breakdown": int(n_brk), "failures": int(n_brk),
                    "count": int(count), "max_abs_seen_pico": float(max_abs),
                })
            elif len(parts) == 7:
                _, r_scale, q_rate_scale, metodo, failures, count, max_abs = parts
                rows.append({
                    "r_scale": float(r_scale), "q_rate_scale": float(q_rate_scale),
                    "metodo": metodo, "failures": int(failures),
                    "n_breakdown": int(failures), "n_budget": 0,
                    "n_converged": int(count) - int(failures),
                    "count": int(count), "max_abs_seen_pico": float(max_abs),
                })
    return pd.DataFrame(rows)


def pivot(df, metodo, value_col):
    sub = df[df["metodo"] == metodo]
    return sub.pivot(index="q_rate_scale", columns="r_scale", values=value_col).sort_index().sort_index(axis=1)


def plot_heatmaps(df, outdir):
    # Só a taxa de falha -- max_abs_seen fica congelado no ultimo sucesso quando
    # a falha e' 100% (getLastFixedPointMaxAbsSeen so atualiza em chamadas OK),
    # entao um heatmap de "margem" contaminaria justamente as celulas mais
    # extremas com um valor obsoleto. A taxa de falha em si e' o dado confiavel.
    r_scales = sorted(df["r_scale"].unique())
    qr_scales = sorted(df["q_rate_scale"].unique())

    fig, axes = plt.subplots(1, len(METHODS_FIXED), figsize=(4.2 * len(METHODS_FIXED), 4.2))
    for j, metodo in enumerate(METHODS_FIXED):
        sub = df[df["metodo"] == metodo]
        fail_rate = pivot(sub.assign(fail_rate=sub["failures"] / sub["count"]), metodo, "fail_rate")
        ax = axes[j]
        im = ax.imshow(fail_rate.values, aspect="auto", cmap="Reds", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(r_scales)))
        ax.set_xticklabels(["%.0e" % r for r in r_scales], rotation=90, fontsize=6)
        ax.set_yticks(range(len(qr_scales)))
        ax.set_yticklabels(["%.0e" % q for q in qr_scales], fontsize=7)
        ax.set_title(metodo, fontsize=9)
        ax.set_xlabel("escala R")
        if j == 0:
            ax.set_ylabel("escala Q (taxas)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Taxa de falha Q13.18 por metodo (escala R x escala Q das taxas) -- "
                  "note a faixa moderada (R~0,1-100) onde os metodos se diferenciam", fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, "fig_sweep_qr_seguranca.png")
    fig.savefig(out, dpi=120)
    print("figura salva: %s" % out)


MODERATE_R = (0.1, 100.0)  # faixa fisicamente plausivel de sintonia (fora dela, TODOS os
                            # metodos falham -- ver nota abaixo); e' aqui que os metodos
                            # realmente se diferenciam.


def print_report(df):
    print("=== Q4: mapa de seguranca Q/R por metodo _FIXED ===\n")

    print("--- taxa de falha por r_scale, agregando q_rate_scale (0-100%) ---")
    g = df[df["metodo"].isin(METHODS_FIXED)].groupby(["r_scale", "metodo"])[["failures", "count"]].sum()
    g["fail_pct"] = 100.0 * g["failures"] / g["count"]
    piv = g["fail_pct"].unstack("metodo").sort_index()
    print(piv[METHODS_FIXED].round(1).to_string())

    print("\nNOTA: fora de r_scale em [1e-2, 1e2] a falha vai a 60-100% nos CINCO metodos")
    print("_FIXED de forma quase igual (float fica em 0%% em toda a grade -- ver secao")
    print("de comparacao com float mais abaixo) -- nao e um limite que alguma variante")
    print("evita melhor que outra, e um teto rigido do formato Q13.18: overflow de")
    print("ENTRADA (Rd, na conversao f2q) no lado de r_scale grande, overflow provavel")
    print("(nao confirmado) de R^-1 dentro do laco no lado de r_scale pequeno -- ver as")
    print("duas notas mais abaixo para o mecanismo de cada lado.")
    print("A comparacao entre metodos so faz sentido dentro da faixa moderada; e mesmo")
    print("dentro dela, r_scale=0,1 e 1,0 ainda mostram falha nao-desprezivel e variavel")
    print("por metodo (ver tabela acima) -- a faixa realmente limpa (0%% em todos) so")
    print("comeca em r_scale=10.\n")

    print("--- zona moderada (r_scale em [%.2g, %.2g]) -- onde os metodos se diferenciam ---" % MODERATE_R)
    mask = (df["r_scale"] >= MODERATE_R[0]) & (df["r_scale"] <= MODERATE_R[1])
    for metodo in METHODS_FIXED:
        sub = df[mask & (df["metodo"] == metodo)]
        total_fail = sub["failures"].sum()
        total_count = sub["count"].sum()
        print("%-20s falhas=%5d/%6d (%.2f%%)" % (
            metodo, total_fail, total_count, 100.0 * total_fail / max(total_count, 1)))

    if "n_budget" in df.columns and df["n_budget"].sum() > 0:
        print("\n--- breakdown vs budget em r_scale>=1000 ---")
        print("ATE 2026-08-17 este dado estava contaminado: um bug em AutoLQR.cpp:237 fazia")
        print("computeGainMatrixSDA_Fixed() retornar sem marcar lastOutcome=Breakdown no unico")
        print("caminho de overflow de ENTRADA do metodo, entao o SDA_FIXED devolvia telemetria")
        print("CONGELADA da chamada anterior nao relacionada (outcome=0, sempre a mesma tripla)")
        print("em vez do breakdown real -- isso gerou a alegacao falsa, ja publicada num rascunho")
        print("do artigo, de que o SDA_FIXED seria 'imune' a R grande. Corrigido; ver")
        print("docs/auditoria_solvers_riccati.md Secao 15.1. Com o fix, os cinco metodos:")
        sub_hi = df[(df["r_scale"] >= 1000) & (df["metodo"].isin(METHODS_FIXED))]
        g_hi = sub_hi.groupby("metodo")[["n_converged", "n_budget", "n_breakdown", "count"]].sum()
        print(g_hi.to_string())
        print("-> 100%% BREAKDOWN, uniforme nos cinco metodos (era so no SDA_FIXED que a telemetria")
        print("   escondia isso). Mecanismo identificado ANALITICAMENTE (nao e so 'overflow/pivo")
        print("   singular' generico): Rd[0][0] = 55,5 x r_scale estoura o teto +-8192 do Q13.18")
        print("   JA na conversao de ENTRADA (f2q), antes de qualquer laco de duplicacao rodar --")
        print("   limiar r_scale = 8192/55,5 ~= 147,6, coerente com a falha aparecer exatamente")
        print("   no salto da grade de 1e2 (0%% falha) para 1e3 (100%% falha).")

    print("\n--- nota sobre max_abs_seen em r_scale>=1000 ---")
    sub_hi = df[(df["r_scale"] >= 1000) & (df["metodo"] == "ASDA_FIXED")]
    if sub_hi["max_abs_seen_pico"].nunique() == 1:
        print("max_abs_seen fica CONGELADO em %.3f para todas as %d combinacoes com r_scale>=1000"
              % (sub_hi["max_abs_seen_pico"].iloc[0], len(sub_hi)))
        print("(getLastFixedPointMaxAbsSeen so atualiza em chamadas bem-sucedidas -- com 100%%")
        print("de falha, o campo fica parado no ultimo sucesso antes da fronteira). Isso e")
        print("CONSISTENTE com o mecanismo identificado acima: o overflow acontece na conversao")
        print("de ENTRADA (f2q de Rd), antes do laco de duplicacao comecar -- max_abs_seen so")
        print("instrumenta o INTERIOR do laco, entao nunca chega a ver o estouro.")

    print("\n--- r_scale MUITO PEQUENO (<=1e-2): segundo modo de falha, simetrico, NAO documentado")
    print("    antes da revisao de 2026-08-18 (ver docs/auditoria_solvers_riccati.md Secao 15.1) ---")
    lo = df[(df["r_scale"] <= 1e-2) & (df["metodo"].isin(METHODS_FIXED))]
    g_lo = lo.groupby("metodo")[["n_converged", "n_budget", "n_breakdown", "count"]].sum()
    print(g_lo.to_string())
    lo_float = df[(df["r_scale"] <= 1e-2) & (df["metodo"].isin(METHODS_FLOAT))]
    tot_brk_float_lo = lo_float["n_breakdown"].sum() if "n_breakdown" in lo_float.columns else lo_float["failures"].sum()
    print("float na mesma faixa: %d breakdowns / %d execucoes" % (tot_brk_float_lo, lo_float["count"].sum()))
    print("-> ~60-61%% de breakdown nos cinco metodos _FIXED, 0%% no float -- exclusivo do ponto")
    print("   fixo, quase igual entre os cinco algoritmos (nao e peculiaridade de um so). Cai")
    print("   para quase zero em r_scale=1e-1 (transicao nitida entre 1e-2 e 1e-1 -- ver tabela")
    print("   acima). Mecanismo PROVAVEL, NAO CONFIRMADO nesta sessao: ao contrario do overflow")
    print("   de entrada em r_scale grande, este parece ser overflow INTERNO ao laco -- R pequeno")
    print("   faz R^-1 (usado em G=B*R^-1*B' dentro da recursao simpletica) crescer, e essa")
    print("   magnitude interna, nao Rd em si, estoura o range +-8192. Nao instrumentado ponto a")
    print("   ponto para confirmar onde exatamente no laco isso ocorre -- trabalho futuro.")

    print("\n--- comparacao com os metodos float (deveriam ser robustos em toda a grade) ---")
    for metodo in METHODS_FLOAT:
        sub = df[df["metodo"] == metodo]
        total_fail = sub["failures"].sum()
        total_count = sub["count"].sum()
        print("%-20s falhas totais=%6d/%6d (%.2f%%)" % (
            metodo, total_fail, total_count, 100.0 * total_fail / max(total_count, 1)))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "outputs", "serial_sweep_qr_v4.txt")
    outdir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    df = load_summaries(path)
    print("linhas SUMMARY carregadas: %d\n" % len(df))
    print_report(df)
    plot_heatmaps(df, outdir)
