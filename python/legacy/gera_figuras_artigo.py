"""
Gera as figuras do artigo DINAME 2027 em qualidade de publicação: rótulos em
inglês, fontes legíveis no tamanho impresso, distinção por marcador/hachura
além da cor (o template pede legibilidade em P&B explicitamente) e saída em
PDF vetorial.

NENHUMA medição nova: consome apenas os artefatos já capturados em outputs/.
Cada figura declara sua fonte de dados no docstring da função que a gera, para
que todo número do artigo seja rastreável.

Uso:
    python gera_figuras_artigo.py [--outdir <dir>]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import trajetorias as trj
from bench_trajetorias import load_device_csv
from gerar_relatorio_bateria import summaries_to_df, gains_erro_k
from analisa_sweep_qr import load_summaries as load_sweep_summaries, CEILING

REPO = os.path.join(os.path.dirname(__file__), "..")
# v4 (2026-08-18): critério de parada casado RE-CALIBRADO (Frobenius relativa,
# tau=1e-3 -- nao mais 1e-6, que estava abaixo do piso de quantizacao do
# Q13.18 -- orcamento 200 para os 12 metodos), gamma do SDA-SS medido em 0.7
# (nao mais 0.5), e o bug do lastOutcome em computeGainMatrixSDA_Fixed()
# corrigido (SDA_FIXED nao reporta mais telemetria congelada em R grande).
# Sob esse criterio, os 12 metodos convergem 100% (46152/46152 cada) --
# ver docs/auditoria_solvers_riccati.md, Secao 15. NAO usar mais _v3 (tau
# errado, abaixo do piso de quantizacao) nem _v2 (gamma errado).
BATERIA = os.path.join(REPO, "outputs", "serial_capture_bateria_v4.txt")
SWEEP = os.path.join(REPO, "outputs", "serial_sweep_qr_v4.txt")
MALHA = os.path.join(REPO, "outputs", "malha_fechada_trajetorias_v4.csv")

TRAJ_ORDER = ["T1_espiral", "T2_figura8", "T3_chirp", "T4_degrau_yaw"]
TRAJ_LABEL = {
    "T1_espiral": "T1: growing-radius spiral",
    "T2_figura8": "T2: figure-eight",
    "T3_chirp": "T3: attitude chirp",
    "T4_degrau_yaw": "T4: steps + yaw spin",
}
# Ordem de apresentacao: float primeiro, depois os pares _FIXED
METHOD_ORDER = ["SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA", "ITERATIVE",
                "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED",
                "ADDA_FIXED", "ITERATIVE_FIXED"]
METHOD_LABEL = {m: m.replace("_FIXED", "-fx").replace("_", "-") for m in METHOD_ORDER}

# Padroes de hachura para distinguir barras em P&B (nao so por cor)
HATCH_FLOAT = ""
HATCH_FIXED = "///"


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
    })


def _is_fixed(m):
    return m.endswith("_FIXED")


def _bar_style(m):
    """Cinza claro + hachura para _FIXED, cinza escuro solido para float --
    legivel em P&B."""
    if _is_fixed(m):
        return dict(color="0.80", edgecolor="black", hatch=HATCH_FIXED, linewidth=0.6)
    return dict(color="0.45", edgecolor="black", hatch=HATCH_FLOAT, linewidth=0.6)


# Janela de tempo mostrada por trajetoria. Plotar os 60 s inteiros de T2/T3/T4
# a 192 Hz (11538 pontos) transforma o sinal num borrao solido no tamanho
# impresso: T2 tem 7.5 ciclos, T3 varre ate 8 Hz e T4 comuta a cada 2 s. As
# janelas abaixo mostram a FORMA de onda de cada uma; a legenda da figura
# declara que a trajetoria continua por 60 s.
FIG1_WINDOW = {
    "T1_espiral": (0.0, 60.0),   # completo: o que importa aqui e o crescimento do raio
    "T2_figura8": (0.0, 16.0),   # 2 periodos de 8 s
    "T3_chirp": (0.0, 12.0),     # ~0.2 -> 1.8 Hz, mostra a varredura comecando
    "T4_degrau_yaw": (0.0, 8.0), # 4 degraus
}


def fig1_trajectories(outdir):
    """Fig. 1 -- as quatro trajetorias. Fonte: python/trajetorias.py (formas
    fechadas e deterministicas; nenhum dado medido)."""
    dados = trj.gerar_todas()
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.2))
    for j, name in enumerate(TRAJ_ORDER):
        d = dados[name]
        t0, t1 = FIG1_WINDOW[name]
        sel = (d["t"] >= t0) & (d["t"] <= t1)
        t = d["t"][sel]

        ax = axes[0, j]
        ax.plot(t, np.rad2deg(d["phi"][sel]), "-", color="0.15", lw=0.6, label=r"$\phi$")
        ax.plot(t, np.rad2deg(d["theta"][sel]), "--", color="0.55", lw=0.6, label=r"$\theta$")
        ax.set_title(TRAJ_LABEL[name], fontsize=8)
        ax.set_xlim(t0, t1)
        if j == 0:
            ax.set_ylabel("attitude (deg)")
        ax.legend(ncol=2, frameon=False, handlelength=1.4, fontsize=7, loc="upper right")

        ax2 = axes[1, j]
        ax2.plot(t, np.rad2deg(d["p"][sel]), "-", color="0.15", lw=0.6, label="$p$")
        ax2.plot(t, np.rad2deg(d["r"][sel]), ":", color="0.55", lw=0.8, label="$r$")
        ax2.set_xlabel("time (s)")
        ax2.set_xlim(t0, t1)
        if j == 0:
            ax2.set_ylabel("body rate (deg/s)")
        ax2.legend(ncol=2, frameon=False, handlelength=1.4, fontsize=7, loc="upper right")
    fig.tight_layout()
    out = os.path.join(outdir, "fig1_trajectories.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("  " + out)


def fig2_timing(outdir, df_sum_all):
    """Fig. 2 -- tempo medio de execucao por metodo, escala LINEAR (nao mais
    log: sob tau=1e-3 todos os 12 metodos ficam entre 2 e 11 ms, uma faixa de
    apenas 5x que uma escala log achatava desnecessariamente e que escondia o
    2.4x que e a tese central do artigo). Fonte: linhas SUMMARY,ALL de
    serial_capture_bateria_v4.txt (46.152 pontos por metodo, ESP32-S2,
    tau=1e-3, gamma=0.7)."""
    sub = df_sum_all.set_index("metodo").reindex(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    x = np.arange(len(METHOD_ORDER))
    for i, m in enumerate(METHOD_ORDER):
        ax.bar(x[i], sub.loc[m, "mean_us"] / 1000.0, yerr=sub.loc[m, "std_us"] / 1000.0,
               capsize=2.5, error_kw=dict(lw=0.6, capthick=0.6), **_bar_style(m))
    # Referencia = o periodo real do laco de voo (5.2 ms), nao um alvo generico
    # de 80 Hz: um solver acima da linha nao consegue rodar todo ciclo.
    ax.axhline(5.2, color="black", ls="--", lw=0.9)
    ax.text(len(METHOD_ORDER) - 0.4, 5.35, "flight loop period (5.2 ms)",
            ha="right", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=45, ha="right")
    ax.set_ylabel("mean solve time (ms)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.45", edgecolor="black", linewidth=0.6),
               plt.Rectangle((0, 0), 1, 1, facecolor="0.80", edgecolor="black",
                             hatch=HATCH_FIXED, linewidth=0.6)]
    ax.legend(handles, ["single-precision float", "fixed-point Q13.18"],
              frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(outdir, "fig2_timing.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("  " + out)


def fig3_accuracy(outdir, df_erro):
    """Fig. 3 -- erro RMS de K contra a referencia scipy (float64, validada
    contra o DAREX), POR TRAJETORIA. Fonte: linhas GAIN da bateria v4
    (tau=1e-3, gamma=0.7, 100% convergencia em todos os 12 metodos -- o vies
    do SUMMARY que antes so contava os poucos casos convergidos do
    ITERATIVE_FIXED sob tau=1e-6 desapareceu, RUN e SUMMARY concordam agora),
    decimadas 1:50, confrontadas em python/gerar_relatorio_bateria.py.

    Por trajetoria, nao agregado: a media sobre as quatro e dominada por T4
    (4.7e-6 contra 1.4e-8 em T1/T2 para os metodos float), o que achataria a
    separacao float-vs-fixed de ~3 ordens de grandeza que se ve em T1/T2.
    Fonte aumentada (8pt, era 6pt) e figura mais alta para legibilidade."""
    trajs = [t for t in TRAJ_ORDER if t in df_erro["traj"].unique()]
    fig, axes = plt.subplots(1, len(trajs), figsize=(7.2, 3.1), sharey=True)
    x = np.arange(len(METHOD_ORDER))
    for j, t in enumerate(trajs):
        agg = df_erro[df_erro["traj"] == t].groupby("metodo")["erro_K"].mean().reindex(METHOD_ORDER)
        ax = axes[j]
        for i, m in enumerate(METHOD_ORDER):
            ax.bar(x[i], agg.loc[m], **_bar_style(m))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=90, fontsize=8)
        ax.set_title(TRAJ_LABEL[t], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        if j == 0:
            ax.set_ylabel(r"mean RMS error of $K$")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.45", edgecolor="black", linewidth=0.6),
               plt.Rectangle((0, 0), 1, 1, facecolor="0.80", edgecolor="black",
                             hatch=HATCH_FIXED, linewidth=0.6)]
    fig.legend(handles, ["single-precision float", "fixed-point Q13.18"],
               frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    out = os.path.join(outdir, "fig3_accuracy.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("  " + out)


def fig4_qr_map(outdir):
    """Fig. 4 -- mapa de seguranca (Q,R): fracao de falhas por combinacao de
    escala, para os cinco metodos _FIXED. Fonte: linhas SUMMARY de
    serial_sweep_qr_v4.txt (195.000 execucoes no ESP32-S2, bug do lastOutcome
    corrigido -- os 5 metodos falham identicamente em R_scale>=1e3, nao mais
    so 4 deles -- ver docs/auditoria_solvers_riccati.md, Secao 15.1). O eixo
    log10(R) cobre -6 a 6, entao o mapa mostra os DOIS extremos de falha sem
    nenhuma mudanca de codigo: R grande (overflow de entrada, Secao 15.1) e o
    modo simetrico em R pequeno, R_scale<=1e-2 (overflow interno hipotetizado,
    Secao 15.1 'Atualizacao 2026-08-18'), ambos como faixas escuras nas bordas
    do eixo x."""
    df = load_sweep_summaries(SWEEP)
    methods = ["SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"]
    r_scales = sorted(df["r_scale"].unique())
    q_scales = sorted(df["q_rate_scale"].unique())

    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 2.1), sharey=True)
    im = None
    for j, m in enumerate(methods):
        sub = df[df["metodo"] == m].copy()
        sub["fr"] = sub["failures"] / sub["count"]
        piv = sub.pivot(index="q_rate_scale", columns="r_scale", values="fr").sort_index().sort_index(axis=1)
        ax = axes[j]
        im = ax.imshow(piv.values, aspect="auto", cmap="Greys", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(r_scales)))
        ax.set_xticklabels([("%g" % np.log10(r)) for r in r_scales], fontsize=6)
        ax.set_yticks(range(len(q_scales)))
        ax.set_yticklabels([("%g" % np.log10(q)) for q in q_scales], fontsize=7)
        ax.set_title(METHOD_LABEL[m], fontsize=8)
        ax.set_xlabel(r"$\log_{10}$ $R$ scale", fontsize=8)
        if j == 0:
            ax.set_ylabel(r"$\log_{10}$ $Q_{\omega}$ scale", fontsize=8)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.012)
    cb.set_label("failure fraction", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    out = os.path.join(outdir, "fig4_qr_safety_map.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("  " + out)


def fig5_closed_loop(outdir):
    """Fig. 5 -- malha fechada: custo acumulado J normalizado pelo do
    controlador float64 de referencia. Fonte: outputs/malha_fechada_trajetorias_v4.csv
    (python/malha_fechada_trajetorias.py, emulador Q13.18 sincronizado com
    gamma=0.7/tau=1e-3). Nota de escopo: nao inclui ITERATIVE_FIXED (o script
    so simula os 5 controladores de duplicacao _FIXED contra a referencia
    SDA_float64).

    Os cinco controladores ficam dentro de +-0.3% de J em todas as
    trajetorias -- ruido, nao sinal, numa escala onde o matplotlib
    auto-escalaria de forma enganosa. Eixo Y fixado explicitamente em
    +-0.5% e o valor numerico anotado sobre cada barra, em vez de deixar a
    escala amplificar visualmente uma diferenca que nao existe."""
    if not os.path.exists(MALHA):
        print("  [pulado] %s ainda nao existe" % MALHA)
        return
    df = pd.read_csv(MALHA)
    ctrls = ["SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"]
    trajs = [t for t in TRAJ_ORDER if t in df["traj"].unique()]

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    width = 0.15
    x = np.arange(len(trajs))
    ax.set_ylim(-0.5, 0.5)
    for i, c in enumerate(ctrls):
        vals = []
        for t in trajs:
            base = df[(df["traj"] == t) & (df["controller"] == "SDA_float64")]["J_total"].values
            cur = df[(df["traj"] == t) & (df["controller"] == c)]["J_total"].values
            vals.append((cur[0] / base[0] - 1.0) * 100.0 if len(base) and len(cur) else np.nan)
        xpos = x + (i - 2) * width
        bars = ax.bar(xpos, vals, width, label=METHOD_LABEL[c],
                       color=str(0.35 + 0.12 * i), edgecolor="black", linewidth=0.5, hatch=HATCH_FIXED)
        for xp, v in zip(xpos, vals):
            if np.isnan(v):
                continue
            va = "bottom" if v >= 0 else "top"
            offset = 0.02 if v >= 0 else -0.02
            ax.text(xp, v + offset, "%+.2f" % v, ha="center", va=va, fontsize=5.2, rotation=90)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([TRAJ_LABEL[t] for t in trajs], fontsize=8)
    ax.set_ylabel("LQR cost $J$ penalty\nvs. float64 (%)")
    ax.legend(frameon=False, ncol=5, fontsize=7, loc="upper center")
    fig.tight_layout()
    out = os.path.join(outdir, "fig5_closed_loop.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("  " + out)


def main(outdir):
    _setup_style()
    os.makedirs(outdir, exist_ok=True)
    print("gerando figuras em %s" % outdir)

    fig1_trajectories(outdir)

    pontos, runs, gains, summaries = load_device_csv(BATERIA)
    df_sum = summaries_to_df(summaries)
    fig2_timing(outdir, df_sum[df_sum["traj"] == "ALL"])

    print("  (calculando erro de K vs. scipy -- alguns minutos)")
    df_erro = gains_erro_k(gains)
    fig3_accuracy(outdir, df_erro)

    fig4_qr_map(outdir)
    fig5_closed_loop(outdir)
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=r"G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\artigo_diname\Figures")
    args = ap.parse_args()
    main(args.outdir)
