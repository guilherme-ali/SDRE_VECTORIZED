"""
Consolida os três resultados da bateria de trajetórias em um relatório único
(markdown + figuras): a captura on-device (experiments/benchmark_solvers.cpp), a
malha fechada (malha_fechada_trajetorias.py) e a comparação contra a
referência scipy (bench_trajetorias.py --compare).

Uso:
    python gerar_relatorio_bateria.py \
        --device outputs/serial_capture_bateria_trajetorias.txt \
        --malha-fechada outputs/malha_fechada_trajetorias.csv \
        --saida outputs/relatorio_bateria.md
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
from bench_trajetorias import load_device_csv, build_Ad_Bd_Qd_Rd, k_from_P, DT
import trajetorias as trj
from scipy.linalg import solve_discrete_are

TRAJ_ORDER = ["T1_espiral", "T2_figura8", "T3_chirp", "T4_degrau_yaw"]
METHOD_ORDER = ["SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA", "ITERATIVE",
                "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED",
                "ADDA_FIXED", "ITERATIVE_FIXED"]

def summaries_to_df(summaries):
    cols = ["traj", "metodo", "mean_us", "std_us", "max_us", "mean_iters",
            "max_iters", "mean_res", "max_res", "failures", "count"]
    rows = []
    for row in summaries:
        d = dict(zip(cols, row))
        rows.append(d)
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def gains_erro_k(gains, sample_every=1):
    """Erro relativo de Frobenius de K (device) vs. referencia scipy, por
    metodo/trajetoria: ||K_dev-K_ref||_F / ||K_ref||_F (a mesma definicao de
    e_K usada no artigo, Secao "Reference solution and metrics").
    Usa as trajetorias completas pre-computadas (T4 depende de recursao desde
    k=0 — nao dá pra reconstruir um ponto isolado sem rodar a serie inteira,
    mas gerar as 4 series completas custa menos de 1s com numpy)."""
    dados_completos = trj.gerar_todas()
    rows = []
    for i, g in enumerate(gains):
        if i % sample_every != 0:
            continue
        traj, k, metodo = g[0], int(g[1]), g[2]
        d = dados_completos.get(traj)
        if d is None or k >= len(d["t"]):
            continue
        K_dev = np.array([float(x) for x in g[3:3 + 18]]).reshape(3, 6)
        phi, theta, p, q, r = d["phi"][k], d["theta"][k], d["p"][k], d["q"][k], d["r"][k]
        Ad, Bd, Qd, Rd = build_Ad_Bd_Qd_Rd(phi, theta, p, q, r)
        try:
            P_ref = solve_discrete_are(Ad, Bd, Qd, Rd)
        except Exception:
            continue
        K_ref = k_from_P(Ad, Bd, Qd, Rd, P_ref)
        norm_ref = np.linalg.norm(K_ref)
        e = float(np.linalg.norm(K_dev - K_ref) / norm_ref) if norm_ref > 0 else float("nan")
        rows.append({"traj": traj, "k": k, "metodo": metodo, "erro_K": e})
    return pd.DataFrame(rows)


def fig_tempos(df_sum, outpath):
    trajs = [t for t in TRAJ_ORDER if t in df_sum["traj"].unique()]
    fig, axes = plt.subplots(1, len(trajs), figsize=(5 * len(trajs), 5), sharey=True)
    if len(trajs) == 1:
        axes = [axes]
    for ax, traj in zip(axes, trajs):
        sub = df_sum[df_sum["traj"] == traj].set_index("metodo").reindex(
            [m for m in METHOD_ORDER if m in df_sum["metodo"].unique()])
        ax.bar(sub.index, sub["mean_us"], yerr=sub["std_us"], capsize=3)
        ax.axhline(12500, color="red", linestyle="--", linewidth=1, label="teto 80 Hz (12,5 ms)")
        ax.set_title(traj)
        ax.set_ylabel("tempo medio (us)")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def fig_erro_k(df_erro, outpath):
    agg = df_erro.groupby(["traj", "metodo"])["erro_K"].agg(["mean", "max"]).reset_index()
    trajs = [t for t in TRAJ_ORDER if t in agg["traj"].unique()]
    fig, axes = plt.subplots(1, len(trajs), figsize=(5 * len(trajs), 5), sharey=True)
    if len(trajs) == 1:
        axes = [axes]
    for ax, traj in zip(axes, trajs):
        sub = agg[agg["traj"] == traj].set_index("metodo").reindex(
            [m for m in METHOD_ORDER if m in agg["metodo"].unique()])
        ax.bar(sub.index, sub["mean"], label="medio")
        ax.scatter(sub.index, sub["max"], color="red", marker="x", label="pior caso", zorder=3)
        ax.set_yscale("log")
        ax.set_title(traj)
        ax.set_ylabel("erro RMS(K) vs. scipy (log)")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def fig_malha_fechada(df_mf, outpath):
    trajs = [t for t in TRAJ_ORDER if t in df_mf["traj"].unique()]
    fig, axes = plt.subplots(1, len(trajs), figsize=(5 * len(trajs), 5), sharey=False)
    if len(trajs) == 1:
        axes = [axes]
    for ax, traj in zip(axes, trajs):
        sub = df_mf[df_mf["traj"] == traj].set_index("controller")
        order = [c for c in ["SDA_float64", "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED",
                              "SDA_SCALED_FIXED", "ADDA_FIXED"] if c in sub.index]
        sub = sub.reindex(order)
        colors = ["#333333" if c == "SDA_float64" else "#1f77b4" for c in order]
        ax.bar(sub.index, sub["J_total"], color=colors)
        ax.set_title(traj)
        ax.set_ylabel("custo acumulado J")
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main(device_path, malha_fechada_path, saida_md):
    outdir = os.path.dirname(saida_md)
    os.makedirs(outdir, exist_ok=True)

    pontos, runs, gains, summaries = load_device_csv(device_path)
    df_sum = summaries_to_df(summaries)
    df_sum_traj = df_sum[df_sum["traj"] != "ALL"]
    df_sum_all = df_sum[df_sum["traj"] == "ALL"]

    print("processando erro de K (pode levar alguns minutos)...")
    df_erro = gains_erro_k(gains)

    df_mf = pd.read_csv(malha_fechada_path) if os.path.exists(malha_fechada_path) else None

    fig_tempos(df_sum_traj, os.path.join(outdir, "fig_tempos_bateria.png"))
    fig_erro_k(df_erro, os.path.join(outdir, "fig_erro_k_bateria.png"))
    if df_mf is not None:
        fig_malha_fechada(df_mf, os.path.join(outdir, "fig_malha_fechada.png"))

    lines = []
    lines.append("# Relatório — bateria de trajetórias (ESP32-S2)\n")
    lines.append(f"Pontos capturados: {len(pontos)}. Linhas RUN: {len(runs)}. "
                 f"Linhas GAIN: {len(gains)}. Amostras de erro de K calculadas: {len(df_erro)}.\n")

    lines.append("## Resumo geral (todas as trajetórias)\n")
    lines.append(df_sum_all.sort_values("mean_us")[
        ["metodo", "mean_us", "std_us", "max_us", "mean_iters", "mean_res", "failures", "count"]
    ].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    for traj in TRAJ_ORDER:
        if traj not in df_sum_traj["traj"].unique():
            continue
        lines.append(f"## {traj}\n")
        sub = df_sum_traj[df_sum_traj["traj"] == traj].sort_values("mean_us")
        lines.append(sub[["metodo", "mean_us", "std_us", "max_us", "mean_iters", "mean_res", "failures"]]
                      .to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        erro_sub = df_erro[df_erro["traj"] == traj].groupby("metodo")["erro_K"].agg(["mean", "max"])
        if not erro_sub.empty:
            lines.append("Erro RMS(K) vs. referência scipy (dupla precisão):\n")
            lines.append(erro_sub.sort_values("mean").to_markdown(floatfmt=".3e"))
            lines.append("")

    if df_mf is not None:
        lines.append("## Malha fechada (rastreamento + custo acumulado)\n")
        lines.append(df_mf[["traj", "controller", "rms_total_deg", "J_total", "n_fallback"]]
                      .to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Figuras\n")
    lines.append("- `fig_tempos_bateria.png` — tempo médio ± desvio por método, por trajetória, com o teto de 80 Hz.")
    lines.append("- `fig_erro_k_bateria.png` — erro RMS(K) vs. referência scipy, médio e pior caso, por trajetória.")
    lines.append("- `fig_malha_fechada.png` — custo acumulado J em malha fechada, float64 vs. Q13.18, por trajetória.")

    with open(saida_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"gravado: {saida_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "serial_capture_bateria_v5_6traj.txt"))
    ap.add_argument("--malha-fechada", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "malha_fechada_v6_6traj.csv"))
    ap.add_argument("--saida", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "relatorio_bateria.md"))
    args = ap.parse_args()
    main(args.device, args.malha_fechada, args.saida)
