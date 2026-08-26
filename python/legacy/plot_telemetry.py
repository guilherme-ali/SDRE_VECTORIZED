"""
plot_telemetry.py — Lê um log de voo do drone SDRE e plota gráficos.

USO:
    python plot_telemetry.py                         # lista logs e pede escolha
    python plot_telemetry.py logs/device-xxx.log     # abre direto
    python plot_telemetry.py logs/device-xxx.log --save  # salva PNGs ao lado do log

FORMATO DO LOG:
    Procura blocos entre "=== TELEMETRY DUMP START ===" e "=== TELEMETRY DUMP END ===".
    Se houver múltiplos blocos (várias sessões no mesmo log), usa o maior.

FIGURAS GERADAS:
    Fig 1 — Atitude (roll / pitch / yaw) + Taxas angulares (p / q / r)
    Fig 2 — Torques de controle SDRE (u_roll / u_pitch / u_yaw)
    Fig 3 — Velocidade dos motores (ω = sqrt(ω²) em rad/s)
    Fig 4 — Diagnóstico de vibração: FFT do gyro + histograma de Δu
"""

import sys
import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.signal import welch

# ─── Parâmetros visuais ──────────────────────────────────────────────────────
COLORS = {
    "roll":    "#e74c3c",
    "pitch":   "#2980b9",
    "yaw":     "#27ae60",
    "p":       "#e74c3c",
    "q":       "#2980b9",
    "r":       "#27ae60",
    "u_roll":  "#e74c3c",
    "u_pitch": "#2980b9",
    "u_yaw":   "#27ae60",
    "w1":      "#e74c3c",
    "w2":      "#2980b9",
    "w3":      "#f39c12",
    "w4":      "#8e44ad",
}
PLT_STYLE = "seaborn-v0_8-darkgrid"

# Limites de sanidade para rejeitar linhas corrompidas (struct mismatch legado)
SANITY = {
    "t_ms":      (0,     1_000_000),
    "roll_deg":  (-180,  180),
    "pitch_deg": (-180,  180),
    "yaw_deg":   (-360,  360),
    "p_dps":     (-2000, 2000),
    "q_dps":     (-2000, 2000),
    "r_dps":     (-2000, 2000),
}

COLS = [
    "t_ms", "roll_deg", "pitch_deg", "yaw_deg",
    "roll_ref", "pitch_ref", "yaw_ref",
    "p_dps", "q_dps", "r_dps",
    "u_roll", "u_pitch", "u_yaw",
    "w1_sq", "w2_sq", "w3_sq", "w4_sq",
]

# ─── Extração do CSV ──────────────────────────────────────────────────────────

def _extract_blocks(text: str) -> list[list[str]]:
    """Retorna lista de blocos de linhas CSV (entre os marcadores)."""
    blocks = []
    current = []
    inside = False
    for line in text.splitlines():
        if "TELEMETRY DUMP START" in line:
            inside = True
            current = []
        elif "TELEMETRY DUMP END" in line:
            if current:
                blocks.append(current)
            inside = False
        elif inside:
            current.append(line.strip())
    return blocks


def _parse_block(lines: list[str]) -> pd.DataFrame | None:
    """Converte linhas de um bloco em DataFrame. Retorna None se vazio."""
    data_lines = []
    for line in lines:
        # Pula header, 'Samples:', linhas vazias
        if not line or not line[0].isdigit() and not line[0] == "-":
            continue
        parts = line.split(",")
        if len(parts) != 17:
            continue
        try:
            row = [float(p) for p in parts]
        except ValueError:
            continue
        data_lines.append(row)
    if not data_lines:
        return None
    df = pd.DataFrame(data_lines, columns=COLS)
    return df


def _apply_sanity(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in SANITY.items():
        if col in df.columns:
            mask &= df[col].between(lo, hi)
    df = df[mask].copy()
    df = df.drop_duplicates(subset="t_ms")
    df = df.sort_values("t_ms").reset_index(drop=True)
    return df


def parse_legacy_log(text: str) -> pd.DataFrame | None:
    """Faz o parse do formato antigo de log que printava continuamente."""
    pattern = re.compile(r"Roll:([\-\.\d]+),Pitch:([\-\.\d]+),Yaw:([\-\.\d]+),P:([\-\.\d]+),Q:([\-\.\d]+),R:([\-\.\d]+)", re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        return None
    
    data = []
    t_ms = 0
    dt_ms = 20 # Assume 50Hz for legacy logs
    for m in matches:
        data.append([
            t_ms, float(m[0]), float(m[1]), float(m[2]),
            0.0, 0.0, 0.0,
            float(m[3]), float(m[4]), float(m[5]),
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ])
        t_ms += dt_ms
        
    df = pd.DataFrame(data, columns=COLS)
    return df


def load_log(path: str) -> tuple[pd.DataFrame, str]:
    """Carrega o log, escolhe o maior bloco, aplica filtro de sanidade."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = _extract_blocks(text)
    
    df = None
    n_blocks = 0
    
    if blocks:
        dfs = [_parse_block(b) for b in blocks]
        dfs = [d for d in dfs if d is not None and len(d) >= 5]
        if dfs:
            df = max(dfs, key=len)
            n_blocks = len(dfs)
            
    if df is None:
        df = parse_legacy_log(text)
        if df is None:
            raise ValueError(f"Nenhum dado de telemetria válido encontrado em: {path}")
        print("  [aviso] Log em formato legado detectado. Assumindo dt=20ms e omitindo motores/controle.")
        n_blocks = 1

    before = len(df)
    df = _apply_sanity(df)
    removed = before - len(df)
    if removed > 0:
        print(f"  [sanidade] {removed}/{before} linhas rejeitadas (corrupção de struct)")

    # Tempo relativo em segundos
    df["t_s"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0

    # Omega em rad/s (sqrt de w_sq, clamp em 0)
    for i, col in enumerate(["w1_sq", "w2_sq", "w3_sq", "w4_sq"], start=1):
        df[f"w{i}"] = np.sqrt(np.clip(df[col], 0, None))

    # Detecta fase de voo (motores realmente girando)
    w_sum = df["w1_sq"] + df["w2_sq"] + df["w3_sq"] + df["w4_sq"]
    HOVER_THRESHOLD = 1e5  # ω² mínimo relevante
    df["flying"] = w_sum > HOVER_THRESHOLD

    log_name = Path(path).name
    info = f"{log_name}  ·  {len(df)} amostras  ·  {df['t_s'].iloc[-1]:.1f} s  ·  {n_blocks} bloco(s)"
    return df, info


# ─── Helpers de plot ─────────────────────────────────────────────────────────

def _shade_flight(ax, df: pd.DataFrame):
    """Fundo azul claro nas regiões onde drone está voando."""
    t = df["t_s"].values
    fl = df["flying"].values
    in_flight = False
    t_start = None
    for i, f in enumerate(fl):
        if f and not in_flight:
            t_start = t[i]
            in_flight = True
        elif not f and in_flight:
            ax.axvspan(t_start, t[i], color="#aed6f1", alpha=0.25, linewidth=0)
            in_flight = False
    if in_flight:
        ax.axvspan(t_start, t[-1], color="#aed6f1", alpha=0.25, linewidth=0)


def _twin_legend(ax, handles, labels, loc="upper right"):
    ax.legend(handles, labels, loc=loc, fontsize=8, framealpha=0.7)


# ─── Figuras ─────────────────────────────────────────────────────────────────

def fig_attitude(df: pd.DataFrame, title_prefix: str) -> plt.Figure:
    """Fig 1: Atitude (roll/pitch/yaw) + Taxas angulares (p/q/r)"""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"{title_prefix}\nAtitude e Taxas Angulares", fontsize=11, fontweight="bold")

    t = df["t_s"]

    # Atitude
    ax = axes[0]
    for col, label, c in [("roll_deg","roll","roll"), ("pitch_deg","pitch","pitch"), ("yaw_deg","yaw","yaw")]:
        # Medido (linha cheia)
        ax.plot(t, df[col], color=COLORS[c], lw=1.2, label=label, alpha=0.9)
        
        # Referência (linha tracejada)
        ref_col = col.replace("_deg", "_ref")
        if ref_col in df.columns:
            ax.plot(t, df[ref_col], color=COLORS[c], lw=1.0, ls="--", alpha=0.5, label=f"{label}_ref")

    _shade_flight(ax, df)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Ângulo (°)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Atitude (Euler) — Linha cheia: medido, Tracejada: referência", fontsize=9)

    # Taxas
    ax = axes[1]
    for col, label, c in [("p_dps","p (roll)","p"), ("q_dps","q (pitch)","q"), ("r_dps","r (yaw)","r")]:
        ax.plot(t, df[col], color=COLORS[c], lw=1.0, label=label)
    _shade_flight(ax, df)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Taxa angular (°/s)")
    ax.set_xlabel("Tempo (s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Taxas Angulares (corpo)", fontsize=9)

    fig.tight_layout()
    return fig


def fig_control(df: pd.DataFrame, title_prefix: str) -> plt.Figure:
    """Fig 2: Torques SDRE — u_roll / u_pitch / u_yaw (mN·m)"""
    # Converter N·m → mN·m para legibilidade
    df = df.copy()
    for col in ["u_roll", "u_pitch", "u_yaw"]:
        df[col] *= 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"{title_prefix}\nTorques de Controle SDRE", fontsize=11, fontweight="bold")

    pairs = [("u_roll","Roll","u_roll"), ("u_pitch","Pitch","u_pitch"), ("u_yaw","Yaw","u_yaw")]
    for ax, (col, label, c) in zip(axes, pairs):
        ax.plot(df["t_s"], df[col], color=COLORS[c], lw=1.0, label=f"u_{label.lower()}")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        _shade_flight(ax, df)
        ax.set_ylabel(f"τ_{label.lower()} (mN·m)")
        ax.legend(fontsize=8, loc="upper right")
        # Anotação: std em região de voo
        voo = df[df["flying"]]
        if len(voo) > 5:
            std = voo[col].std()
            ax.set_title(f"{label}  —  σ={std:.3f} mN·m em voo", fontsize=9)

    axes[-1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    return fig


def fig_motors(df: pd.DataFrame, title_prefix: str) -> plt.Figure:
    """Fig 3: Velocidade dos motores (ω rad/s) + RPM estimado."""
    RPM_FACTOR = 60.0 / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"{title_prefix}\nVelocidade dos Motores", fontsize=11, fontweight="bold")

    ax = axes[0]
    motor_labels = [
        ("w1", "M1 FR (CW)",  "w1"),
        ("w2", "M2 RR (CCW)", "w2"),
        ("w3", "M3 RL (CW)",  "w3"),
        ("w4", "M4 FL (CCW)", "w4"),
    ]
    for col, label, c in motor_labels:
        ax.plot(df["t_s"], df[col], color=COLORS[c], lw=1.0, label=label)
    _shade_flight(ax, df)
    ax.set_ylabel("ω (rad/s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Velocidade angular dos motores", fontsize=9)

    # Desequilíbrio: max - min entre os 4 motores (por amostra)
    ax2 = axes[1]
    omega_mat = df[["w1","w2","w3","w4"]].values
    delta = omega_mat.max(axis=1) - omega_mat.min(axis=1)
    ax2.plot(df["t_s"], delta, color="#7f8c8d", lw=0.9, label="Δω max-min")
    _shade_flight(ax2, df)
    ax2.set_ylabel("Δω (rad/s)")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_title("Desequilíbrio entre motores (max−min)", fontsize=9)
    ax2.legend(fontsize=8)

    # Anotar motores saturados em zero
    for col, label, c in motor_labels:
        zeros_pct = (df[col] == 0).mean() * 100
        if zeros_pct > 1:
            ax.annotate(f"{label}: {zeros_pct:.0f}% em 0",
                        xy=(0.01, 0.05 + motor_labels.index((col,label,c)) * 0.08),
                        xycoords="axes fraction", fontsize=7, color=COLORS[c])

    fig.tight_layout()
    return fig


def fig_vibration(df: pd.DataFrame, title_prefix: str) -> plt.Figure:
    """Fig 4: Diagnóstico de vibração — PSD do gyro + histograma de Δu."""
    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(f"{title_prefix}\nDiagnóstico de Vibração", fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Estimativa de dt (usa mediana de diffs pra robustez)
    dt = np.median(np.diff(df["t_s"].values))
    fs = 1.0 / dt if dt > 0 else 50.0

    # PSD do gyro (Welch)
    for idx, (col, label, c) in enumerate([
        ("p_dps","p (roll)","p"), ("q_dps","q (pitch)","q"), ("r_dps","r (yaw)","r")
    ]):
        ax = fig.add_subplot(gs[0, idx])
        sig = df[col].values - df[col].mean()
        f_ax, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)//4))
        ax.semilogy(f_ax, psd, color=COLORS[c], lw=1.2)
        ax.set_title(f"PSD gyro {label}", fontsize=9)
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("(°/s)²/Hz")
        ax.set_xlim(0, fs/2)

    # Histograma de Δu (variação amostral dos torques) — revela chattering
    ax_hist = fig.add_subplot(gs[1, :])
    voo = df[df["flying"]]
    bins = 80
    for col, label, c in [("u_roll","u_roll","u_roll"),("u_pitch","u_pitch","u_pitch"),("u_yaw","u_yaw","u_yaw")]:
        if len(voo) > 2:
            delta = np.diff(voo[col].values) * 1000  # mN·m
            ax_hist.hist(delta, bins=bins, alpha=0.5, color=COLORS[c], label=label, density=True)
    ax_hist.set_xlabel("Δtorque entre amostras consecutivas (mN·m)")
    ax_hist.set_ylabel("Densidade")
    ax_hist.set_title("Variação amostral dos torques em voo  (chattering → distribuição larga)", fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.axvline(0, color="k", lw=0.8, ls="--")

    # Caixa de estatísticas
    if len(voo) > 2:
        stats_lines = []
        for col, label in [("p_dps","p"),("q_dps","q"),("r_dps","r")]:
            std = voo[col].std()
            peak = voo[col].abs().max()
            stats_lines.append(f"σ({label})={std:.1f} dps  peak={peak:.1f} dps")
        stats_lines.append(f"N amostras em voo: {len(voo)}")
        fig.text(0.01, 0.3, "\n".join(stats_lines), fontsize=8,
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def pick_log_interactively() -> str:
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("Pasta 'logs/' não encontrada. Execute na raiz do projeto.")
    logs = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        raise FileNotFoundError("Nenhum arquivo .log em logs/")

    print("\nLogs disponíveis (mais recentes primeiro):")
    for i, p in enumerate(logs[:30]):
        size_kb = p.stat().st_size / 1024
        print(f"  {i+1:2d}.  {p.name}  ({size_kb:.0f} KB)")
    if len(logs) > 30:
        print(f"  ... (+{len(logs)-30} arquivos omitidos)")

    while True:
        try:
            choice = input("\nNúmero do log (ou caminho completo): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(logs):
                    return str(logs[idx])
            elif os.path.isfile(choice):
                return choice
            print("  Opção inválida, tente novamente.")
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Plota gráficos de telemetria do drone SDRE a partir de um log.")
    parser.add_argument("log", nargs="?", help="Caminho do arquivo .log")
    parser.add_argument("--save", action="store_true",
                        help="Salva as figuras como PNG no mesmo diretório do log")
    parser.add_argument("--no-show", action="store_true",
                        help="Não exibe janelas (útil em pipelines)")
    args = parser.parse_args()

    log_path = args.log or pick_log_interactively()

    print(f"\nCarregando: {log_path}")
    df, info = load_log(log_path)
    print(f"  {info}")

    try:
        plt.style.use(PLT_STYLE)
    except OSError:
        plt.style.use("seaborn-darkgrid")

    prefix = Path(log_path).stem

    figs = [
        ("atitude",    fig_attitude(df, prefix)),
        ("controle",   fig_control(df, prefix)),
        ("motores",    fig_motors(df, prefix)),
        ("vibracao",   fig_vibration(df, prefix)),
    ]

    if args.save:
        base = Path(log_path).parent
        for name, fig in figs:
            out = base / f"{Path(log_path).stem}_{name}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Salvo: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
