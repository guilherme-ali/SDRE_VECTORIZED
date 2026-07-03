"""
identify_inertia_ols.py — Identificacao CONJUNTA de Ixx, Iyy, Izz e Ir por
minimos quadrados ordinarios (equation-error method, Klein & Morelli 2006),
a partir de um unico log de voo (telemetria do firmware SDRE_VECTORIZED).

METODO: as equacoes de Euler de corpo rigido com acoplamento giroscopico do
rotor (mesma convencao de updateSystemMatrix() em src/main.cpp) sao

    Ixx*dp/dt = (Iyy-Izz)*q*r - Ir*q*omega_r + tau_x
    Iyy*dq/dt = (Izz-Ixx)*p*r + Ir*p*omega_r + tau_y
    Izz*dr/dt = (Ixx-Iyy)*p*q                + tau_z

que sao lineares nos quatro momentos de inercia (equation-error method).
Isolando os torques -- ja conhecidos, pois b e d foram caracterizados em
bancada -- e agrupando os termos que multiplicam cada momento de inercia,
obtem-se um sistema linear UNICO no vetor theta = [Ixx, Iyy, Izz, Ir]^T:

    tau_x =  Ixx*dp   - Iyy*(q*r) + Izz*(q*r)  + Ir*(q*omega_r) + c0x
    tau_y =  Ixx*(p*r) + Iyy*dq   - Izz*(p*r)  - Ir*(p*omega_r) + c0y
    tau_z = -Ixx*(p*q) + Iyy*(p*q) + Izz*dr                     + c0z

em que c0x, c0y, c0z sao interceptos proprios de cada eixo, que absorvem
offsets de trim e a polarizacao do giroscopio. Cada linha, a cada instante
de amostragem, contribui uma amostra da matriz de regressores X; empilhando
as tres equacoes ao longo de toda a janela de telemetria, o sistema e
resolvido de uma so vez por minimos quadrados ordinarios:

    theta_hat = (X^T X)^-1 X^T y

y reune as amostras de tau_x, tau_y, tau_z e X as colunas de regressores
acima. Os erros-padrao dos parametros provem da matriz de covariancia
sigma^2*(X^T X)^-1, e a qualidade do ajuste e aferida pelo R^2.

Vantagem sobre uma regressao eixo-a-eixo: Ixx, Iyy e Izz aparecem em MAIS
DE UMA das tres equacoes -- por exemplo Izz aparece via q*r (eq. x), p*r
(eq. y) E dr (eq. z) -- entao mesmo um eixo pouco excitado (ex.: yaw, se
yaw_ref nunca for comandado) ainda se beneficia da informacao vinda dos
outros dois na regressao conjunta.

USO:
    python identify_inertia_ols.py                       # log padrao (melhor voo)
    python identify_inertia_ols.py logs/outro.log
    python identify_inertia_ols.py --fc 8 --tau-motor 60 --no-plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter

# ─── Parâmetros físicos CONHECIDOS (b, d, L medidos em bancada) ───────────────
# IMPORTANTE: usar os valores que estavam EM VIGOR quando o log foi gravado
# (commit 19b4016, 2026-06-02 16:31 — 1h11 antes do log), NÃO os valores atuais
# do working tree do main.cpp (2.98e-8 / 0.044): estes são de uma identificação
# POSTERIOR (commits 772083f/4d3e8ee/1e662b6, 2026-06-30/07-01) e ainda não
# commitada/re-flashada. Usar os valores atuais aqui contaminaria a reconstrução
# do torque com um d/b sistematicamente errado (confirmado empiricamente: o d/b
# que minimiza o resíduo contra u_yaw real neste log é ≈0.0493 ≈ 0.05, não 0.044).
B_COEFF = 2.94e-8  # N/(rad/s)^2 — empuxo medido (hélice 55mm, main.cpp@19b4016:56)
D_COEFF = 0.05 * B_COEFF  # N·m/(rad/s)^2 — arrasto medido (main.cpp@19b4016:59)
L_ARM = 0.060 * 0.70710678  # m — braço efetivo em config X (main.cpp:46, inalterado)
MAX_RPM = 26423.0  # RPM @ 100% duty (main.cpp@19b4016:58, inalterado)
MAX_OMEGA = MAX_RPM * 2.0 * np.pi / 60.0
MAX_OMEGA_SQ = MAX_OMEGA**2  # limite de clamp do firmware

# Valores ATUAIS do firmware (working tree, main.cpp:42-45 — inclui correções
# posteriores ao log; usados apenas como referência de comparação no relatório)
FW = dict(Ixx=16.57e-6, Iyy=15.57e-6, Izz=29.80e-6, Ir=1.02e-7)

COLS = [
    "t_ms",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "roll_ref",
    "pitch_ref",
    "yaw_ref",
    "p_dps",
    "q_dps",
    "r_dps",
    "u_roll",
    "u_pitch",
    "u_yaw",
    "w1_sq",
    "w2_sq",
    "w3_sq",
    "w4_sq",
]

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_LOG = (
    LOG_DIR / "device-monitor-260602-174243.log"
)  # melhor voo (excitação suficiente)
DEFAULT_LOG = (
    LOG_DIR / "device-monitor-260703-113154.log"
)  # melhor voo (excitação suficiente

# ─── Parsing (mesmo formato de plot_telemetry.py / identify_params.py) ────────


def extract_blocks(text: str) -> list[list[str]]:
    blocks, current, inside = [], [], False
    for line in text.splitlines():
        if "TELEMETRY DUMP START" in line:
            inside, current = True, []
        elif "TELEMETRY DUMP END" in line:
            if current:
                blocks.append(current)
            inside = False
        elif inside:
            current.append(line.strip())
    return blocks


def parse_log(path: Path) -> pd.DataFrame | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = extract_blocks(text)
    if not blocks:
        return None
    best = max(blocks, key=len)
    rows = []
    for line in best:
        parts = line.split(",")
        if len(parts) != 17:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COLS)
    df = df.drop_duplicates(subset="t_ms").sort_values("t_ms").reset_index(drop=True)
    return df


def check_mixer(df: pd.DataFrame):
    """Sanity check: confere se b, L, d acima (época do log) batem com o voo.

    Nas amostras sem clamp, a alocação física direta é exata:
        b·L·(-w1-w2+w3+w4) == 2·u_roll   (fator 2: bug conhecido da inversa do firmware)
        b·L·(+w1-w2-w3+w4) == 2·u_pitch
        d·(-w1+w2-w3+w4)   == 1·u_yaw
    Erro ~0 => mesmo firmware; erro grande => época diferente (log suspeito).
    """
    w = df[["w1_sq", "w2_sq", "w3_sq", "w4_sq"]].to_numpy()
    free = (w > 1.0).all(axis=1)
    if free.sum() < 50:
        return (np.nan, np.nan, np.nan, float(free.mean()))
    w = w[free]
    tau_r = B_COEFF * L_ARM * (-w[:, 0] - w[:, 1] + w[:, 2] + w[:, 3])
    tau_p = B_COEFF * L_ARM * (+w[:, 0] - w[:, 1] - w[:, 2] + w[:, 3])
    tau_y = D_COEFF * (-w[:, 0] + w[:, 1] - w[:, 2] + w[:, 3])

    def rel_err(a, b_):
        s = np.std(b_)
        return np.nan if s == 0 else float(np.std(a - b_) / s)

    return (
        rel_err(tau_r, 2.0 * df["u_roll"].to_numpy()[free]),
        rel_err(tau_p, 2.0 * df["u_pitch"].to_numpy()[free]),
        rel_err(tau_y, 1.0 * df["u_yaw"].to_numpy()[free]),
        float(free.mean()),
    )


# ─── Pré-processamento ─────────────────────────────────────────────────────────


def preprocess(df: pd.DataFrame, fs: float, fc: float, tau_motor_ms: float):
    """Retorna dict de sinais filtrados em grade uniforme + máscara de validade."""
    t = df["t_ms"].to_numpy() / 1000.0
    d2r = np.pi / 180.0
    p = df["p_dps"].to_numpy() * d2r
    q = df["q_dps"].to_numpy() * d2r
    r = df["r_dps"].to_numpy() * d2r

    w_sq_raw = df[["w1_sq", "w2_sq", "w3_sq", "w4_sq"]].to_numpy()
    # ω² logado é pré-clamp superior; o motor recebe o valor saturado em [0, MAX]
    w_sq = np.clip(w_sq_raw, 0.0, MAX_OMEGA_SQ)
    sat = (w_sq_raw < 1.0) | (w_sq_raw > 1.02 * MAX_OMEGA_SQ)
    sat_any = sat.any(axis=1)

    valid_src = (
        ~sat_any
        & (np.abs(df["roll_deg"].to_numpy()) < 60.0)
        & (np.abs(df["pitch_deg"].to_numpy()) < 60.0)
        & (np.abs(df["p_dps"].to_numpy()) < 1500.0)
        & (np.abs(df["q_dps"].to_numpy()) < 1500.0)
        & (np.abs(df["r_dps"].to_numpy()) < 1500.0)
        & (w_sq.mean(axis=1) > 0.05 * MAX_OMEGA_SQ)  # motores realmente girando
    )

    tu = np.arange(t[0], t[-1], 1.0 / fs)

    def interp(y):
        return np.interp(tu, t, y)

    sig = dict(
        p=interp(p),
        q=interp(q),
        r=interp(r),
        w1=interp(np.sqrt(w_sq[:, 0])),
        w2=interp(np.sqrt(w_sq[:, 1])),
        w3=interp(np.sqrt(w_sq[:, 2])),
        w4=interp(np.sqrt(w_sq[:, 3])),
        w1s=interp(w_sq[:, 0]),
        w2s=interp(w_sq[:, 1]),
        w3s=interp(w_sq[:, 2]),
        w4s=interp(w_sq[:, 3]),
    )

    idx_near = np.clip(np.searchsorted(t, tu), 0, len(t) - 1)
    mask = valid_src[idx_near]
    # Erosão (filtfilt espalha contaminação de amostras inválidas)
    guard = int(0.10 * fs)
    bad = ~mask
    if bad.any():
        bad = np.convolve(bad.astype(float), np.ones(2 * guard + 1), mode="same") > 0
        mask = ~bad

    # Atraso de 1ª ordem do motor (ω real responde ao comando com lag)
    if tau_motor_ms > 0:
        alpha = np.exp(-1.0 / (fs * tau_motor_ms * 1e-3))
        for k in ("w1", "w2", "w3", "w4", "w1s", "w2s", "w3s", "w4s"):
            sig[k] = lfilter([1 - alpha], [1, -alpha], sig[k])

    omega_r = -sig["w1"] + sig["w2"] - sig["w3"] + sig["w4"]
    tau_x = B_COEFF * L_ARM * (-sig["w1s"] - sig["w2s"] + sig["w3s"] + sig["w4s"])
    tau_y = B_COEFF * L_ARM * (+sig["w1s"] - sig["w2s"] - sig["w3s"] + sig["w4s"])
    tau_z = D_COEFF * (-sig["w1s"] + sig["w2s"] - sig["w3s"] + sig["w4s"])

    # Passa-baixas zero-phase idêntico em ambos os lados da regressão
    b_f, a_f = butter(4, fc / (fs / 2.0))

    # Fatores pré-filtrados antes do produto (evita retificação de ruído de alta
    # frequência no produto); o produto final também é filtrado logo abaixo.
    pb = filtfilt(b_f, a_f, sig["p"])
    qb = filtfilt(b_f, a_f, sig["q"])
    rb = filtfilt(b_f, a_f, sig["r"])

    raw = dict(
        p=sig["p"],
        q=sig["q"],
        r=sig["r"],
        qr=qb * rb,
        pr=pb * rb,
        pq=pb * qb,
        q_omega_r=qb * omega_r,
        p_omega_r=pb * omega_r,
        tau_x=tau_x,
        tau_y=tau_y,
        tau_z=tau_z,
    )
    filt = {k: filtfilt(b_f, a_f, v) for k, v in raw.items()}

    dt = 1.0 / fs
    filt["dp"] = np.gradient(filt["p"], dt)
    filt["dq"] = np.gradient(filt["q"], dt)
    filt["dr"] = np.gradient(filt["r"], dt)
    filt["one"] = np.ones_like(filt["p"])

    edge = int(0.25 * fs)
    mask[:edge] = False
    mask[-edge:] = False
    return filt, mask, tu


# ─── Regressão conjunta (equation-error method) ────────────────────────────────


def build_stacked_system(filt: dict, mask: np.ndarray):
    """Monta o sistema linear empilhado para theta=[Ixx,Iyy,Izz,Ir,c0x,c0y,c0z].

    tau_x =  Ixx*dp   - Iyy*qr + Izz*qr + Ir*(q*omega_r)  + c0x
    tau_y =  Ixx*pr   + Iyy*dq - Izz*pr - Ir*(p*omega_r)  + c0y
    tau_z = -Ixx*pq   + Iyy*pq + Izz*dr                    + c0z
    """
    m = mask
    n = int(m.sum())
    zero_col = np.zeros(n)
    one_col = filt["one"][m]

    X_x = np.column_stack(
        [
            filt["dp"][m],
            -filt["qr"][m],
            filt["qr"][m],
            filt["q_omega_r"][m],
            one_col,
            zero_col,
            zero_col,
        ]
    )
    X_y = np.column_stack(
        [
            filt["pr"][m],
            filt["dq"][m],
            -filt["pr"][m],
            -filt["p_omega_r"][m],
            zero_col,
            one_col,
            zero_col,
        ]
    )
    X_z = np.column_stack(
        [
            -filt["pq"][m],
            filt["pq"][m],
            filt["dr"][m],
            zero_col,
            zero_col,
            zero_col,
            one_col,
        ]
    )

    X = np.vstack([X_x, X_y, X_z])
    y = np.concatenate([filt["tau_x"][m], filt["tau_y"][m], filt["tau_z"][m]])
    return X, y, (X_x, X_y, X_z)


def ols(y: np.ndarray, X: np.ndarray):
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    stderr = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return theta, stderr, r2, resid, cov


def r2_of(y_blk: np.ndarray, resid_blk: np.ndarray) -> float:
    ss_tot = float(((y_blk - y_blk.mean()) ** 2).sum())
    return 1.0 - float(resid_blk @ resid_blk) / ss_tot if ss_tot > 0 else np.nan


def physical_checks(
    Ixx: float, Iyy: float, Izz: float, Ir: float, yaw_ref_std_deg: float
) -> list[str]:
    """Valida theta_hat contra restrições físicas do tensor de inércia que NÃO
    fazem parte do modelo de regressão (por isso podem ser violadas pelo OLS
    quando faltar excitação em algum eixo). Retorna lista de avisos (vazia se
    tudo consistente)."""
    warnings = []
    for name, val in (("Ixx", Ixx), ("Iyy", Iyy), ("Izz", Izz), ("Ir", Ir)):
        if val <= 0:
            warnings.append(
                f"{name} <= 0 ({val:.3e}) — fisicamente inválido, estimativa não confiável"
            )
    if Izz > Ixx + Iyy:
        warnings.append(
            f"Izz > Ixx+Iyy ({Izz:.3e} > {(Ixx+Iyy):.3e}) — viola a desigualdade "
            "triangular do tensor de inércia (|Ii-Ij| <= Ik <= Ii+Ij)"
        )
    if Ixx > Iyy + Izz:
        warnings.append(
            f"Ixx > Iyy+Izz ({Ixx:.3e} > {(Iyy+Izz):.3e}) — viola a desigualdade triangular"
        )
    if Iyy > Ixx + Izz:
        warnings.append(
            f"Iyy > Ixx+Izz ({Iyy:.3e} > {(Ixx+Izz):.3e}) — viola a desigualdade triangular"
        )
    if yaw_ref_std_deg < 1.0:
        warnings.append(
            f"yaw nunca comandado neste voo (std(yaw_ref)={yaw_ref_std_deg:.2f}°) — "
            "eq. z (yaw) contribui pouca informação real p/ Izz/Ir (r e ω_r dominados por ruído/acoplamento)"
        )
    return warnings


# ─── Main ───────────────────────────────────────────────────────────────────────


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )  # console Windows (cp1252) quebra em 'Ī' etc.
    ap = argparse.ArgumentParser(
        description="Identificação conjunta de Ixx, Iyy, Izz, Ir "
        "(equation-error method, sistema empilhado das 3 equações de Euler)"
    )
    ap.add_argument(
        "log", nargs="?", default=None, help="arquivo de log (default: melhor voo)"
    )
    ap.add_argument("--fs", type=float, default=200.0, help="reamostragem [Hz]")
    ap.add_argument("--fc", type=float, default=8.0, help="passa-baixas [Hz]")
    ap.add_argument(
        "--tau-motor",
        type=float,
        default=None,
        help="lag do motor fixo [ms] (default: varre --tau-scan e escolhe pelo R² global)",
    )
    ap.add_argument(
        "--tau-scan",
        default=",".join(str(t) for t in range(20, 105, 5)),
        help="constantes de tempo do motor a testar [ms]",
    )
    ap.add_argument(
        "--rel-err-max",
        type=float,
        default=0.30,
        help="incerteza relativa (erro/valor) acima da qual o parâmetro é marcado como AVISO",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else DEFAULT_LOG
    df = parse_log(log_path)
    if df is None or len(df) < 300:
        sys.exit(f"Log inválido ou com poucas amostras: {log_path}")

    err = check_mixer(df)
    if np.isfinite(err[0]) and max(err[0], err[1], err[2]) > 0.01:
        print(
            f"[aviso] {log_path.name}: mixer não confere exatamente "
            f"(err={err[0]:.3f}/{err[1]:.3f}/{err[2]:.3f}) — possível época de firmware diferente"
        )
    else:
        print(
            f"[ok]   {log_path.name}: mixer confere (mesma época de firmware) | sem clamp: {err[3]*100:.0f}%"
        )

    taus = (
        [args.tau_motor]
        if args.tau_motor is not None
        else [float(x) for x in args.tau_scan.split(",")]
    )

    best = None
    print(
        "\nVarredura do lag do motor (selecionada pelo R² global do sistema empilhado):"
    )
    for tau in taus:
        filt, mask, _ = preprocess(df, args.fs, args.fc, tau)
        if int(mask.sum()) < 200:
            print(
                f"  tau_motor={tau:5.1f} ms -> amostras válidas insuficientes, pulando"
            )
            continue
        X, y, blocks = build_stacked_system(filt, mask)
        theta, stderr, r2, resid, cov = ols(y, X)
        print(f"  tau_motor={tau:5.1f} ms -> R²_global={r2:.4f}  N={int(mask.sum())}")
        if best is None or r2 > best[0]:
            best = (r2, tau, filt, mask, X, y, blocks, theta, stderr, resid, cov)

    if best is None:
        sys.exit("Nenhuma configuração produziu amostras válidas suficientes.")

    r2, tau_best, filt, mask, X, y, (X_x, X_y, X_z), theta, stderr, resid, cov = best
    n_ax = X_x.shape[0]
    n_total = int(mask.sum())

    y_x, y_y, y_z = y[:n_ax], y[n_ax : 2 * n_ax], y[2 * n_ax :]
    r_x, r_y, r_z = resid[:n_ax], resid[n_ax : 2 * n_ax], resid[2 * n_ax :]
    r2_x, r2_y, r2_z = r2_of(y_x, r_x), r2_of(y_y, r_y), r2_of(y_z, r_z)

    Ixx, Iyy, Izz, Ir, c0x, c0y, c0z = theta
    e_Ixx, e_Iyy, e_Izz, e_Ir, e_c0x, e_c0y, e_c0z = stderr
    cond = np.linalg.cond(X.T @ X)

    print("\n" + "=" * 78)
    print(f"REGRESSÃO CONJUNTA (equation-error / OLS empilhado) — {log_path.name}")
    print(
        f"tau_motor={tau_best:.0f} ms | fc={args.fc:.0f} Hz | fs={args.fs:.0f} Hz | "
        f"N={n_total} amostras/eixo ({3*n_total} linhas empilhadas)"
    )
    print("=" * 78)
    print(f"  R² global (sistema empilhado) : {r2:.4f}")
    print(f"  R² eq. x (roll)               : {r2_x:.4f}")
    print(f"  R² eq. y (pitch)              : {r2_y:.4f}")
    print(f"  R² eq. z (yaw)                : {r2_z:.4f}")
    print(f"  número de condição de XᵀX     : {cond:.2e}")
    print("-" * 78)
    rows = [
        ("Ixx [kg·m²]", Ixx, e_Ixx, FW["Ixx"]),
        ("Iyy [kg·m²]", Iyy, e_Iyy, FW["Iyy"]),
        ("Izz [kg·m²]", Izz, e_Izz, FW["Izz"]),
        ("Ir  [kg·m²]", Ir, e_Ir, FW["Ir"]),
    ]
    for name, v, e, ref in rows:
        rel = abs(e / v) if v != 0 else np.inf
        tag = " OK  " if (v > 0 and rel <= args.rel_err_max) else "AVISO"
        print(
            f"  [{tag}] {name}  {v:>12.4e} ± {e:>9.2e} ({rel*100:4.0f}%)   "
            f"(firmware: {ref:>10.3e} -> razão {v/ref:5.2f}x)"
        )
    print("-" * 78)
    print(f"  c0x (intercepto roll)  [N·m] : {c0x:+.3e} ± {e_c0x:.2e}")
    print(f"  c0y (intercepto pitch) [N·m] : {c0y:+.3e} ± {e_c0y:.2e}")
    print(f"  c0z (intercepto yaw)   [N·m] : {c0z:+.3e} ± {e_c0z:.2e}")
    print("=" * 78)

    yaw_ref_std = float(df["yaw_ref"].std())
    phys_warnings = physical_checks(Ixx, Iyy, Izz, Ir, yaw_ref_std)
    if phys_warnings:
        print(
            "\nAVISOS FÍSICOS (não fazem parte do modelo de regressão; checados a posteriori):"
        )
        for w in phys_warnings:
            print(f"  ! {w}")

    print("\nBloco sugerido para src/main.cpp:")
    suffix = (
        ""
        if not phys_warnings
        else "  // AVISO: ver seção de avisos físicos acima, não copiar sem revisão"
    )
    print(f"const float Ixx   = {Ixx*1e6:.2f}e-6f;{suffix if Ixx <= 0 else ''}")
    print(f"const float Iyy   = {Iyy*1e6:.2f}e-6f;{suffix if Iyy <= 0 else ''}")
    print(
        f"const float Izz   = {Izz*1e6:.2f}e-6f;{suffix if (Izz <= 0 or Izz > Ixx+Iyy) else ''}"
    )
    print(f"const float Ir    = {Ir:.3e}f;{suffix if Ir <= 0 else ''}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yhat_x, yhat_y, yhat_z = X_x @ theta, X_y @ theta, X_z @ theta
        t_axis = np.arange(n_total) / args.fs

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        panels = [
            ("roll (τx)", y_x, yhat_x),
            ("pitch (τy)", y_y, yhat_y),
            ("yaw (τz)", y_z, yhat_z),
        ]
        for ax, (label, y_meas, y_mod) in zip(axes, panels):
            ax.plot(t_axis, y_meas, lw=0.8, label="torque reconstruído (medido)")
            ax.plot(t_axis, y_mod, lw=0.8, label="modelo ajustado (OLS conjunto)")
            ax.set_ylabel(f"{label} [N·m]")
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("amostra válida (índice sequencial após máscara)")
        fig.suptitle(
            f"Identificação conjunta (equation-error) — {log_path.name} "
            f"(tau_m={tau_best:.0f} ms, fc={args.fc:.0f} Hz, R²={r2:.3f})"
        )
        fig.tight_layout()
        out = (
            Path(__file__).resolve().parent / "outputs" / "identify_inertia_ols_fit.png"
        )
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"\nGráfico salvo em: {out}")


if __name__ == "__main__":
    main()
