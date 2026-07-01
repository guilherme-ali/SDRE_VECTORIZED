"""
identify_params.py — Identificação de Ixx, Iyy, Izz, Ir e MOTOR_D_COEFF a partir
de logs de voo (telemetria do firmware SDRE_VECTORIZED).

MODELO (equações de Euler de corpo rígido, mesma convenção de
updateSystemMatrix() em src/main.cpp):

    Ixx·dp/dt = (Iyy-Izz)·q·r - Ir·q·ω_r + τ_roll
    Iyy·dq/dt = (Izz-Ixx)·p·r + Ir·p·ω_r + τ_pitch
    Izz·dr/dt = (Ixx-Iyy)·p·q + τ_yaw

TORQUES via matriz de alocação física (a mesma comentada em
lib/utils/utils.cpp::calculateMotorOmegaSq):

    τ_roll  = b·L·(-ω1² - ω2² + ω3² + ω4²)
    τ_pitch = b·L·(+ω1² - ω2² - ω3² + ω4²)
    τ_yaw   =   d·(-ω1² + ω2² - ω3² + ω4²)
    ω_r     = -ω1 + ω2 - ω3 + ω4

NOTAS IMPORTANTES (descobertas na análise dos logs):
 1. O firmware loga w_sq PRÉ-clamp superior, mas JÁ clampado em zero (utils.cpp
    zera ω²<0). Aqui aplicamos também o clamp superior (limite físico do motor)
    e invalidamos amostras saturadas.
 2. A inversa de alocação implementada no firmware usa 1/(2bL); a inversa exata
    da matriz física exigiria 1/(4bL). Resultado: o torque de roll/pitch
    REALMENTE aplicado é 2× o u comandado (yaw está correto). A identificação
    não é afetada (usa w_sq direto), mas isso serve de checagem de época do
    firmware: τ_recon == 2·u_roll exatamente (amostras sem clamp).
 3. Em voo pairado ω_r ≈ 0 e os produtos q·r, p·q são dominados por ruído —
    esses regressores são removidos das equações de roll/pitch/yaw.
 4. Voos com pouca excitação (std de p,q,r baixa) sofrem viés de atenuação em
    malha fechada e dão inércias absurdas: há um gate de qualidade por R².
 5. Izz absoluto: a estimação direta usa os termos cruzados de Coriolis
    (q·r na eq. de roll, p·r na de pitch), validados pela desigualdade
    triangular (|coef| <= 1 exato). Nos logs atuais yaw nunca é comandado
    (yaw_ref == 0 em todos), então r é só ruído acoplado: a rota roll sai do
    limite físico (rejeitada) e a rota pitch é válida porém pouco informativa
    (CI de ±~90%). Um voo com manobras de yaw simultâneas a roll/pitch
    colapsa esse erro. Alternativa: --izz-mode ratio usa a âncora geométrica
    Izz = ratio·(Ixx+Iyy)/2 (default 1.8, validada via Ir, ~12%).

ESTRATÉGIA DE IDENTIFICAÇÃO:
    b (MOTOR_B_COEFF) foi MEDIDO na bancada -> ancora a escala absoluta.
    - Roll : dp = a·τ_roll (+ c·p + c0)  -> Ixx = 1/a   (2 modelos = bracket)
    - Pitch: dq = a·τ_pitch (+ c·q + c0) -> Iyy = 1/a
    - Yaw  : dr = cs·s_yaw + c0
             cs = d/Izz
    - Izz  = ratio·(Ixx+Iyy)/2  (âncora geométrica, validada por Ir)
    - d    = cs·Izz   (corrige MOTOR_D_COEFF)
    - Ir   = valor fixo do firmware (não estimável pela nova eq de yaw)

USO:
    python identify_params.py                        # usa o log padrão (melhor voo)
    python identify_params.py --all                  # varre logs/ inteira
    python identify_params.py logs/a.log logs/b.log  # logs específicos
    python identify_params.py --fc 8 --no-plot
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter

# ─── Parâmetros físicos conhecidos (mesmos do firmware, src/main.cpp) ─────────
B_COEFF = 2.94e-8  # N/(rad/s)^2 — empuxo MEDIDO (hélice 55mm)
L_ARM = 0.060 * 0.70710678  # m — braço efetivo (config X)
MAX_RPM = 26423.0
MAX_OMEGA = MAX_RPM * 2.0 * np.pi / 60.0  # ~2767 rad/s
MAX_OMEGA_SQ = MAX_OMEGA**2  # ~7.656e6 (limite de clamp do firmware)

# Valores atuais do firmware (apenas para comparação no relatório)
FW = dict(Ixx=16.57e-6, Iyy=16.57e-6, Izz=29.80e-6, Ir=1.02e-7, d=0.05 * B_COEFF)

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
DEFAULT_LOG = LOG_DIR / "device-monitor-260602-211127.log"
DEFAULT_LOG = (
    LOG_DIR / "device-monitor-260602-174243.log"
)  # melhor voo (excitação suficiente)
# Logs anteriores a 26-05-15 10:30 usam hélice 45mm / b antigo (git 188677b)
MIN_LOG_STAMP = ("260515", "103000")

# ─── Parsing (mesmo formato do plot_telemetry.py) ─────────────────────────────


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


# ─── Checagem de época do firmware via mixer ──────────────────────────────────


def check_mixer(df: pd.DataFrame):
    """Verifica se o voo foi gravado com as constantes ATUAIS de b, L e d.

    Nas amostras sem clamp em zero, a álgebra da alocação é exata:
        b·L·(-w1-w2+w3+w4) == 2·u_roll   (fator 2 = bug da inversa, ver nota 2)
        b·L·(+w1-w2-w3+w4) == 2·u_pitch
        d·(-w1+w2-w3+w4)   == 1·u_yaw
    Erro ~0 => mesmo firmware; erro grande => época diferente (descartar log).
    """
    w = df[["w1_sq", "w2_sq", "w3_sq", "w4_sq"]].to_numpy()
    free = (w > 1.0).all(axis=1)
    if free.sum() < 50:
        return (np.nan, np.nan, np.nan, float(free.mean()))
    w = w[free]
    tau_r = B_COEFF * L_ARM * (-w[:, 0] - w[:, 1] + w[:, 2] + w[:, 3])
    tau_p = B_COEFF * L_ARM * (+w[:, 0] - w[:, 1] - w[:, 2] + w[:, 3])
    tau_y = FW["d"] * (-w[:, 0] + w[:, 1] - w[:, 2] + w[:, 3])

    def rel_err(a, b_):
        s = np.std(b_)
        return np.nan if s == 0 else float(np.std(a - b_) / s)

    return (
        rel_err(tau_r, 2.0 * df["u_roll"].to_numpy()[free]),
        rel_err(tau_p, 2.0 * df["u_pitch"].to_numpy()[free]),
        rel_err(tau_y, 1.0 * df["u_yaw"].to_numpy()[free]),
        float(free.mean()),
    )


# ─── Pré-processamento de um log ──────────────────────────────────────────────


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
    # Saturação relevante (inclui clamp em zero): dinâmica de atuação desconhecida
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
    tau_roll = B_COEFF * L_ARM * (-sig["w1s"] - sig["w2s"] + sig["w3s"] + sig["w4s"])
    tau_pitch = B_COEFF * L_ARM * (+sig["w1s"] - sig["w2s"] - sig["w3s"] + sig["w4s"])
    s_yaw = -sig["w1s"] + sig["w2s"] - sig["w3s"] + sig["w4s"]  # τ_yaw = d·s_yaw

    # Passa-baixas zero-phase IDÊNTICO em ambos os lados da regressão
    b_f, a_f = butter(4, fc / (fs / 2.0))

    # Produtos não-lineares (q·r, p·r) formados com taxas JÁ band-limitadas:
    # vibração de alta frequência em q e r se retifica no produto (gera conteúdo
    # DC espúrio correlacionado com a atividade dos motores) se formados crus.
    pb = filtfilt(b_f, a_f, sig["p"])
    qb = filtfilt(b_f, a_f, sig["q"])
    rb = filtfilt(b_f, a_f, sig["r"])

    raw = dict(
        p=sig["p"],
        q=sig["q"],
        r=sig["r"],
        qr=qb * rb,
        pr=pb * rb,
        q_omega_r=qb * omega_r,
        p_omega_r=pb * omega_r,
        tau_roll=tau_roll,
        tau_pitch=tau_pitch,
        s_yaw=s_yaw,
        omega_r=omega_r,
    )
    filt = {k: filtfilt(b_f, a_f, v) for k, v in raw.items()}

    dt = 1.0 / fs
    filt["dp"] = np.gradient(filt["p"], dt)
    filt["dq"] = np.gradient(filt["q"], dt)
    filt["dr"] = np.gradient(filt["r"], dt)
    filt["domega_r"] = np.gradient(filt["omega_r"], dt)
    filt["one"] = np.ones_like(filt["p"])

    edge = int(0.25 * fs)
    mask[:edge] = False
    mask[-edge:] = False
    return filt, mask, tu


# ─── Mínimos quadrados com erros-padrão ───────────────────────────────────────


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
    return theta, stderr, r2, resid


def fit_axes(filt: dict, mask: np.ndarray):
    """Regressões dos 3 eixos.

    Roll/pitch em DOIS modelos: só torque, e torque + termo ∝ taxa. O termo de
    taxa absorve o erro de fase do modelo de lag na frequência de oscilação
    dominante (não é amortecimento físico); os dois resultados delimitam a
    inércia (bracket de incerteza estrutural).
    """
    m = mask
    out = {}
    for ax, dy, tq, rate in [
        ("roll", "dp", "tau_roll", "p"),
        ("pitch", "dq", "tau_pitch", "q"),
    ]:
        X1 = np.column_stack([filt[tq][m], filt["one"][m]])
        X2 = np.column_stack([filt[tq][m], filt[rate][m], filt["one"][m]])
        out[ax] = (ols(filt[dy][m], X1), ols(filt[dy][m], X2))
    Xy = np.column_stack([filt["s_yaw"][m], filt["one"][m]])
    out["yaw"] = ols(filt["dr"][m], Xy)
    return out


def izz_from_cross(cat: dict, mask: np.ndarray, Ixx: float, Iyy: float):
    """Estimação direta de Izz pelos termos cruzados de Coriolis.

    Roll : dp = a_tau·τ_roll  + a_p·p + a_qr·(q·r),  a_qr = (Iyy-Izz)/Ixx
    Pitch: dq = a_tau·τ_pitch + a_q·q + b_pr·(p·r),  b_pr = (Izz-Ixx)/Iyy

    VALIDAÇÃO FÍSICA: pela desigualdade triangular do tensor de inércia
    (|Ixx-Iyy| <= Izz <= Ixx+Iyy) vale EXATAMENTE |a_qr| <= 1 e |b_pr| <= 1.
    Coeficiente fora do limite (além de 2σ) => estimativa dominada por viés de
    malha fechada (r sem excitação externa) e é rejeitada.
    """
    m = mask
    Xr = np.column_stack([cat["tau_roll"][m], cat["p"][m], cat["qr"][m]])
    ta, sa, _, _ = ols(cat["dp"][m], Xr)
    Xp = np.column_stack([cat["tau_pitch"][m], cat["q"][m], cat["pr"][m]])
    tb, sb, _, _ = ols(cat["dq"][m], Xp)

    a_qr, a_err = ta[2], sa[2]
    b_pr, b_err = tb[2], sb[2]
    cands = []
    # Izz = Iyy - a_qr·Ixx  (rota roll) ; Izz = Ixx + b_pr·Iyy (rota pitch)
    for coef, err, izz, ierr, rota in [
        (a_qr, a_err, Iyy - a_qr * Ixx, a_err * Ixx, "roll (q·r)"),
        (b_pr, b_err, Ixx + b_pr * Iyy, b_err * Iyy, "pitch (p·r)"),
    ]:
        valid = abs(coef) - 2.0 * err <= 1.0
        cands.append(
            dict(rota=rota, coef=coef, err=err, izz=izz, izz_err=ierr, valid=valid)
        )

    ok = [c for c in cands if c["valid"]]
    if ok:
        w = [1.0 / max(c["izz_err"], 1e-12) ** 2 for c in ok]
        izz = sum(wi * c["izz"] for wi, c in zip(w, ok)) / sum(w)
        izz_err = float(np.sqrt(1.0 / sum(w)))
    else:
        izz, izz_err = np.nan, np.nan
    return izz, izz_err, cands


def ir_from_cross(cat: dict, mask: np.ndarray, Ixx: float, Iyy: float):
    """Estimação direta de Ir pelos termos p·omega_r e q·omega_r.
    Roll : dp = a_tau·τ_roll  + a_p·p + a_qw·(q·omega_r)
           a_qw = -Ir/Ixx  -> Ir = -a_qw * Ixx
    Pitch: dq = a_tau·τ_pitch + a_q·q + b_pw·(p·omega_r)
           b_pw = Ir/Iyy  -> Ir = b_pw * Iyy
    """
    m = mask
    Xr = np.column_stack([cat["tau_roll"][m], cat["p"][m], cat["q_omega_r"][m], cat["one"][m]])
    ta, sa, _, _ = ols(cat["dp"][m], Xr)
    
    Xp = np.column_stack([cat["tau_pitch"][m], cat["q"][m], cat["p_omega_r"][m], cat["one"][m]])
    tb, sb, _, _ = ols(cat["dq"][m], Xp)

    a_qw, a_err = ta[2], sa[2]
    b_pw, b_err = tb[2], sb[2]
    
    ir_roll = -a_qw * Ixx
    ir_roll_err = a_err * Ixx
    
    ir_pitch = b_pw * Iyy
    ir_pitch_err = b_err * Iyy
    
    return [
        dict(rota="roll (q·wr)", coef=a_qw, err=a_err, ir=ir_roll, ir_err=ir_roll_err),
        dict(rota="pitch (p·wr)", coef=b_pw, err=b_err, ir=ir_pitch, ir_err=ir_pitch_err)
    ]


def axis_reliability(r2_torque, I, I_err, r2_min, rel_err_max):
    """Critério de confiabilidade de Ixx/Iyy: R² mínimo do modelo só-torque e
    incerteza relativa (já inclui o bracket dos 2 modelos, via axis_inertia)."""
    if r2_torque < r2_min:
        return False, f"R²={r2_torque:.2f} < {r2_min:.2f} (pouca excitação)"
    rel = I_err / I
    if rel > rel_err_max:
        return False, f"incerteza relativa {rel*100:.0f}% > {rel_err_max*100:.0f}%"
    return True, ""


def ir_reliability(ir_cross, rel_err_max):
    """Critério de confiabilidade de Ir via termos cruzados de roll/pitch: as duas
    rotas devem ser fisicamente válidas (Ir > 0 a 2σ) E concordar entre si dentro
    do erro combinado. Retorna (ok, ir_estimado, ir_err_estimado, motivo)."""
    valid = [c for c in ir_cross if c["ir"] - 2.0 * c["ir_err"] > 0.0]
    if len(valid) < len(ir_cross):
        bad = ", ".join(c["rota"] for c in ir_cross if c not in valid)
        return False, None, None, f"rota(s) rejeitada(s) (Ir<=0 a 2σ): {bad}"
    ir_a, ir_b = valid[0]["ir"], valid[1]["ir"]
    err_a, err_b = valid[0]["ir_err"], valid[1]["ir_err"]
    if abs(ir_a - ir_b) > 2.0 * (err_a + err_b):
        return (
            False, None, None,
            f"rotas discordam entre si ({valid[0]['rota']}={ir_a:.2e} vs "
            f"{valid[1]['rota']}={ir_b:.2e}) — provável falta de excitação de yaw",
        )
    w_a, w_b = 1.0 / err_a**2, 1.0 / err_b**2
    ir_est = (w_a * ir_a + w_b * ir_b) / (w_a + w_b)
    ir_est_err = (w_a + w_b) ** -0.5
    rel = ir_est_err / ir_est
    if rel > rel_err_max:
        return (
            False, ir_est, ir_est_err,
            f"incerteza relativa {rel*100:.0f}% > {rel_err_max*100:.0f}%",
        )
    return True, ir_est, ir_est_err, ""


def axis_inertia(fit_pair):
    """(I_modelo1, I_modelo2, err_combinado, R2_1, R2_2) de um eixo roll/pitch."""
    (t1, s1, r21, _), (t2, s2, r22, _) = fit_pair
    I1, I2 = 1.0 / t1[0], 1.0 / t2[0]
    err_stat = max(s1[0] / t1[0] ** 2, s2[0] / t2[0] ** 2)
    err = abs(I1 - I2) / 2.0 + err_stat
    return I1, I2, err, r21, r22


# ─── Main ─────────────────────────────────────────────────────────────────────


def collect_logs(args) -> list[Path]:
    if args.logs:
        return [Path(p) for p in args.logs]
    if not args.all:
        return [DEFAULT_LOG]
    out = []
    for p in sorted(LOG_DIR.glob("device-monitor-*.log")):
        parts = p.stem.split("-")
        if len(parts) < 4:
            continue
        date, hhmm = parts[2], parts[3]
        if (date, hhmm.ljust(6, "0")) >= MIN_LOG_STAMP:
            out.append(p)
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) quebra em 'Ī' etc.
    ap = argparse.ArgumentParser(description="Identificação de Ixx, Iyy, Izz, Ir e d")
    ap.add_argument("logs", nargs="*", default=None, help="arquivos de log")
    ap.add_argument(
        "--all",
        action="store_true",
        help="varre toda a pasta logs/ (a partir de 26-05-15)",
    )
    ap.add_argument("--fs", type=float, default=200.0, help="reamostragem [Hz]")
    ap.add_argument("--fc", type=float, default=8.0, help="passa-baixas [Hz]")
    ap.add_argument(
        "--tau-scan",
        default=",".join(str(t) for t in range(20, 105, 5)),
        help="constantes de tempo do motor a testar [ms]",
    )
    ap.add_argument(
        "--izz-mode",
        choices=["auto", "cross", "ratio"],
        default="auto",
        help="cross: estima Izz dos termos q·r/p·r (requer voo com "
        "excitação de yaw); ratio: âncora geométrica; "
        "auto: cross se passar na validação física, senão ratio",
    )
    ap.add_argument(
        "--izz-ratio",
        type=float,
        default=1.8,
        help="âncora geométrica Izz = ratio·(Ixx+Iyy)/2 (modo ratio)",
    )
    ap.add_argument(
        "--r2-min",
        type=float,
        default=0.40,
        help="gate de qualidade: R² mínimo do fit de roll (só torque)",
    )
    ap.add_argument(
        "--rel-err-max",
        type=float,
        default=0.30,
        help="incerteza relativa (erro/valor) máxima para considerar Ixx/Iyy/Izz/Ir "
        "confiáveis; acima disso o valor entra na seção AVISOS do relatório",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # ── Carrega, deduplica e filtra por época ──
    frames, seen = [], set()
    for p in collect_logs(args):
        try:
            df = parse_log(p)
        except Exception:
            continue
        if df is None or len(df) < 300:
            continue
        h = hashlib.md5(df.head(20).to_csv().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        err = check_mixer(df)
        if not np.isfinite(err[0]) or max(err[0], err[1], err[2]) > 0.01:
            print(
                f"[skip] {p.name}: mixer não confere (época de firmware diferente "
                f"ou clamp excessivo) err={err[0]:.3f}/{err[1]:.3f}/{err[2]:.3f}"
            )
            continue
        frames.append((p.name, df))
        print(
            f"[ok]   {p.name}: {len(df)} amostras | sem clamp: {err[3]*100:.0f}% | "
            f"mixer exato (mesma época de firmware)"
        )
    if not frames:
        sys.exit("Nenhum log valido.")

    # ── Gate de qualidade por voo (excitação suficiente) ──
    kept = []
    for name, df in frames:
        filt, mask, _ = preprocess(df, args.fs, args.fc, 60.0)
        if mask.sum() < 200:
            print(f"[skip] {name}: só {int(mask.sum())} amostras válidas")
            continue
        fits = fit_axes(filt, mask)
        r2 = fits["roll"][0][2]
        if r2 < args.r2_min:
            print(
                f"[skip] {name}: R²_roll={r2:.2f} < {args.r2_min} "
                f"(pouca excitação -> viés de malha fechada)"
            )
            continue
        kept.append((name, df))
    if not kept:
        sys.exit("Nenhum voo passou no gate de qualidade.")
    print(f"\nVoos usados: {', '.join(n for n, _ in kept)}")

    # yaw nunca comandado -> ω_r e r não têm sinal genuíno (só ruído/vazamento não-linear):
    # causa raiz recorrente de Izz (rota cruzada) e Ir não serem identificáveis.
    yaw_ref_std_max = max(float(df["yaw_ref"].std()) for _, df in kept)
    yaw_excited = yaw_ref_std_max > 1.0

    # ── Varredura da constante de tempo do motor (dados agrupados) ──
    # Demeaning POR VOO (Frisch-Waugh): cada voo tem trim/offset de CG próprio;
    # um intercepto único compartilhado distorceria os coeficientes agrupados.
    taus = [float(x) for x in args.tau_scan.split(",")]
    best = None
    print()
    for tau in taus:
        filts, masks = [], []
        for _, df in kept:
            f, m, _ = preprocess(df, args.fs, args.fc, tau)
            for k, v in f.items():
                if k != "one" and m.sum() > 0:
                    f[k] = v - v[m].mean()
            filts.append(f)
            masks.append(m)
        cat = {k: np.concatenate([f[k] for f in filts]) for k in filts[0]}
        mask = np.concatenate(masks)
        fits = fit_axes(cat, mask)
        score = fits["roll"][0][2] + fits["pitch"][0][2]
        print(
            f"  tau_motor={tau:5.1f} ms -> R² roll={fits['roll'][0][2]:.3f} "
            f"pitch={fits['pitch'][0][2]:.3f} yaw={fits['yaw'][2]:.3f}"
        )
        if best is None or score > best[0]:
            best = (score, tau, fits, cat, mask)

    _, tau_best, fits, cat, mask = best
    n_used = int(mask.sum())

    # ── Conversão para parâmetros físicos ──
    Ixx1, Ixx2, Ixx_err, r2x1, r2x2 = axis_inertia(fits["roll"])
    Iyy1, Iyy2, Iyy_err, r2y1, r2y2 = axis_inertia(fits["pitch"])
    Ixx, Iyy = (Ixx1 + Ixx2) / 2.0, (Iyy1 + Iyy2) / 2.0
    p_std = float(cat["p"][mask].std())
    q_std = float(cat["q"][mask].std())
    Ixx_ok, Ixx_reason = axis_reliability(r2x1, Ixx, Ixx_err, args.r2_min, args.rel_err_max)
    Iyy_ok, Iyy_reason = axis_reliability(r2y1, Iyy, Iyy_err, args.r2_min, args.rel_err_max)
    if not Ixx_ok:
        Ixx_reason += f" [excitação medida: std(p)={p_std:.2f} rad/s, N={n_used} amostras]"
    if not Iyy_ok:
        Iyy_reason += f" [excitação medida: std(q)={q_std:.2f} rad/s, N={n_used} amostras]"

    # ── Izz: estimação direta (termos cruzados) e/ou âncora geométrica ──
    izz_x, izz_x_err, cross = izz_from_cross(cat, mask, Ixx, Iyy)
    print("\nEstimação direta de Izz (termos cruzados de Coriolis):")
    for c in cross:
        status = (
            "VÁLIDO" if c["valid"] else "REJEITADO (|coef| > 1: viés de malha fechada)"
        )
        print(
            f"  rota {c['rota']:12s}: coef = {c['coef']:+8.3f} ± {c['err']:.3f} "
            f"(limite físico |coef|<=1) -> Izz = {c['izz']*1e6:8.1f}e-6  [{status}]"
        )

    use_cross = (args.izz_mode == "cross") or (
        args.izz_mode == "auto" and np.isfinite(izz_x)
    )
    if use_cross and np.isfinite(izz_x):
        Izz, Izz_err = izz_x, izz_x_err
        izz_src = "termos cruzados (estimado dos dados)"
        Izz_ok = (Izz_err / Izz) <= args.rel_err_max
        Izz_reason = (
            ""
            if Izz_ok
            else f"incerteza relativa {Izz_err/Izz*100:.0f}% > {args.rel_err_max*100:.0f}% "
            "(rota cruzada pouco informativa)"
        )
    else:
        if args.izz_mode == "cross":
            print(
                "  [aviso] modo cross forçado mas ambas as rotas foram rejeitadas;"
                " caindo para âncora geométrica."
            )
        Izz = args.izz_ratio * (Ixx + Iyy) / 2.0
        Izz_err = args.izz_ratio * (Ixx_err + Iyy_err) / 2.0 + 0.1 * Izz
        izz_src = f"âncora geométrica {args.izz_ratio}·(Ixx+Iyy)/2 (NÃO estimado)"
        Izz_ok = False
        if args.izz_mode == "ratio":
            Izz_reason = "--izz-mode ratio forçado — não estimado dos dados por escolha"
        else:
            rejected = ", ".join(c["rota"] for c in cross if not c["valid"]) or "ambas"
            Izz_reason = (
                f"rota(s) cruzada(s) rejeitada(s) ({rejected}: |coef|>1, "
                "viés de malha fechada)"
            )
            if not yaw_excited:
                Izz_reason += (
                    f"; causa provável: yaw não comandado neste(s) voo(s) "
                    f"(std(yaw_ref)={yaw_ref_std_max:.2f}°)"
                )
    print(f"  -> Izz adotado: {Izz*1e6:.1f}e-6 via {izz_src}")
    print(
        f"  Limite físico superior (des. triangular): Izz <= Ixx+Iyy = "
        f"{(Ixx+Iyy)*1e6:.1f}e-6"
    )

    ty, sy, r2yaw, _ = fits["yaw"]
    d_over_Izz = ty[0]

    d = d_over_Izz * Izz
    d_err = abs(sy[0] * Izz) + abs(d_over_Izz * Izz_err)
    Izz_via_Ir_fw = np.nan

    # A equação de yaw tem coeficiente 0 no termo de inércia do rotor (a rota antiga
    # -Ir/Izz·dω_r/dt foi removida por não ter respaldo físico nessa equação). A rota
    # alternativa usa os termos giroscópicos cruzados de roll/pitch, onde Ir de fato
    # aparece no modelo; só é aceita se as duas rotas forem fisicamente válidas
    # (Ir>0 a 2σ) e concordarem entre si — senão caímos no valor fixo do firmware.
    ir_cross = ir_from_cross(cat, mask, Ixx, Iyy)
    print("\nEstimação de Ir via Roll/Pitch (efeito giroscópico dos motores):")
    for c in ir_cross:
        print(f"  rota {c['rota']:12s}: coef = {c['coef']:+8.3e} ± {c['err']:.3e} "
              f"-> Ir = {c['ir']:8.2e} ± {c['ir_err']:.2e} (ref: {FW['Ir']:.2e})")

    Ir_ok, ir_est, ir_est_err, Ir_reason = ir_reliability(ir_cross, args.rel_err_max)
    if Ir_ok:
        Ir, Ir_err = ir_est, ir_est_err
    else:
        Ir, Ir_err = FW["Ir"], 0.0
        if not yaw_excited:
            Ir_reason += (
                f"; causa provável: sem excitação de yaw/ω_r neste(s) voo(s) "
                f"(std(yaw_ref)={yaw_ref_std_max:.2f}°)"
            )
    Ir_over_Izz = Ir / Izz


    print("\n" + "=" * 74)
    print(
        f"RESULTADOS  (tau_motor={tau_best:.0f} ms | fc={args.fc:.0f} Hz | "
        f"{n_used} amostras de {len(kept)} voo(s))"
    )
    print("=" * 74)
    print(f"  R² roll  : {r2x1:.3f} (só torque) / {r2x2:.3f} (torque+taxa)")
    print(f"  R² pitch : {r2y1:.3f} (só torque) / {r2y2:.3f} (torque+taxa)")
    print(f"  R² yaw   : {r2yaw:.3f}")
    print("-" * 74)
    rows = [
        ("Ixx [kg·m²]", Ixx, Ixx_err, FW["Ixx"], Ixx_ok, Ixx_reason),
        ("Iyy [kg·m²]", Iyy, Iyy_err, FW["Iyy"], Iyy_ok, Iyy_reason),
        ("Izz [kg·m²]", Izz, Izz_err, FW["Izz"], Izz_ok, Izz_reason),
        ("Ir  [kg·m²]", Ir, Ir_err, FW["Ir"], Ir_ok, Ir_reason),
    ]
    avisos = []
    for name, v, e, ref, ok, reason in rows:
        tag = " OK  " if ok else "AVISO"
        print(
            f"  [{tag}] {name}  {v:>12.4e} ± {e:>9.2e}   "
            f"(firmware: {ref:>10.3e}  -> razão {v/ref:5.2f}x)"
        )
        if not ok:
            avisos.append(f"{name.strip()}: {reason}")
    print(
        f"  [ -- ] d [N·m·s²]   {d:>12.4e} ± {d_err:>9.2e}   "
        f"(firmware: {FW['d']:>10.3e}  -> razão {d/FW['d']:5.2f}x)"
    )
    if avisos:
        print("-" * 74)
        print("  AVISOS — log insuficiente p/ estes valores (NÃO usar sem revisão):")
        for a in avisos:
            print(f"    ! {a}")
    print("-" * 74)
    print(
        f"  Bracket Ixx: [{min(Ixx1,Ixx2):.3e}, {max(Ixx1,Ixx2):.3e}]  "
        f"Iyy: [{min(Iyy1,Iyy2):.3e}, {max(Iyy1,Iyy2):.3e}]"
    )
    print(f"  d/Izz  = {d_over_Izz:+.3e} ± {sy[0]:.1e}  [1/(m·s²)·...]")
    print(f"  d/b    = {d/B_COEFF:.4f} m   (firmware usa 0.05)")
    Izz_geo = args.izz_ratio * (Ixx + Iyy) / 2.0
    print(
        f"  Cross-checks de Izz: geométrico ({args.izz_ratio}·Ī) = {Izz_geo:.3e} | "
        f"via Ir do firmware ({FW['Ir']:.2e}) = {Izz_via_Ir_fw:.3e}"
    )
    print("=" * 74)

    def aviso_suffix(ok):
        return "" if ok else "  // AVISO: ver secao AVISOS acima, nao copiar sem revisao"

    print("\nBloco sugerido para src/main.cpp:")
    print(f"const float Ixx   = {Ixx*1e6:.2f}e-6f;{aviso_suffix(Ixx_ok)}")
    print(f"const float Iyy   = {Iyy*1e6:.2f}e-6f;{aviso_suffix(Iyy_ok)}")
    print(f"const float Izz   = {Izz*1e6:.2f}e-6f;{aviso_suffix(Izz_ok)}")
    print(f"const float Ir    = {Ir:.3e}f;{aviso_suffix(Ir_ok)}")
    print(f"const float MOTOR_D_COEFF = {d/B_COEFF:.4f}f * MOTOR_B_COEFF;")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = mask
        t = np.arange(len(m)) / args.fs
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        panels = [
            ("roll", "dp", fits["roll"][0][0], ["tau_roll", "one"]),
            ("pitch", "dq", fits["pitch"][0][0], ["tau_pitch", "one"]),
            ("yaw", "dr", fits["yaw"][0], ["s_yaw", "one"]),
        ]
        for ax, (axis, ykey, theta, xkeys) in zip(axes, panels):
            X = np.column_stack([cat[k] for k in xkeys])
            yhat = X @ theta
            y = cat[ykey].copy()
            yh = yhat.copy()
            y[~m] = np.nan
            yh[~m] = np.nan
            ax.plot(t, y, lw=0.8, label=f"d{ykey[1]}/dt medido (filtrado)")
            ax.plot(t, yh, lw=0.8, label="modelo ajustado")
            ax.set_ylabel(f"{axis} [rad/s²]")
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("t [s] (voos concatenados)")
        fig.suptitle(
            f"Identificação — medido vs. modelo (tau_m={tau_best:.0f} ms, "
            f"fc={args.fc:.0f} Hz)"
        )
        fig.tight_layout()
        out = Path(__file__).resolve().parent / "outputs" / "identify_params_fit.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"\nGráfico salvo em: {out}")


if __name__ == "__main__":
    main()
