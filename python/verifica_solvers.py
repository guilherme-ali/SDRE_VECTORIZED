"""
Fase 3 da auditoria dos solvers de Riccati (lib/AUTOLQR): espelho em float64
de cada variante EXATAMENTE como o C++ implementa, para separar bug de
fórmula (persiste em float64) de erro de precisão (só aparece em float32).

Lê os casos exportados por test/native/verify_solvers.cpp em outputs/cases/
(mesmas matrizes A,B,Q,R que alimentaram o harness nativo — garante que
Python e C++ estão auditando exatamente os mesmos números) e:

  1. resolve cada caso com o oráculo scipy.linalg.solve_discrete_are;
  2. roda a transcrição float64 de SDA_SS, ASDA e SDA_SCALED *como estão hoje*
     em lib/AUTOLQR/AutoLQR.cpp, mostrando que o resíduo da DARE continua
     grande mesmo sem nenhum erro de arredondamento float32;
  3. roda a mesma transcrição *com a correção planejada* (Fase 4.2/4.3/4.4),
     mostrando que o resíduo cai para a ordem de grandeza do SDA de referência;
  4. prova numericamente, em float64, que a forma V/W do ADDA (AutoLQR.cpp:
     2096-2140) é idêntica ao SDA (identidade push-through
     (I+HG)^-1 H = H(I+GH)^-1), comparando P_ADDA e P_SDA ponto a ponto.

Ver plano: C:\\Users\\guilh\\.claude\\plans\\nesse-repositorio-ha-diversos-tingly-hopcroft.md
"""

import csv
import glob
import os
import numpy as np
from scipy.linalg import solve_discrete_are

CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "cases")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "outputs", "verify_float64_mirror.csv")

# Casos que contam a história completa da auditoria (ver plano, Fase 2):
# C1 = ponto de operação real; C5/C6c = expõe SDA_Scaled; C6a = expõe SDA-SS;
# C6b = expõe ASDA. Não precisamos varrer os 228 casos aqui — o host já fez
# isso; aqui o objetivo é isolar álgebra de precisão nesses representativos.
STORY_CASES = ["C1_hover", "C5_unequal_scale", "C6a_scalar_ss", "C6b_scalar_asda", "C6c_2x2_scaled"]


def read_case(name):
    path = os.path.join(CASES_DIR, f"{name}.csv")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    n = int(rows[0][1])
    m = int(rows[1][1])
    idx = 2
    mats = {}
    label = None
    buf = []
    for row in rows[idx:]:
        if len(row) == 1 and row[0] in ("A", "B", "Q", "R"):
            if label is not None:
                mats[label] = np.array(buf, dtype=np.float64)
            label = row[0]
            buf = []
        else:
            buf.append([float(x) for x in row])
    if label is not None:
        mats[label] = np.array(buf, dtype=np.float64)
    return n, m, mats["A"], mats["B"], mats["Q"], mats["R"]


def dare_residual(A, B, Q, R, P):
    S = R + B.T @ P @ B
    resid = A.T @ P @ A - P - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A) + Q
    qn = max(np.linalg.norm(Q), 1e-30)
    return np.linalg.norm(resid) / qn


def sym(M):
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# SDA de referência — box de Chu, Fan, Lin & Wang (2004), Int. J. Control
# 77(8):767-788, p.770. Idêntico a computeGainMatrixSDA() em AutoLQR.cpp:357-553.
# ---------------------------------------------------------------------------
def sda_reference(A, B, Q, R, tol=1e-12, max_iter=100):
    n = A.shape[0]
    Ak = A.copy()
    Gk = B @ np.linalg.solve(R, B.T)
    Hk = Q.copy()
    for it in range(1, max_iter + 1):
        W = np.eye(n) + Gk @ Hk
        Wi = np.linalg.inv(W)
        A_next = Ak @ Wi @ Ak
        G_next = Gk + Ak @ Wi @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ Wi @ Ak
        diff = np.linalg.norm(H_next - Hk) / max(np.linalg.norm(H_next), 1e-30)
        Ak, Gk, Hk = A_next, sym(G_next), sym(H_next)
        if diff < tol:
            return Hk, it
    return Hk, max_iter


# ---------------------------------------------------------------------------
# SDA-SS *como implementado* — AutoLQR.cpp:1416-1478. Shift afim isolado em A,
# H0 sem fator de shift. gamma=0.3 hardcoded (linha 1392).
# ---------------------------------------------------------------------------
def sda_ss_as_implemented(A, B, Q, R, gamma=0.3, tol=1e-12, max_iter=100):
    n = A.shape[0]
    Ak = (A - gamma * np.eye(n)) / (1 - gamma)
    Gk = (B @ np.linalg.solve(R, B.T)) / (1 - gamma) ** 2
    Hk = Q.copy()
    for it in range(1, max_iter + 1):
        W = np.eye(n) + Gk @ Hk
        Wi = np.linalg.inv(W)
        A_next = Ak @ Wi @ Ak
        G_next = Gk + Ak @ Wi @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ Wi @ Ak
        diff = np.linalg.norm(H_next - Hk) / max(np.linalg.norm(H_next), 1e-30)
        Ak, Gk, Hk = A_next, sym(G_next), sym(H_next)
        if diff < tol:
            return Hk, it
    return Hk, max_iter


# ---------------------------------------------------------------------------
# ASDA *como implementado* — AutoLQR.cpp:1608-1771. beta^2 só em G na
# inicialização (1647-1649), sem dividir H; P = H_k direto (1746), sem
# reverter o produto acumulado dos s_i (1666-1675).
# ---------------------------------------------------------------------------
def asda_as_implemented(A, B, Q, R, tol=1e-12, max_iter=100):
    n = A.shape[0]
    Ak = A.copy()
    Gk = B @ np.linalg.solve(R, B.T)
    Hk = Q.copy()

    normA = np.linalg.norm(Ak)
    normG = np.linalg.norm(Gk)
    normH = np.linalg.norm(Hk)
    _ = 1.0 / max(normA, 1e-30)  # alpha_k: calculado e nunca usado no C++ (linha 1639-1641)
    beta_k = np.sqrt(normH / max(normG, 1e-30))
    Gk = (beta_k ** 2) * Gk  # BUG: só G recebe beta^2; H não é dividido

    for it in range(1, max_iter + 1):
        s = np.sqrt(np.linalg.norm(Hk) / max(np.linalg.norm(Gk), 1e-30))
        s = min(max(s, 0.1), 10.0)
        Gk = s * Gk
        Hk = Hk / s

        W = np.eye(n) + Gk @ Hk
        Wi = np.linalg.inv(W)
        A_next = Ak @ Wi @ Ak
        G_next = Gk + Ak @ Wi @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ Wi @ Ak
        diff = np.linalg.norm(H_next - Hk) / max(np.linalg.norm(H_next), 1e-30)
        Ak, Gk, Hk = A_next, sym(G_next), sym(H_next)
        if diff < tol:
            return Hk, it  # BUG: falta multiplicar por cum_s aqui
    return Hk, max_iter


# ---------------------------------------------------------------------------
# ASDA *corrigido* (Fase 4.2 do plano): eq.(34) aplicada como escrita
# (G*=s0, H/=s0 também na inicialização), acumula prod(s_i), e devolve
# P = H_k * prod(s_i).
# ---------------------------------------------------------------------------
def asda_fixed(A, B, Q, R, tol=1e-12, max_iter=100):
    n = A.shape[0]
    Ak = A.copy()
    Gk = B @ np.linalg.solve(R, B.T)
    Hk = Q.copy()

    normG = np.linalg.norm(Gk)
    normH = np.linalg.norm(Hk)
    beta_k = np.sqrt(normH / max(normG, 1e-30))
    beta_k = min(max(beta_k, 0.1), 10.0)
    Gk = beta_k * Gk
    Hk = Hk / beta_k
    cum_s = beta_k

    for it in range(1, max_iter + 1):
        s = np.sqrt(np.linalg.norm(Hk) / max(np.linalg.norm(Gk), 1e-30))
        s = min(max(s, 0.1), 10.0)
        Gk = s * Gk
        Hk = Hk / s
        cum_s *= s

        W = np.eye(n) + Gk @ Hk
        Wi = np.linalg.inv(W)
        A_next = Ak @ Wi @ Ak
        G_next = Gk + Ak @ Wi @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ Wi @ Ak
        diff = np.linalg.norm(H_next - Hk) / max(np.linalg.norm(H_next), 1e-30)
        Ak, Gk, Hk = A_next, sym(G_next), sym(H_next)
        if diff < tol:
            return Hk * cum_s, it
    return Hk * cum_s, max_iter


# ---------------------------------------------------------------------------
# SDA_Scaled *como implementado* — AutoLQR.cpp:1826-1990. H0 = D Q D (deveria
# ser D^-1 Q D^-1) e P = D^-1 P_hat D^-1 (deveria ser D P_hat D).
# ---------------------------------------------------------------------------
def sda_scaled_as_implemented(A, B, Q, R, tol=1e-12, max_iter=100):
    n = A.shape[0]
    d = 1.0 / np.sqrt(np.maximum(np.linalg.norm(A, axis=1), 1e-30))
    D = np.diag(d)
    Dinv = np.diag(1.0 / d)

    Ak = D @ A @ Dinv
    Gk = D @ (B @ np.linalg.solve(R, B.T)) @ D
    Hk = D @ Q @ D  # BUG: deveria ser Dinv @ Q @ Dinv

    for it in range(1, max_iter + 1):
        W = np.eye(n) + Gk @ Hk
        Wi = np.linalg.inv(W)
        A_next = Ak @ Wi @ Ak
        G_next = Gk + Ak @ Wi @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ Wi @ Ak
        diff = np.linalg.norm(H_next - Hk) / max(np.linalg.norm(H_next), 1e-30)
        Ak, Gk, Hk = A_next, sym(G_next), sym(H_next)
        if diff < tol:
            return Dinv @ Hk @ Dinv, it  # BUG: deveria ser D @ Hk @ D
    return Dinv @ Hk @ Dinv, max_iter


# ---------------------------------------------------------------------------
# SDA_Scaled *corrigido* (Fase 4.3 do plano): H0 = D^-1 Q D^-1, P = D P_hat D.
# ---------------------------------------------------------------------------
def sda_scaled_fixed(A, B, Q, R, tol=1e-12, max_iter=100):
    n = A.shape[0]
    d = 1.0 / np.sqrt(np.maximum(np.linalg.norm(A, axis=1), 1e-30))
    D = np.diag(d)
    Dinv = np.diag(1.0 / d)

    Ak = D @ A @ Dinv
    Gk = D @ (B @ np.linalg.solve(R, B.T)) @ D
    Hk = Dinv @ Q @ Dinv

    for it in range(1, max_iter + 1):
        W = np.eye(n) + Gk @ Hk
        Wi = np.linalg.inv(W)
        A_next = Ak @ Wi @ Ak
        G_next = Gk + Ak @ Wi @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ Wi @ Ak
        diff = np.linalg.norm(H_next - Hk) / max(np.linalg.norm(H_next), 1e-30)
        Ak, Gk, Hk = A_next, sym(G_next), sym(H_next)
        if diff < tol:
            return D @ Hk @ D, it
    return D @ Hk @ D, max_iter


# ---------------------------------------------------------------------------
# ADDA *como implementado* — AutoLQR.cpp:2096-2140 (forma V/W). Prova numérica
# de que é idêntico ao SDA (push-through: (I+HG)^-1 H = H(I+GH)^-1).
# ---------------------------------------------------------------------------
def adda_as_implemented(A, B, Q, R, tol=1e-12, max_iter=100):
    n = A.shape[0]
    Ak = A.copy()
    Gk = B @ np.linalg.solve(R, B.T)
    Hk = Q.copy()
    for it in range(1, max_iter + 1):
        V = np.linalg.inv(np.eye(n) + Gk @ Hk)
        W = np.linalg.inv(np.eye(n) + Hk @ Gk)
        A_next = Ak @ V @ Ak
        G_next = Gk + Ak @ V @ Gk @ Ak.T
        H_next = Hk + Ak.T @ W @ Hk @ Ak
        diff = np.linalg.norm(H_next - Hk) / max(np.linalg.norm(H_next), 1e-30)
        Ak, Gk, Hk = A_next, sym(G_next), sym(H_next)
        if diff < tol:
            return Hk, it
    return Hk, max_iter


def main():
    rows_out = []
    print(f"{'caso':22s} {'metodo':22s} {'resid_f64':>12s}  nota")
    print("-" * 80)

    for case in STORY_CASES:
        n, m, A, B, Q, R = read_case(case)

        P_scipy = solve_discrete_are(A, B, Q, R)
        resid_scipy = dare_residual(A, B, Q, R, P_scipy)
        rows_out.append([case, "scipy_oracle", resid_scipy, ""])
        print(f"{case:22s} {'scipy_oracle':22s} {resid_scipy:12.3e}  oraculo")

        P_sda, it_sda = sda_reference(A, B, Q, R)
        r = dare_residual(A, B, Q, R, P_sda)
        rows_out.append([case, "SDA_ref_f64", r, f"it={it_sda}"])
        print(f"{case:22s} {'SDA_ref_f64':22s} {r:12.3e}  it={it_sda}")

        P_ss, it_ss = sda_ss_as_implemented(A, B, Q, R)
        r = dare_residual(A, B, Q, R, P_ss)
        rows_out.append([case, "SDA_SS_atual_f64", r, f"it={it_ss}"])
        print(f"{case:22s} {'SDA_SS_atual_f64':22s} {r:12.3e}  it={it_ss}  <- deveria bater com SDA_ref")

        P_asda_bug, it_a1 = asda_as_implemented(A, B, Q, R)
        r = dare_residual(A, B, Q, R, P_asda_bug)
        rows_out.append([case, "ASDA_atual_f64", r, f"it={it_a1}"])
        print(f"{case:22s} {'ASDA_atual_f64':22s} {r:12.3e}  it={it_a1}  <- bug reverte-P")

        P_asda_fix, it_a2 = asda_fixed(A, B, Q, R)
        r = dare_residual(A, B, Q, R, P_asda_fix)
        rows_out.append([case, "ASDA_corrigido_f64", r, f"it={it_a2}"])
        print(f"{case:22s} {'ASDA_corrigido_f64':22s} {r:12.3e}  it={it_a2}  <- Fase 4.2")

        P_sc_bug, it_s1 = sda_scaled_as_implemented(A, B, Q, R)
        r = dare_residual(A, B, Q, R, P_sc_bug)
        rows_out.append([case, "SDA_SCALED_atual_f64", r, f"it={it_s1}"])
        print(f"{case:22s} {'SDA_SCALED_atual_f64':22s} {r:12.3e}  it={it_s1}  <- expoentes de D invertidos")

        P_sc_fix, it_s2 = sda_scaled_fixed(A, B, Q, R)
        r = dare_residual(A, B, Q, R, P_sc_fix)
        rows_out.append([case, "SDA_SCALED_corrigido_f64", r, f"it={it_s2}"])
        print(f"{case:22s} {'SDA_SCALED_corrigido_f64':22s} {r:12.3e}  it={it_s2}  <- Fase 4.3")

        P_adda, it_ad = adda_as_implemented(A, B, Q, R)
        r = dare_residual(A, B, Q, R, P_adda)
        rows_out.append([case, "ADDA_atual_f64", r, f"it={it_ad}"])
        equiv = np.linalg.norm(P_adda - P_sda) / max(np.linalg.norm(P_sda), 1e-30)
        rows_out.append([case, "ADDA_vs_SDA_diff_relativa", equiv, f"it_adda={it_ad},it_sda={it_sda}"])
        print(f"{case:22s} {'ADDA_atual_f64':22s} {r:12.3e}  it={it_ad}")
        print(f"{case:22s} {'ADDA_vs_SDA_diff':22s} {equiv:12.3e}  <- push-through: deveria ser ~eps_maquina")
        print()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "method", "dare_residual_f64", "note"])
        w.writerows(rows_out)
    print(f"Escrito: {OUT_CSV}")


if __name__ == "__main__":
    main()
