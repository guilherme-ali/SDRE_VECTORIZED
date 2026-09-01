"""
Espelho em Python da bateria de trajetórias de experiments/benchmark_solvers.cpp:
para cada ponto de operação (phi,theta,p,q,r) das 4 trajetórias de
python/trajetorias.py, monta Ad,Bd,Qd,Rd (mesma discretização analítica de
2a ordem do firmware, archive/test_archive/main_backup.cpp:910-1046) e calcula:

  1. a referência-ouro em dupla precisão via scipy.linalg.solve_discrete_are
     — resolve a validação circular apontada pelo revisor 2 do CBA 2026
     (o artigo submetido usava o próprio método iterativo como referência);
  2. o espelho float64 do SDA (verifica_solvers.sda_reference) como
     representante dos cinco métodos de duplicação em float — pós-auditoria,
     todos os cinco convergem para o MESMO P algebricamente (ver
     docs/auditoria_solvers_riccati.md); a diferença entre eles em produção
     é de robustez numérica sob float32/quantização, não de fórmula, e essa
     diferença só é observável no hardware real (por isso este script
     também sabe ler e confrontar a captura serial do ESP32);
  3. os seis métodos em Q13.18 emulado (python/fixedpoint_q.py), com o setup
     por método replicado das rotinas *_Fixed() de lib/AUTOLQR/AutoLQR.cpp.

Uso:
    python bench_trajetorias.py --decimacao 20 --saida outputs/bench_trajetorias_host.csv
    python bench_trajetorias.py --compare outputs/bench_trajetorias_s2.csv

Sem --compare, roda em modo "dry-run" host-only (não precisa do hardware) —
útil para validar a metodologia antes da bateria de ~50 min no ESP32-S2.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, os.path.dirname(__file__))
import fixedpoint_q as fxq
import trajetorias as trj
from verifica_solvers import sda_reference, dare_residual

# ===== Parâmetros físicos reais (idênticos a archive/test_archive/main_backup.cpp e ao C++ do benchmark) =====
IXX, IYY, IZZ, IR = 42.95e-6, 37.77e-6, 76.15e-6, 1.02e-7
L_ARM = 0.060 * 0.70710678
MOTOR_B, MOTOR_D = 2.98e-8, 0.05 * 2.98e-8
MAX_RPM = 26423.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
DT = trj.DT  # 0.006 s — mesmo período do firmware

ROLL_MAX, PITCH_MAX, YAW_MAX = math.radians(45), math.radians(45), math.radians(90)
P_MAX, Q_MAX, R_MAX = math.radians(300), math.radians(300), math.radians(200)
Q_DIAG = np.array([1 / ROLL_MAX**2, 1 / PITCH_MAX**2, 1 / YAW_MAX**2,
                    1 / P_MAX**2, 1 / Q_MAX**2, 1 / R_MAX**2])
PERC_TAU = 0.5
MAX_TAU_ROLL = 2.0 * MOTOR_B * L_ARM * MAX_OMEGA**2 * PERC_TAU
MAX_TAU_PITCH = MAX_TAU_ROLL
MAX_TAU_YAW = 4.0 * MOTOR_D * MAX_OMEGA**2 * PERC_TAU
R_DIAG = np.array([1 / MAX_TAU_ROLL**2, 1 / MAX_TAU_PITCH**2, 1 / MAX_TAU_YAW**2])


def build_Ad_Bd_Qd_Rd(phi, theta, p, q, r, omega_r=0.0):
    """Porta em numpy (denso, sem exploração de esparsidade — não precisa
    de performance no host) da mesma discretização analítica de 2a ordem
    de archive/test_archive/main_backup.cpp:910-1046 / experiments/benchmark_solvers.cpp."""
    n = 6
    A = np.zeros((n, n))
    sR, cR = math.sin(phi), math.cos(phi)
    sP, cP = math.sin(theta), math.cos(theta)
    tP = sP / cP
    A[0, 3] = 1.0
    A[0, 4] = sR * tP
    A[0, 5] = cR * tP
    A[1, 4] = cR
    A[1, 5] = -sR
    A[2, 4] = sR / cP
    A[2, 5] = cR / cP
    # Fatoracao SDC com alpha_1=1, alpha_2=-1, alpha_3=0 (ver artigo CBA, Eq. 14):
    # A_22 = [[0, a1*r - Ir*Omega_r/Ixx, a2*q],
    #         [b1*r + Ir*Omega_r/Iyy, 0, b2*p],
    #         [c1*q, c2*p, 0]]
    # com c1 = alpha_3*(Ixx-Iyy)/Izz = 0 e c2 = (1-alpha_3)*(Ixx-Iyy)/Izz.
    # Logo o termo nao-nulo da ultima linha e A[5,4] = (Ixx-Iyy)/Izz * p, NAO
    # A[5,3]. (Estava trocado aqui ate 2026-08-15; o C++ sempre esteve certo --
    # main_backup.cpp:940 `A54 = Ixx_Iyy_over_Izz * p`.)
    A[3, 4] = (IYY - IZZ) / IXX * r - (IR / IXX) * omega_r
    A[4, 3] = (IR / IYY) * omega_r
    A[4, 5] = (IZZ - IXX) / IYY * p
    A[5, 4] = (IXX - IYY) / IZZ * p

    B = np.zeros((n, 3))
    B[3, 0] = 1.0 / IXX
    B[4, 1] = 1.0 / IYY
    B[5, 2] = 1.0 / IZZ

    Qc = np.diag(Q_DIAG)
    Rc = np.diag(R_DIAG)

    A2 = A @ A
    Ad = np.eye(n) + A * DT + A2 * (DT * DT * 0.5)
    AB = A @ B
    Bd = B * DT + AB * (DT * DT * 0.5)
    ATQ_QA = A.T @ Qc + Qc @ A
    Qd = Qc * DT + ATQ_QA * (DT * DT * 0.5)
    BTQB = B.T @ Qc @ B
    Rd = Rc * DT + BTQB * (DT ** 3 / 3.0)
    return Ad, Bd, Qd, Rd


# ---------------------------------------------------------------------------
# Setup por método em Q13.18, espelhando as rotinas *_Fixed() de AutoLQR.cpp
# (ver docs/auditoria_solvers_riccati.md, Seção 8). Cada função retorna
# (P_hat_float, K_float, status, iters) já revertido para float64.
# ---------------------------------------------------------------------------
def _solve_doubling_q(A, B, Q, R, variant, sh=fxq.Q_SHIFT_DEFAULT, extra=None):
    n = A.shape[0]
    st = fxq.Status()
    G = B @ np.linalg.solve(R, B.T)
    A0, G0, H0 = A, G, Q
    if extra is not None:
        A0, G0, H0 = extra(A, G, Q)
    Aq = fxq.mat_to_q(A0, sh, st)
    Gq = fxq.mat_to_q(G0, sh, st)
    Hq = fxq.mat_to_q(H0, sh, st)
    # Criterio casado com o hardware (Secao 15.2 da auditoria, 2026-08-18):
    # Frobenius relativa, tolerancia 1e-3 (invRelTolerance=1000), orcamento
    # 200 -- era 1e-6/1000000, abaixo do piso de quantizacao do Q13.18
    # (medido em ~5.6e-5 no pior caso), corrigido apos a re-varredura de
    # tolerancia (Exp. 0) desta sessao.
    _, _, Hkq, cum_s, ok = fxq.doubling_loop_q(Aq, Gq, Hq, n, sh, variant,
                                                max_iterations=200, inv_rel_tolerance=1000, st=st)
    if not ok or st.overflow:
        return None, st
    Hk = fxq.q_to_mat(Hkq, n, n, sh)
    return Hk * cum_s, st


def sda_fixed_q(A, B, Q, R):
    P, st = _solve_doubling_q(A, B, Q, R, fxq.VARIANT_STANDARD)
    return P, st


def asda_fixed_q(A, B, Q, R):
    P, st = _solve_doubling_q(A, B, Q, R, fxq.VARIANT_ADAPTIVE_SCALING)
    return P, st


def adda_fixed_q(A, B, Q, R):
    P, st = _solve_doubling_q(A, B, Q, R, fxq.VARIANT_ALTERNATING_VW)
    return P, st


def sda_ss_fixed_q(A, B, Q, R, gamma=0.7):
    # gamma=0.7: valor medido no Exp. 3 (docs/auditoria_solvers_riccati.md,
    # Secao 15.3) -- domina o antigo default gamma=0.5 nas duas aritmeticas
    # (12,5% mais rapido, residuo 8-15% menor). AutoLQR::ssGamma default
    # tambem foi atualizado para 0.7 em lib/AUTOLQR/AutoLQR.cpp.
    #
    # HISTORICO -- a versão anterior a esta usava gamma=0.3 com o shift afim só em A
    # (A0=(A-gI)/(1-g), G0=G/(1-g)^2), que é uma DARE DIFERENTE da que o
    # device resolve. lib/AUTOLQR/AutoLQR.cpp:computeGainMatrixSDA_SS_Fixed()
    # (e o par float computeGainMatrixSDA_SS()) usam o shift completo do
    # pencil simplético 12x12: N1 = [[I-gA,-gG0],[gH0,I-gA']],
    # Phi=N1^-1, e (Â,Ĝ,Ĥ) extraídos dos quatro blocos de Phi — ver comentário
    # de implementação em AutoLQR.cpp para a derivação. Essa discrepância
    # produzia erro_K_SDA_SS_FIXED ~9e-4 no espelho host contra ~4e-5 medido
    # no hardware (ver docs/auditoria_solvers_riccati.md, Seção 13, e a nota
    # de limitação em outputs/relatorio_bateria*.md).
    def setup(A, G, Q):
        n = A.shape[0]
        H0 = Q
        I = np.eye(n)
        N1 = np.block([
            [I - gamma * A, -gamma * G],
            [gamma * H0, I - gamma * A.T],
        ])
        Phi = np.linalg.inv(N1)
        Phi11, Phi12 = Phi[:n, :n], Phi[:n, n:]
        Phi21, Phi22 = Phi[n:, :n], Phi[n:, n:]
        AmG = A - gamma * I
        ATmG = A.T - gamma * I
        Ahat = Phi11 @ AmG - Phi12 @ H0
        Ghat = Phi11 @ G + Phi12 @ ATmG
        Hhat = -Phi21 @ AmG + Phi22 @ H0
        return Ahat, Ghat, Hhat
    P, st = _solve_doubling_q(A, B, Q, R, fxq.VARIANT_STANDARD, extra=setup)
    return P, st


def sda_scaled_fixed_q(A, B, Q, R):
    n = A.shape[0]
    d = 1.0 / np.sqrt(np.maximum(np.linalg.norm(A, axis=1), 1e-30))
    D = np.diag(d)
    Dinv = np.diag(1.0 / d)

    def setup(A, G, Q):
        return D @ A @ Dinv, D @ G @ D, Dinv @ Q @ Dinv
    P_hat, st = _solve_doubling_q(A, B, Q, R, fxq.VARIANT_STANDARD, extra=setup)
    if P_hat is None:
        return None, st
    return D @ P_hat @ D, st


def k_from_P(A, B, Q, R, P):
    S = R + B.T @ P @ B
    return np.linalg.solve(S, B.T @ P @ A)


FIXED_SOLVERS = {
    "SDA_FIXED": sda_fixed_q,
    "SDA_SS_FIXED": sda_ss_fixed_q,
    "ASDA_FIXED": asda_fixed_q,
    "SDA_SCALED_FIXED": sda_scaled_fixed_q,
    "ADDA_FIXED": adda_fixed_q,
}


class IterativeFixedGain:
    """Espelho de AutoLQR::computeGainMatrixIterative_Fixed() (AutoLQR.cpp:2680-2799)
    -- iteracao de valor em Q13.18, com warm-start GENUINO via P_warm mantido como
    estado da instancia (nao um dict compartilhado entre chamadores, para nao
    misturar warm-start entre trajetorias/controladores diferentes sem querer).
    Fora de FIXED_SOLVERS de proposito: esse dict e' consumido em varios lugares
    (bench_trajetorias.run_dry/run_compare) que assumem funcoes STATELESS
    (uma chamada = um resultado independente); a iteracao de valor so faz
    sentido com estado persistente entre chamadas consecutivas -- criar uma
    instancia nova por (trajetoria, controlador) e' responsabilidade do
    chamador (ver malha_fechada_trajetorias.py).

    Uso: `g = IterativeFixedGain(); P, st = g(Ad, Bd, Qd, Rd)` -- __call__ tem a
    MESMA assinatura (A,B,Q,R)->(P,st) dos *_fixed_q acima, entao serve como
    gain_fn em simulate_closed_loop() sem mudar sua interface.
    """

    def __init__(self, sh=fxq.Q_SHIFT_DEFAULT, max_iterations=200, inv_rel_tolerance=1000):
        self.sh = sh
        self.max_iterations = max_iterations
        self.inv_rel_tolerance = inv_rel_tolerance
        self.P_warm = None  # None == zero-inicializado, espelha o ctor de AutoLQR

    def __call__(self, A, B, Q, R):
        n = A.shape[0]
        sh = self.sh
        st = fxq.Status()

        Aq = fxq.mat_to_q(A, sh, st)
        Bq = fxq.mat_to_q(B, sh, st)
        Qq = fxq.mat_to_q(Q, sh, st)
        Rq = fxq.mat_to_q(R, sh, st)
        if st.overflow:
            return None, st

        ATq = fxq.transpose_q(Aq, n, n)
        BTq = fxq.transpose_q(Bq, n, B.shape[1])
        m = B.shape[1]

        # Regularizacao 1e-5 (AutoLQR.cpp:2712) -- 1e-8 do par float vira zero
        # exato em Q13.18 (0,0026 LSB); 1e-5 fica acima do piso de quantizacao.
        eps = fxq.f2q(1e-5, sh, st)

        Pw_norm = 0.0 if self.P_warm is None else float(np.sum(np.abs(self.P_warm)))
        has_warm_start = Pw_norm > 1e-6
        if has_warm_start:
            Pk = fxq.mat_to_q(self.P_warm, sh, st)
        else:
            Pk = list(Qq)
        if st.overflow:
            return None, st

        nn = n * n
        iters = self.max_iterations
        for it in range(self.max_iterations):
            PA = fxq.matmul_q(Pk, Aq, n, n, n, sh, st)
            PB = fxq.matmul_q(Pk, Bq, n, n, m, sh, st)
            ATPA = fxq.matmul_q(ATq, PA, n, n, n, sh, st)
            BTPB = fxq.matmul_q(BTq, PB, m, n, m, sh, st)
            BTPA = fxq.matmul_q(BTq, PA, m, n, n, sh, st)

            S = fxq.add_q(Rq, BTPB)
            S = list(S)
            for i in range(m):
                S[i * m + i] += eps

            Sinv, ok = fxq.invert_q(S, m, sh, st)
            if not ok:
                return None, st

            Ktmp = fxq.matmul_q(Sinv, BTPA, m, m, n, sh, st)
            ATPB = fxq.matmul_q(ATq, PB, n, n, m, sh, st)
            corr = fxq.matmul_q(ATPB, Ktmp, n, m, n, sh, st)

            Pnext = [Qq[i] + ATPA[i] - corr[i] for i in range(nn)]
            # Simetrizacao por divisao INTEIRA truncada (AutoLQR.cpp:2751,
            # `(a+b)/2` em q_t) -- deliberadamente NAO arredondada, espelha o
            # dither de 1 LSB documentado na auditoria (Secao 15.2/13).
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = Pnext[i * n + j], Pnext[j * n + i]
                    # C++ integer division trunca em direcao a zero; replica isso:
                    s = a + b
                    avg = -((-s) // 2) if s < 0 else s // 2
                    Pnext[i * n + j] = avg
                    Pnext[j * n + i] = avg

            diffSq = 0.0
            hSq = 0.0
            for i in range(nn):
                d = fxq.q2f(Pnext[i], sh) - fxq.q2f(Pk[i], sh)
                h = fxq.q2f(Pk[i], sh)
                diffSq += d * d
                hSq += h * h
            Pk = Pnext
            relF = math.sqrt(diffSq / hSq) if hSq > 1e-20 else math.sqrt(diffSq)
            if relF < (1.0 / float(self.inv_rel_tolerance)):
                iters = it + 1
                break

        if st.overflow:
            return None, st

        PA = fxq.matmul_q(Pk, Aq, n, n, n, sh, st)
        PB = fxq.matmul_q(Pk, Bq, n, n, m, sh, st)
        BTPB = fxq.matmul_q(BTq, PB, m, n, m, sh, st)
        BTPA = fxq.matmul_q(BTq, PA, m, n, n, sh, st)
        S = list(fxq.add_q(Rq, BTPB))
        for i in range(m):
            S[i * m + i] += eps
        Sinv, ok = fxq.invert_q(S, m, sh, st)
        if not ok:
            return None, st
        Kq = fxq.matmul_q(Sinv, BTPA, m, m, n, sh, st)
        if st.overflow:
            return None, st

        P_out = fxq.q_to_mat(Pk, n, n, sh)
        self.P_warm = P_out.copy()  # atualiza p/ a proxima chamada (AutoLQR.cpp:2791)
        st.iterations = iters
        return P_out, st


def rms(K1, K2):
    return float(np.sqrt(np.mean((K1 - K2) ** 2)))


def run_dry(decimacao, saida):
    dados = trj.gerar_todas()
    rows = []
    n_total = sum(len(d["t"]) for d in dados.values())
    n_amostrado = 0
    for traj_nome, d in dados.items():
        idx = range(0, len(d["t"]), decimacao)
        for k in idx:
            phi, theta, p, q, r = d["phi"][k], d["theta"][k], d["p"][k], d["q"][k], d["r"][k]
            Ad, Bd, Qd, Rd = build_Ad_Bd_Qd_Rd(phi, theta, p, q, r)
            try:
                P_ref = solve_discrete_are(Ad, Bd, Qd, Rd)
            except Exception:
                continue
            K_ref = k_from_P(Ad, Bd, Qd, Rd, P_ref)

            P_sda, it_sda = sda_reference(Ad, Bd, Qd, Rd)
            K_sda = k_from_P(Ad, Bd, Qd, Rd, P_sda)
            row = {
                "traj": traj_nome, "k": k, "t": d["t"][k],
                "phi": phi, "theta": theta, "p": p, "q": q, "r": r,
                "residuo_ref": dare_residual(Ad, Bd, Qd, Rd, P_ref),
                "erro_K_sda_float64_vs_ref": rms(K_sda, K_ref),
                "iters_sda_float64": it_sda,
            }
            for nome, fn in FIXED_SOLVERS.items():
                P_fx, st = fn(Ad, Bd, Qd, Rd)
                if P_fx is None:
                    row[f"erro_K_{nome}"] = float("nan")
                    row[f"overflow_{nome}"] = True
                    row[f"max_abs_seen_{nome}"] = st.max_abs_seen
                    continue
                K_fx = k_from_P(Ad, Bd, Qd, Rd, P_fx)
                row[f"erro_K_{nome}"] = rms(K_fx, K_ref)
                row[f"overflow_{nome}"] = st.overflow
                row[f"max_abs_seen_{nome}"] = st.max_abs_seen
            rows.append(row)
            n_amostrado += 1

    if not rows:
        print("nenhum ponto amostrado (decimacao muito grande?)")
        return

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"gravado: {saida} ({n_amostrado}/{n_total} pontos amostrados, decimacao=1:{decimacao})")

    q13_18_ceiling_real = 2.0 ** 13  # 8192 — faixa representavel em UNIDADES REAIS (nao no inteiro Q-format)
    for nome in FIXED_SOLVERS:
        vals = [row[f"erro_K_{nome}"] for row in rows if not math.isnan(row[f"erro_K_{nome}"])]
        maxabs_real = [fxq.q2f(row[f"max_abs_seen_{nome}"], fxq.Q_SHIFT_DEFAULT) for row in rows]
        ovf = sum(1 for row in rows if row[f"overflow_{nome}"])
        if vals:
            pico = max(maxabs_real)
            print(f"{nome:20s} erro_K medio={np.mean(vals):.3e}  max={np.max(vals):.3e}  "
                  f"overflow={ovf}/{len(rows)}  max_abs_seen(pico)={pico:.1f} "
                  f"(teto Q13.18={q13_18_ceiling_real:.0f}, margem={q13_18_ceiling_real/max(pico,1e-9):.1f}x)")
        else:
            print(f"{nome:20s} TODAS as amostras falharam (overflow/singular)")


def load_device_csv(path):
    """Lê a captura serial do ESP32 (linhas PT,/RUN,/GAIN,/SUMMARY,) — ver
    cabecalho de experiments/benchmark_solvers.cpp para o formato de cada uma."""
    pontos, runs, gains, summaries = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            tag = parts[0]
            if tag == "PT":
                pontos.append(parts[1:])
            elif tag == "RUN":
                runs.append(parts[1:])
            elif tag == "GAIN":
                gains.append(parts[1:])
            elif tag == "SUMMARY":
                summaries.append(parts[1:])
    return pontos, runs, gains, summaries


def run_compare(device_csv):
    pontos, runs, gains, summaries = load_device_csv(device_csv)
    print(f"captura do dispositivo: {len(pontos)} pontos, {len(runs)} linhas RUN, "
          f"{len(gains)} linhas GAIN, {len(summaries)} linhas SUMMARY")

    print("\n=== SUMMARY (do proprio ESP32) ===")
    print(f"{'traj':14s} {'metodo':20s} {'mean_us':>10s} {'std_us':>9s} {'max_us':>9s} "
          f"{'mean_it':>8s} {'mean_res':>11s} {'fail':>6s} {'count':>7s}")
    for row in summaries:
        traj, metodo, mean_us, std_us, max_us, mean_it, max_it, mean_res, max_res, fail, count = row
        print(f"{traj:14s} {metodo:20s} {float(mean_us):10.2f} {float(std_us):9.2f} {float(max_us):9.2f} "
              f"{float(mean_it):8.3f} {float(mean_res):11.3e} {fail:>6s} {count:>7s}")

    print("\n=== Erro de K (GAIN do device) vs. referencia scipy (dupla precisao, host) ===")
    erros = {}
    # CORRIGIDO 2026-08-19: a versao anterior chamava nome_fn(np.array([k*DT])),
    # um array de 1 elemento, contra um dicionario hardcoded com so 4 entradas
    # (T5/T6 seriam silenciosamente ignorados via .get()->None->continue) --
    # e _derivar_central() (trajetorias.py) faz x[1] em d[0]=(x[1]-x[0])/dt,
    # que sempre daria IndexError num array de tamanho 1 se a trajetoria caisse
    # no ramo de diferenca central (nao regressiva). gerar_todas() ja calcula o
    # array completo (10000 pontos) uma vez por trajetoria; indexar por k evita
    # o caso de borda da diferenca central E generaliza para as 6 trajetorias
    # sem precisar listar cada uma a mao.
    dados_todas = trj.gerar_todas()
    for g in gains:
        traj, k, metodo = g[0], int(g[1]), g[2]
        K_dev = np.array([float(x) for x in g[3:3 + 18]]).reshape(3, 6)
        d_traj = dados_todas.get(traj)
        if d_traj is None or k >= len(d_traj["phi"]):
            continue
        phi, theta, p, q, r = d_traj["phi"][k], d_traj["theta"][k], d_traj["p"][k], d_traj["q"][k], d_traj["r"][k]
        Ad, Bd, Qd, Rd = build_Ad_Bd_Qd_Rd(phi, theta, p, q, r)
        try:
            P_ref = solve_discrete_are(Ad, Bd, Qd, Rd)
        except Exception:
            continue
        K_ref = k_from_P(Ad, Bd, Qd, Rd, P_ref)
        e = rms(K_dev, K_ref)
        erros.setdefault(metodo, []).append(e)

    for metodo, vals in sorted(erros.items()):
        print(f"{metodo:20s} n={len(vals):5d}  erro_K medio={np.mean(vals):.3e}  max={np.max(vals):.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decimacao", type=int, default=20,
                     help="passo de amostragem sobre os 46152 pontos no modo dry-run (padrao: 20)")
    ap.add_argument("--saida", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "bench_trajetorias_host.csv"))
    ap.add_argument("--compare", metavar="CSV_DISPOSITIVO",
                     help="confronta uma captura serial do ESP32 (experiments/benchmark_solvers.cpp) contra a referencia scipy")
    args = ap.parse_args()

    if args.compare:
        run_compare(args.compare)
    else:
        run_dry(args.decimacao, args.saida)
