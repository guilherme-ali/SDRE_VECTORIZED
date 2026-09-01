"""
Trajetórias de atitude usadas na bateria de benchmark dos solvers DARE
(substitui o passeio aleatório de experiments/benchmark_solvers.cpp — ver
revisoes_consolidadas.md, item 2.2, crítica do R2 do CBA 2026).

Espelho em Python de lib/Trajectories/Trajectories.h (fonte única do lado
C++, incluída por experiments/benchmark_solvers.cpp, test/tolerance_sweep.cpp,
test/gamma_sweep.cpp e test/sweep_qr.cpp). Qualquer mudança aqui tem de ser
feita nos dois lados — a paridade bit-a-bit é premissa do projeto.

Cada trajetória é uma função fechada e determinística t -> (phi, theta, psi)
[rad]. As velocidades angulares do corpo (p, q, r) são obtidas por um único
diferenciador central de 3 pontos comum a todas — a MESMA forma implementada
em C++ em experiments/benchmark_solvers.cpp (buildTrajectoryPoint), o que garante
que host e alvo produzem a sequência de estados bit-a-bit comparável (dentro
da tolerância de float32). Isso é o que se verifica em
outputs/traj_ref.csv vs. a captura serial (ver plano, Verificação, item 1).

T_s = 6,0 ms — o mesmo período do laço de controle do firmware
(test/verify_gains_onboard.cpp, src/main.cpp), não um valor arbitrário
(resposta à anotação #24 do Reginaldo pedindo justificativa para T_s).
"""

import argparse
import os

import numpy as np

DT = 0.006  # s — período real do laço de controle (não os 12 ms do artigo do CBA)
G = 9.81  # m/s^2
DURATION_S = 60.0
N_POINTS = int(round(DURATION_S / DT))  # 10000

# Saturação de atitude: evita a singularidade de sec(theta) na cinemática de
# Euler e mantém o ponto de operação fisicamente plausível para um
# quadricóptero pequeno. É POR TRAJETÓRIA (espelha phiMaxFor/thetaMaxFor de
# lib/Trajectories/Trajectories.h): T1-T4 e T6 usam os 60° históricos; T5
# (tilt alto) abre para 85°, onde sec(theta) = 11,5.
PHI_MAX = np.deg2rad(60.0)
THETA_MAX = np.deg2rad(60.0)
PHI_MAX_T5 = np.deg2rad(85.0)
THETA_MAX_T5 = np.deg2rad(85.0)


def _tempo():
    return np.arange(N_POINTS) * DT


def _saturar(ang, lim):
    return np.clip(ang, -lim, lim)


def _cinematica_inversa(phi, theta, psi_dot_ignorado, phi_dot, theta_dot, psi_dot):
    """(phi_dot, theta_dot, psi_dot) -> (p, q, r), inversa fechada da
    cinemática de Euler usada em updateSystemMatrixBench() /
    buildC1Hover() (sem inversão numérica de matriz)."""
    sphi, cphi = np.sin(phi), np.cos(phi)
    stheta, ctheta = np.sin(theta), np.cos(theta)
    p = phi_dot - stheta * psi_dot
    q = cphi * theta_dot + sphi * ctheta * psi_dot
    r = -sphi * theta_dot + cphi * ctheta * psi_dot
    return p, q, r


def _derivar_central(x, dt):
    """Diferenciador central de 3 pontos, extremidades por diferença de 1a
    ordem — mesma forma usada em experiments/benchmark_solvers.cpp."""
    d = np.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def _derivar_regressiva(x, dt):
    """Diferença regressiva de 1a ordem; taxa inicial = 0 (parte do repouso,
    fisicamente correto para T4 — hover em t=0). Usada apenas em T4: a
    conformação de 1a ordem ali é uma recursão (phi[k] depende de
    phi[k-1]), então no C++ embarcado ela é gerada em streaming, O(1) de
    memória, sem olhar para o futuro — logo não dá para centrar a
    diferença sem armazenar todo o vetor. Usar a mesma forma regressiva
    aqui mantém Python e C++ bit-a-bit comparáveis também em T4 (ver
    experiments/benchmark_solvers.cpp)."""
    d = np.empty_like(x)
    d[1:] = (x[1:] - x[:-1]) / dt
    d[0] = 0.0
    return d


def _finalizar(t, phi, theta, psi, regressiva=False, phi_max=None, theta_max=None):
    phi = _saturar(phi, PHI_MAX if phi_max is None else phi_max)
    theta = _saturar(theta, THETA_MAX if theta_max is None else theta_max)
    deriv = _derivar_regressiva if regressiva else _derivar_central
    phi_dot = deriv(phi, DT)
    theta_dot = deriv(theta, DT)
    psi_dot = deriv(np.unwrap(psi), DT)
    p, q, r = _cinematica_inversa(phi, theta, None, phi_dot, theta_dot, psi_dot)
    return {
        "t": t, "phi": phi, "theta": theta, "psi": psi,
        "p": p, "q": q, "r": r,
    }


def traj_t1_espiral(t=None):
    """T1 — espiral circular de raio crescente (pedido do Reginaldo): a
    aceleração centrípeta omega^2*R(t) cresce linearmente com o tempo,
    varrendo de quase-hover a inclinação próxima do limite prático.
    Heading (psi) fixo em 0 — o drone translada em círculo sem apontar o
    nariz para o centro, o que já basta para acoplar phi e theta via a
    direção do vetor de empuxo (planicidade diferencial)."""
    if t is None:
        t = _tempo()
    R0, Rdot, w = 0.5, 0.05, 2.0  # m, m/s, rad/s
    R = R0 + Rdot * t
    sw, cw = np.sin(w * t), np.cos(w * t)
    # x = R cos(wt), y = R sin(wt); derivadas 2a ordem em forma fechada
    xdd = -2.0 * Rdot * w * sw - R * w * w * cw
    ydd = 2.0 * Rdot * w * cw - R * w * w * sw
    psi = np.zeros_like(t)
    theta = np.arctan2(xdd, G)
    phi = np.arctan2(-ydd * np.cos(theta), G)
    return _finalizar(t, phi, theta, psi)


def traj_t2_figura8(t=None):
    """T2 — figura-8 (Lissajous 1:2) em atitude: phi e theta em razão de
    frequência 1:2 traçam uma lemniscata no plano (phi, theta), com
    inversões periódicas de sinal nos termos cruzados de A(x). psi
    acompanha com pequena amplitude para acoplar os três eixos."""
    if t is None:
        t = _tempo()
    A = np.deg2rad(25.0)
    T = 8.0  # s, período da figura
    w = 2.0 * np.pi / T
    phi = A * np.sin(w * t)
    theta = A * np.sin(2.0 * w * t)
    psi = np.deg2rad(15.0) * np.sin(w * t)
    return _finalizar(t, phi, theta, psi)


def traj_t3_chirp(t=None):
    """T3 — chirp linear em roll/pitch (defasados 90°): frequência
    instantânea varre 0,2 -> 8 Hz ao longo dos 60 s, excitando toda a banda
    num único experimento; o eixo x = frequência instantânea é
    interpretável diretamente nas figuras."""
    if t is None:
        t = _tempo()
    A = np.deg2rad(25.0)
    f0, f1 = 0.2, 8.0  # Hz
    T = DURATION_S
    fase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * T))
    phi = A * np.sin(fase)
    theta = A * np.sin(fase + np.pi / 2.0)
    psi = np.zeros_like(t)
    return _finalizar(t, phi, theta, psi)


def traj_t4_degrau_yaw(t=None):
    """T4 — degraus agressivos em roll/pitch (±40°, a cada 2 s) conformados
    por 1a ordem (tau=150 ms, p/q ficam finitos) somados a um giro de yaw
    contínuo (psi_dot = 2 rad/s) — pior caso do termo r em A_22 e do
    warm-start do método iterativo: descontinuidades reais, não ruído."""
    if t is None:
        t = _tempo()
    tau = 0.15
    periodo = 2.0

    # Alvo derivado da FASE, espelhando `t4RawStep()` de
    # lib/Trajectories/Trajectories.h.
    #
    # A versao anterior tirava o alvo do SINAL de um seno, e o periodo de 2 s
    # com DT = 6 ms faz a grade cair EXATAMENTE sobre os zeros dessas funcoes
    # (t = 3.0 s no seno, k = 500; t = 1.5 s no cosseno, k = 250). No zero o
    # sinal depende da precisao — sinf(3*pi) em float32 da' -8.7e-8 e
    # sin(3*pi) em float64 da' +3.7e-16, alvos opostos de +-40 graus. Com
    # alpha = 0.0385 sobre 80 graus de diferenca, um passo separava firmware e
    # espelho em 3.08 graus, e o erro reaparecia a cada cruzamento: 1734 dos
    # 10000 pontos, 12.5% de RMS. Isso entrava no e_K como se fosse erro de
    # solver (4.18e-4 em T4 contra 1.5e-6 no ponto real), inflando a coluna
    # float da Tabela 1 do artigo de 1.9e-6 para 3.7e-6.
    #
    # Replicar float32 aqui reduzia mas nao eliminava o problema: para t grande
    # o argumento 2*pi*t/2 ~ 155 rad perde resolucao em float32, e a reducao de
    # argumento do cosf do newlib do ESP32 difere da do numpy — perto de um zero
    # exato isso desloca o cruzamento. Casar bit-a-bit transcendentais entre
    # duas libm nao e' alcancavel; a solucao e' nao depender de transcendental.
    #
    # A forma abaixo decide pelo intervalo de fase, exata em qualquer precisao.
    alpha = DT / (tau + DT)
    amp = np.deg2rad(40.0)
    ciclos = (np.arange(len(t)) * DT) / periodo
    fase = ciclos - np.floor(ciclos)
    alvo_phi_v = np.where(fase < 0.5, amp, -amp)                      # sin >= 0
    alvo_theta_v = np.where((fase < 0.25) | (fase >= 0.75), amp, -amp)  # cos >= 0

    n = len(t)
    phi = np.zeros(n)
    theta = np.zeros(n)
    for k in range(1, n):
        phi[k] = phi[k - 1] + alpha * (alvo_phi_v[k] - phi[k - 1])
        theta[k] = theta[k - 1] + alpha * (alvo_theta_v[k] - theta[k - 1])
    psi_dot = 2.0  # rad/s, giro de guinada sustentado
    psi = np.mod(psi_dot * t + np.pi, 2.0 * np.pi) - np.pi  # wrap [-pi, pi]
    return _finalizar(t, phi, theta, psi, regressiva=True)


def traj_t5_tilt_alto(t=None):
    """T5 — TILT ALTO: theta varre ±80° com período longo (15 s), levando
    sec(theta) a 5,8 no pico (o limite de saturação da trajetória é 85°,
    sec = 11,5, deliberadamente acima da amplitude para nunca encostar na
    singularidade de sec em 90°). A frequência é baixa DE PROPÓSITO: o
    objetivo é isolar o efeito do ÂNGULO sobre o condicionamento da Riccati,
    com taxas (~34°/s) duas ordens de grandeza abaixo das de T1-T4, para que
    nenhum efeito medido possa ser atribuído à taxa. phi fica pequeno (5°,
    período 30 s) só para manter os termos cruzados sin(phi)*tan(theta) e
    sin(phi)/cos(theta) não-nulos; psi = 0.
    Espelha Trajectories::attitudeT5 (lib/Trajectories/Trajectories.h)."""
    if t is None:
        t = _tempo()
    A_theta = np.deg2rad(80.0)
    A_phi = np.deg2rad(5.0)
    periodo = 15.0
    w = 2.0 * np.pi / periodo
    theta = A_theta * np.sin(w * t)
    phi = A_phi * np.sin(0.5 * w * t)
    psi = np.zeros_like(t)
    return _finalizar(t, phi, theta, psi, phi_max=PHI_MAX_T5, theta_max=THETA_MAX_T5)


def traj_t6_taxa_alta(t=None):
    """T6 — TAXA ALTA: extensão direta de T3 (seno defasado 90° em
    roll/pitch), mas em 10 Hz FIXOS e 45° de amplitude, contra os 8 Hz / 25°
    do fim do chirp — o que dá phi_dot = 45°*2*pi*10 = 2827°/s, 2,3x o máximo
    já observado nas quatro trajetórias originais (~1234°/s). Os ângulos
    ficam dentro dos 60° padrão: aqui se isola o efeito da TAXA angular, não
    do ângulo (esse é T5). psi acompanha em 2 Hz para que r também suba.
    Espelha Trajectories::attitudeT6 (lib/Trajectories/Trajectories.h)."""
    if t is None:
        t = _tempo()
    A = np.deg2rad(45.0)
    f6 = 10.0  # Hz
    A_psi = np.deg2rad(30.0)
    w = 2.0 * np.pi * f6
    phi = A * np.sin(w * t)
    theta = A * np.sin(w * t + np.pi / 2.0)
    psi = A_psi * np.sin(0.2 * w * t)  # 2 Hz
    return _finalizar(t, phi, theta, psi)


TRAJETORIAS = {
    "T1_espiral": traj_t1_espiral,
    "T2_figura8": traj_t2_figura8,
    "T3_chirp": traj_t3_chirp,
    "T4_degrau_yaw": traj_t4_degrau_yaw,
    "T5_tilt_alto": traj_t5_tilt_alto,
    "T6_taxa_alta": traj_t6_taxa_alta,
}


def gerar_todas():
    return {nome: fn() for nome, fn in TRAJETORIAS.items()}


def dump_csv(path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj", "k", "t", "phi", "theta", "psi", "p", "q", "r"])
        for nome, dados in gerar_todas().items():
            for k in range(len(dados["t"])):
                w.writerow([
                    nome, k,
                    "%.6f" % dados["t"][k],
                    "%.8f" % dados["phi"][k], "%.8f" % dados["theta"][k], "%.8f" % dados["psi"][k],
                    "%.8f" % dados["p"][k], "%.8f" % dados["q"][k], "%.8f" % dados["r"][k],
                ])
    print("gravado: %s (%d trajetórias x %d pontos)" % (path, len(TRAJETORIAS), N_POINTS))


def plotar(saida_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dados = gerar_todas()
    fig, axes = plt.subplots(len(dados), 3, figsize=(14, 3.2 * len(dados)))
    for i, (nome, d) in enumerate(dados.items()):
        axes[i, 0].plot(d["t"], np.rad2deg(d["phi"]), label="phi")
        axes[i, 0].plot(d["t"], np.rad2deg(d["theta"]), label="theta")
        axes[i, 0].plot(d["t"], np.rad2deg(d["psi"]), label="psi")
        axes[i, 0].set_title("%s — atitude (graus)" % nome)
        axes[i, 0].legend(fontsize=7)

        axes[i, 1].plot(d["t"], np.rad2deg(d["p"]), label="p")
        axes[i, 1].plot(d["t"], np.rad2deg(d["q"]), label="q")
        axes[i, 1].plot(d["t"], np.rad2deg(d["r"]), label="r")
        axes[i, 1].set_title("%s — taxas (graus/s)" % nome)
        axes[i, 1].legend(fontsize=7)

        axes[i, 2].plot(np.rad2deg(d["phi"]), np.rad2deg(d["theta"]))
        axes[i, 2].set_title("%s — phi vs theta" % nome)
        axes[i, 2].set_xlabel("phi (deg)")
        axes[i, 2].set_ylabel("theta (deg)")
    fig.tight_layout()
    out = os.path.join(saida_dir, "trajetorias_preview.png")
    fig.savefig(out, dpi=120)
    print("figura salva: %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", metavar="CSV", help="grava as 6 trajetórias em CSV (traj,k,t,phi,theta,psi,p,q,r)")
    ap.add_argument("--plot", action="store_true", help="salva outputs/trajetorias_preview.png")
    args = ap.parse_args()

    if args.dump:
        dump_csv(args.dump)
    if args.plot:
        outdir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(outdir, exist_ok=True)
        plotar(outdir)
    if not args.dump and not args.plot:
        for nome, d in gerar_todas().items():
            print("%-14s N=%d  phi_max=%6.1f°  theta_max=%6.1f°  "
                  "|p|_max=%7.1f°/s  |q|_max=%7.1f°/s  |r|_max=%7.1f°/s" % (
                      nome, len(d["t"]),
                      np.rad2deg(np.max(np.abs(d["phi"]))),
                      np.rad2deg(np.max(np.abs(d["theta"]))),
                      np.rad2deg(np.max(np.abs(d["p"]))),
                      np.rad2deg(np.max(np.abs(d["q"]))),
                      np.rad2deg(np.max(np.abs(d["r"]))),
                  ))
