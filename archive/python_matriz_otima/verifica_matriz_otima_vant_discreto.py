import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import pandas as pd
import sympy as sp
import time

# --- Parâmetros do Sistema ---
# Parâmetros físicos do VANT
Ixx = 16.57e-6  # Momento de inércia em x
Iyy = 15.57e-6  # Momento de inércia em y
Izz = 29.80e-6  # Momento de inércia em z
Ir = 1.02e-7  # Momento de inércia do rotor
Omega_r = 0.0  # Velocidade angular do rotor

u1_hover = 0.0469 * 9.80665  # m * g
b_coef = 2.94e-8
d_coef = 0.05 * b_coef
L_arm = 0.060 * np.sin(np.pi / 4)

# LIMITES DE RPM (Saturação)
MAX_RPM = 26423.0
MAX_OMEGA = MAX_RPM * (2.0 * np.pi) / 60.0
MAX_OMEGA_SQ = MAX_OMEGA**2

# Condições iniciais [p, q, r, phi, theta, psi]
x0 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

# Matrizes de ponderação para a função de custo
Q = np.diag(
    [
        1.0 / ((300.0 * np.pi / 180.0) ** 2),  # p
        1.0 / ((300.0 * np.pi / 180.0) ** 2),  # q
        1.0 / ((200.0 * np.pi / 180.0) ** 2),  # r
        1.0 / ((45.0 * np.pi / 180.0) ** 2),  # phi
        1.0 / ((45.0 * np.pi / 180.0) ** 2),  # theta
        1.0 / ((90.0 * np.pi / 180.0) ** 2),  # psi
    ]
)
max_tau_roll = 2 * b_coef * L_arm * MAX_OMEGA * MAX_OMEGA / 2
max_tau_pitch = 2 * b_coef * L_arm * MAX_OMEGA * MAX_OMEGA / 2
max_tau_yaw = 2 * d_coef * MAX_OMEGA * MAX_OMEGA

R = np.diag(
    [
        1.0 / (max_tau_roll**2),
        1.0 / (max_tau_pitch**2),
        1.0 / (max_tau_yaw**2),
    ]
)


# Matriz de entrada do sistema B (6x3)
B = np.array(
    [
        [1.0 / Ixx, 0.0, 0.0],
        [0.0, 1.0 / Iyy, 0.0],
        [0.0, 0.0, 1.0 / Izz],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)

# Parâmetros de simulação
t_span = [0, 1]
Ts = 0.0051  # Período de amostragem (20 ms, igual ao main.cpp)
num_steps = int((t_span[1] - t_span[0]) / Ts) + 1
t_eval = np.linspace(t_span[0], t_span[1], num_steps)

# Valores dos parâmetros para testar
amostras = 50
extremidades = 1.0
param1_values = np.linspace(-extremidades, extremidades, amostras)  # Parâmetro alpha1
param2_values = np.linspace(-extremidades, extremidades, amostras)  # Parâmetro alpha2
param3_values = np.linspace(-extremidades, extremidades, amostras)  # Parâmetro alpha3

# --- Geração do ruído para simular vento ---
# Configuração do ruído
np.random.seed(42)  # Define uma semente para reprodutibilidade
ruido_magnitude = 0.01  # Magnitude do ruído (ajuste conforme necessário)


# Gerar ruído correlacionado temporalmente para simular vento
# Vamos usar um processo AR(1) para gerar ruído correlacionado
def gerar_ruido_vento(passos, magnitude, correlacao=0.95):
    """
    Gera ruído correlacionado temporalmente para simular vento
    passos: número de passos de tempo
    magnitude: escala do ruído
    correlacao: fator de correlação temporal (0 a 1)
    """
    ruido = np.zeros((6, passos))

    # Componentes do ruído (p, q, r são afetados pelo vento)
    for i in range(3):  # Aplica nas taxas angulares p, q, r
        ruido_atual = 0
        for t in range(passos):
            ruido_atual = correlacao * ruido_atual + (
                1 - correlacao
            ) * np.random.normal(0, 1)
            ruido[i, t] = ruido_atual * magnitude

    # Adiciona uma componente senoidal de baixa frequência para simular rajadas
    t = np.linspace(0, 2 * np.pi, passos)
    frequencias = [0.5, 0.7, 0.3]  # Diferentes frequências para cada eixo
    for i in range(3):
        ruido[i, :] += magnitude * 0.5 * np.sin(frequencias[i] * t)

    return ruido


# Gera o ruído para toda a simulação (será o mesmo para todas as combinações de parâmetros)
ruido_vento = gerar_ruido_vento(num_steps, ruido_magnitude)

# Visualizar o ruído gerado
plt.figure(figsize=(10, 6))
plt.plot(t_eval, ruido_vento[0, :], label="Ruído em p")
plt.plot(t_eval, ruido_vento[1, :], label="Ruído em q")
plt.plot(t_eval, ruido_vento[2, :], label="Ruído em r")
plt.title("Ruído simulando vento aplicado ao sistema")
plt.xlabel("Tempo (s)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.savefig("python\\outputs\\ruido_vento.png")
plt.close()

print(f"Ruído de vento gerado com magnitude: {ruido_magnitude}")

# --- Otimização: Pré-cálculo Simbólico das Matrizes Ad e Bd ---
print("Iniciando pré-cálculo simbólico das matrizes Ad e Bd...")
setup_start_time = time.time()

# Definir variáveis simbólicas para estados, parâmetros e alphas
p, q, r, phi, theta = sp.symbols("p q r phi theta")
_Ixx, _Iyy, _Izz, _Ir, _Omega_r, _Ts = sp.symbols("Ixx Iyy Izz Ir Omega_r Ts")
_alpha1, _alpha2, _alpha3 = sp.symbols("alpha1 alpha2 alpha3")

# Elementos da matriz A(x) simbólica
a12_sym = _alpha1 * ((_Iyy - _Izz) / _Ixx) * r - (_Ir * _Omega_r / _Ixx)
a13_sym = (1 - _alpha1) * ((_Iyy - _Izz) / _Ixx) * q
a21_sym = _alpha2 * ((_Izz - _Ixx) / _Iyy) * r + (_Ir * _Omega_r / _Iyy)
a23_sym = (1 - _alpha2) * ((_Izz - _Ixx) / _Iyy) * p
a31_sym = _alpha3 * ((_Ixx - _Iyy) / _Izz) * q
a32_sym = (1 - _alpha3) * ((_Ixx - _Iyy) / _Izz) * p

# Matriz A(x) simbólica
A_sym = sp.Matrix(
    [
        [0, a12_sym, a13_sym, 0, 0, 0],
        [a21_sym, 0, a23_sym, 0, 0, 0],
        [a31_sym, a32_sym, 0, 0, 0, 0],
        [1, sp.sin(phi) * sp.tan(theta), sp.cos(phi) * sp.tan(theta), 0, 0, 0],
        [0, sp.cos(phi), -sp.sin(phi), 0, 0, 0],
        [0, sp.sin(phi) / sp.cos(theta), sp.cos(phi) / sp.cos(theta), 0, 0, 0],
    ]
)

# Matrizes simbólicas para discretização
B_sym = sp.Matrix(B)
I_sym = sp.eye(6)

# Discretização simbólica (aproximação de Taylor de 2ª ordem)
Ad_sym = I_sym + A_sym * _Ts + (A_sym @ A_sym * _Ts**2) / 2
Bd_sym = (I_sym * _Ts + (A_sym * _Ts**2) / 2) @ B_sym

# Lista de variáveis na ordem correta para as funções
vars_list = [
    p,
    q,
    r,
    phi,
    theta,
    _alpha1,
    _alpha2,
    _alpha3,
    _Ixx,
    _Iyy,
    _Izz,
    _Ir,
    _Omega_r,
    _Ts,
]

# Criar funções numéricas rápidas a partir das expressões simbólicas
Ad_func = sp.lambdify(vars_list, Ad_sym, "numpy")
Bd_func = sp.lambdify(vars_list, Bd_sym, "numpy")

setup_time = time.time() - setup_start_time
print(f"Pré-cálculo concluído em {setup_time:.4f} segundos.")

# --- Simulação e Geração de Gráficos ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
results = {}

total_simulations = len(param1_values) * len(param2_values) * len(param3_values)
print("Iniciando simulações do controle SDRE Discreto para sistema 6x6...")
print(f"Total de simulações: {total_simulations}")
print(f"Período de amostragem: {Ts} segundos")

param_combinations = list(
    itertools.product(param1_values, param2_values, param3_values)
)

min_cost_so_far = float("inf")
best_params_so_far = None

with tqdm(param_combinations, desc="Simulações", unit="sim", position=0) as pbar:
    for param1, param2, param3 in pbar:
        pbar.set_description(f"α1={param1:.2f}, α2={param2:.2f}, α3={param3:.2f}")

        max_real_eigenvalue_for_sim = [-np.inf]
        eigenvalues_over_time = []

        states = np.zeros((7, num_steps))
        states[:6, 0] = x0
        states[6, 0] = 0.0

        Omega_r_k = 0.0  # Velocidade angular residual inicial

        for k in range(1, num_steps):
            x = states[:6, k - 1]
            pk, qk, rk, phik, thetak, psik = x

            # Evitar instabilidade em tan(theta) e 1/cos(theta)
            if abs(np.cos(thetak)) < 1e-6:
                thetak = np.sign(thetak) * (np.pi / 2 - 1e-6)

            # Argumentos para as funções pré-calculadas
            func_args = (
                pk,
                qk,
                rk,
                phik,
                thetak,
                param1,
                param2,
                param3,
                Ixx,
                Iyy,
                Izz,
                Ir,
                Omega_r_k,
                Ts,
            )

            # Discretização usando as funções pré-calculadas (MUITO MAIS RÁPIDO)
            A_d = Ad_func(*func_args)
            B_d = Bd_func(*func_args)

            Q_d = Q * Ts
            R_d = R

            try:
                P = solve_discrete_are(A_d, B_d, Q_d, R_d)
            except (np.linalg.LinAlgError, ValueError):
                eigenvalues_over_time.append(0.0)
                states[:, k] = states[:, k - 1]
                continue

            K = np.linalg.inv(B_d.T @ P @ B_d + R_d) @ B_d.T @ P @ A_d
            u = -K @ x.reshape(-1, 1)

            # --- Atualiza o Omega_r_k para o próximo passo considerando os torques gerados ---
            u2, u3, u4 = u[0, 0], u[1, 0], u[2, 0]

            # Equações para encontrar a rotação ao quadrado de cada motor (w_i^2)
            w1_sq = (
                u1_hover / (4 * b_coef) - u3 / (2 * L_arm * b_coef) - u4 / (4 * d_coef)
            )
            w2_sq = (
                u1_hover / (4 * b_coef) - u2 / (2 * L_arm * b_coef) + u4 / (4 * d_coef)
            )
            w3_sq = (
                u1_hover / (4 * b_coef) + u3 / (2 * L_arm * b_coef) - u4 / (4 * d_coef)
            )
            w4_sq = (
                u1_hover / (4 * b_coef) + u2 / (2 * L_arm * b_coef) + u4 / (4 * d_coef)
            )

            # Extrai as velocidades angulares individuais reais (rad/s) limitando a zeros e maximos permitidos
            w1 = np.sqrt(max(0.0, w1_sq))
            w2 = np.sqrt(max(0.0, w2_sq))
            w3 = np.sqrt(max(0.0, w3_sq))
            w4 = np.sqrt(max(0.0, w4_sq))

            # A velocidade residual do rotor é dada pela equação citada (Voos 2006)
            Omega_r_k = -w1 + w2 - w3 + w4
            # ----------------------------------------------------------------------------------

            eigenvalues = np.linalg.eigvals(A_d - B_d @ K)
            current_max_eig = np.max(np.abs(eigenvalues))
            if current_max_eig > max_real_eigenvalue_for_sim[0]:
                max_real_eigenvalue_for_sim[0] = current_max_eig

            eigenvalues_over_time.append(current_max_eig)

            x_next = (
                A_d @ x.reshape(-1, 1) + B_d @ u + 0 * ruido_vento[:, k].reshape(-1, 1)
            )
            states[:6, k] = x_next.flatten()

            cost_increment = (x.T @ Q @ x + u.T @ R @ u.reshape(-1, 1))[0, 0] * Ts
            states[6, k] = states[6, k - 1] + cost_increment

        final_cost = states[6, -1]
        key = (param1, param2, param3)
        results[key] = {
            "t": t_eval,
            "states": states[:6, :],
            "cost": final_cost,
            "max_eig": max_real_eigenvalue_for_sim[0],
            "eigenvalues_time": np.array(eigenvalues_over_time),
        }

        if final_cost < min_cost_so_far:
            min_cost_so_far = final_cost
            best_params_so_far = key

        pbar.set_postfix(
            J=f"{final_cost:.8f}",
            λ_max=f"{max_real_eigenvalue_for_sim[0]:.4f}",
            min_J=f"{min_cost_so_far:.8f}",
        )

# --- Exportação para Excel e Geração de Gráficos ---
if results:
    print("\n--- Processando resultados e exportando para Excel ---")
    data = []
    for params, result in results.items():
        data.append(
            {
                "param1_alpha1": params[0],
                "param2_alpha2": params[1],
                "param3_alpha3": params[2],
                "custo_total": result["cost"],
                "maior_autovalor_abs": result["max_eig"],
            }
        )
    df = pd.DataFrame(data)

    # --- CRIAÇÃO DA MÉTRICA DE OTIMIZAÇÃO MULTIOBJETIVO ---

    # 1. Normaliza Custo e Autovalor no intervalo [0, 1]
    # (0 sendo o melhor possível encontrado e 1 a pior simulação)
    df["custo_norm"] = (df["custo_total"] - df["custo_total"].min()) / (
        df["custo_total"].max() - df["custo_total"].min() + 1e-9
    )
    df["autovalor_norm"] = (
        df["maior_autovalor_abs"] - df["maior_autovalor_abs"].min()
    ) / (df["maior_autovalor_abs"].max() - df["maior_autovalor_abs"].min() + 1e-9)

    # 2. Cria score global ponderado: 50% de peso pro custo e 50% pra robustez (autovalor)
    # Quanto menor o score, melhor/mais equilibrada é a combinação
    df["score_global"] = (0.5 * df["custo_norm"]) + (0.5 * df["autovalor_norm"])

    # Adicionando explicitamente a media geométrica solicitada para fins comparativos
    df["media_geometrica"] = np.sqrt(df["custo_total"] * df["maior_autovalor_abs"])

    # Ordena pelo melhor score (o menor possível) em vez de apenas custo
    df = df.sort_values("score_global")

    melhor_config = df.iloc[0]
    print("\n" + "=" * 60)
    print("🏆 MELHOR CONFIGURAÇÃO ENCONTRADA (EQUILÍBRIO CUSTO x ESTABILIDADE)")
    print(f"Alpha 1: {melhor_config['param1_alpha1']:.4f}")
    print(f"Alpha 2: {melhor_config['param2_alpha2']:.4f}")
    print(f"Alpha 3: {melhor_config['param3_alpha3']:.4f}")
    print(
        f"\nCusto Total: {melhor_config['custo_total']:.8f} (Mínimo foi {df['custo_total'].min():.8f})"
    )
    print(
        f"Maior Autovalor: {melhor_config['maior_autovalor_abs']:.4f} (Mínimo foi {df['maior_autovalor_abs'].min():.4f})"
    )
    print(f"Score Global (0=Perfeito, 1=Pior): {melhor_config['score_global']:.6f}")
    print(f"Média Geométrica (comparativo): {melhor_config['media_geometrica']:.6f}")
    print("=" * 60 + "\n")

    excel_filename = "python\\outputs\\resultados_simulacao_sdre_discreto.xlsx"
    try:
        df.to_excel(excel_filename, index=False, sheet_name="Resultados")
        print(f"Arquivo Excel salvo como: {excel_filename}")
    except Exception as e:
        print(f"Erro ao exportar para Excel: {e}")
        csv_filename = "resultados_simulacao_sdre_discreto.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Arquivo CSV salvo como alternativa: {csv_filename}")

    # --- NOVOS GRÁFICOS: Custo e Autovalores ---
    print("\n--- Gerando gráficos de custo e autovalores ---")

    # Função para converter alphas em cores RGB
    def alphas_to_rgb(alpha1, alpha2, alpha3, min_alpha, max_alpha):
        """
        Converte valores de alpha em cores RGB
        alpha1 -> R (vermelho)
        alpha2 -> G (verde)
        alpha3 -> B (azul)
        """
        # Normaliza os valores de alpha para o intervalo [0, 1]
        r = (alpha1 - min_alpha) / (max_alpha - min_alpha)
        g = (alpha2 - min_alpha) / (max_alpha - min_alpha)
        b = (alpha3 - min_alpha) / (max_alpha - min_alpha)
        return (r, g, b)

    # Calcula os valores mínimos e máximos de alpha
    min_alpha = min(param1_values.min(), param2_values.min(), param3_values.min())
    max_alpha = max(param1_values.max(), param2_values.max(), param3_values.max())

    # Gera cores RGB para cada combinação de parâmetros
    cores_rgb = [
        alphas_to_rgb(
            row["param1_alpha1"],
            row["param2_alpha2"],
            row["param3_alpha3"],
            min_alpha,
            max_alpha,
        )
        for _, row in df.iterrows()
    ]

    # Gráfico 1: Distribuição dos Custos
    fig_custos, ax_custos = plt.subplots(figsize=(14, 7))
    custos = df["custo_total"].values
    indices = np.arange(len(custos))

    ax_custos.bar(
        indices, custos, color=cores_rgb, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax_custos.set_xlabel("Índice da Simulação (ordenado por custo)", fontsize=12)
    ax_custos.set_ylabel("Custo Total (J)", fontsize=12)
    ax_custos.set_title(
        "Distribuição dos Custos Totais para Diferentes Combinações de Parâmetros\n(Cores: R=α1, G=α2, B=α3)",
        fontsize=14,
        fontweight="bold",
    )
    ax_custos.grid(True, alpha=0.3)

    # Adicionar linha horizontal para o custo médio
    custo_medio = np.mean(custos)
    ax_custos.axhline(
        y=custo_medio,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Custo Médio: {custo_medio:.6f}",
    )
    ax_custos.legend(fontsize=10)

    # Adicionar legenda de cores
    from matplotlib.patches import Rectangle

    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            fc=(1, 0, 0),
            label=f"α1: {min_alpha:.0f} (preto) → {max_alpha:.0f} (vermelho)",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fc=(0, 1, 0),
            label=f"α2: {min_alpha:.0f} (preto) → {max_alpha:.0f} (verde)",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fc=(0, 0, 1),
            label=f"α3: {min_alpha:.0f} (preto) → {max_alpha:.0f} (azul)",
        ),
    ]
    ax_custos.legend(
        handles=legend_elements, loc="upper right", fontsize=9, title="Mapeamento RGB"
    )

    plt.tight_layout()
    plt.savefig(
        "python\\outputs\\distribuicao_custos.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Gráfico de custos salvo: python\\outputs\\distribuicao_custos.png")

    # Gráfico 2: Distribuição dos Autovalores Máximos
    fig_autovalores, ax_autovalores = plt.subplots(figsize=(14, 7))
    autovalores = df["maior_autovalor_abs"].values

    ax_autovalores.bar(
        indices,
        autovalores,
        color=cores_rgb,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_autovalores.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="Limite de Estabilidade (|λ| = 1)",
    )
    ax_autovalores.set_xlabel("Índice da Simulação (ordenado por custo)", fontsize=12)
    ax_autovalores.set_ylabel("Maior Autovalor Absoluto (|λ|_max)", fontsize=12)
    ax_autovalores.set_title(
        "Distribuição dos Maiores Autovalores do Sistema em Malha Fechada\n(Cores: R=α1, G=α2, B=α3)",
        fontsize=14,
        fontweight="bold",
    )
    ax_autovalores.grid(True, alpha=0.3)

    # Adicionar texto informativo
    n_estaveis = sum(1 for av in autovalores if av < 1.0)
    n_total = len(autovalores)
    ax_autovalores.text(
        0.02,
        0.98,
        f"Sistemas Estáveis: {n_estaveis}/{n_total} ({100*n_estaveis/n_total:.1f}%)",
        transform=ax_autovalores.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    # Adicionar legenda de cores
    ax_autovalores.legend(
        handles=legend_elements
        + [Rectangle((0, 0), 1, 1, fc="red", label="Limite de Estabilidade")],
        loc="upper right",
        fontsize=9,
        title="Mapeamento RGB",
    )

    plt.tight_layout()
    plt.savefig(
        "python\\outputs\\distribuicao_autovalores.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Gráfico de autovalores salvo: python\\outputs\\distribuicao_autovalores.png")

    # Gráfico 3: Relação entre Custo e Autovalor Máximo
    fig_relacao, ax_relacao = plt.subplots(figsize=(11, 9))
    scatter = ax_relacao.scatter(
        autovalores,
        custos,
        c=cores_rgb,
        s=80,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.8,
    )
    ax_relacao.set_xlabel("Maior Autovalor Absoluto (|λ|_max)", fontsize=12)
    ax_relacao.set_ylabel("Custo Total (J)", fontsize=12)
    ax_relacao.set_title(
        "Relação entre Custo Total e Estabilidade do Sistema\n(Cores: R=α1, G=α2, B=α3)",
        fontsize=14,
        fontweight="bold",
    )
    """
    ax_relacao.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=2.5,
        alpha=0.7,
        label="Limite de Estabilidade",
    )
    """
    ax_relacao.grid(True, alpha=0.3)

    # Adicionar legenda de cores
    ax_relacao.legend(
        handles=legend_elements
        + [Rectangle((0, 0), 1, 1, fc="red", label="Limite de Estabilidade")],
        loc="best",
        fontsize=9,
        title="Mapeamento RGB",
    )

    plt.tight_layout()
    plt.savefig(
        "python\\outputs\\relacao_custo_autovalor.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        "Gráfico de relação custo-autovalor salvo: python\\outputs\\relacao_custo_autovalor.png"
    )

    # Gráfico 4: Histograma de Custos com cores RGB
    fig_hist_custo, ax_hist_custo = plt.subplots(figsize=(11, 7))

    # Criar bins e colorir cada barra
    n_bins = 30
    counts, bin_edges, patches = ax_hist_custo.hist(
        custos, bins=n_bins, edgecolor="black", linewidth=0.5
    )

    # Colorir cada barra do histograma com a cor média das simulações naquele bin
    for i, patch in enumerate(patches):
        # Encontra quais simulações estão neste bin
        mask = (custos >= bin_edges[i]) & (custos < bin_edges[i + 1])
        if i == len(patches) - 1:  # Último bin inclui o valor máximo
            mask = (custos >= bin_edges[i]) & (custos <= bin_edges[i + 1])

        if np.any(mask):
            # Calcula a cor média das simulações neste bin
            cores_no_bin = [cores_rgb[j] for j in range(len(cores_rgb)) if mask[j]]
            if cores_no_bin:
                cor_media = np.mean(cores_no_bin, axis=0)
                patch.set_facecolor(cor_media)
                patch.set_alpha(0.7)

    ax_hist_custo.axvline(
        x=custo_medio,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Média: {custo_medio:.6f}",
    )
    ax_hist_custo.axvline(
        x=np.median(custos),
        color="blue",
        linestyle="--",
        linewidth=2.5,
        label=f"Mediana: {np.median(custos):.6f}",
    )
    ax_hist_custo.set_xlabel("Custo Total (J)", fontsize=12)
    ax_hist_custo.set_ylabel("Frequência", fontsize=12)
    ax_hist_custo.set_title(
        "Histograma dos Custos Totais\n(Cores médias por bin: R=α1, G=α2, B=α3)",
        fontsize=14,
        fontweight="bold",
    )
    ax_hist_custo.legend(fontsize=10)
    ax_hist_custo.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("python\\outputs\\histograma_custos.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Histograma de custos salvo: python\\outputs\\histograma_custos.png")

    # Gráfico 5: Histograma de Autovalores com cores RGB
    fig_hist_auto, ax_hist_auto = plt.subplots(figsize=(11, 7))

    # Criar bins e colorir cada barra
    counts, bin_edges, patches = ax_hist_auto.hist(
        autovalores, bins=n_bins, edgecolor="black", linewidth=0.5
    )

    # Colorir cada barra do histograma
    for i, patch in enumerate(patches):
        mask = (autovalores >= bin_edges[i]) & (autovalores < bin_edges[i + 1])
        if i == len(patches) - 1:
            mask = (autovalores >= bin_edges[i]) & (autovalores <= bin_edges[i + 1])

        if np.any(mask):
            cores_no_bin = [cores_rgb[j] for j in range(len(cores_rgb)) if mask[j]]
            if cores_no_bin:
                cor_media = np.mean(cores_no_bin, axis=0)
                patch.set_facecolor(cor_media)
                patch.set_alpha(0.7)

    ax_hist_auto.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="Limite de Estabilidade",
    )
    ax_hist_auto.axvline(
        x=np.mean(autovalores),
        color="blue",
        linestyle="--",
        linewidth=2.5,
        label=f"Média: {np.mean(autovalores):.4f}",
    )
    ax_hist_auto.set_xlabel("Maior Autovalor Absoluto (|λ|_max)", fontsize=12)
    ax_hist_auto.set_ylabel("Frequência", fontsize=12)
    ax_hist_auto.set_title(
        "Histograma dos Maiores Autovalores\n(Cores médias por bin: R=α1, G=α2, B=α3)",
        fontsize=14,
        fontweight="bold",
    )
    ax_hist_auto.legend(fontsize=10)
    ax_hist_auto.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "python\\outputs\\histograma_autovalores.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        "Histograma de autovalores salvo: python\\outputs\\histograma_autovalores.png"
    )

    # Gráfico 6: Visualização 3D da distribuição de parâmetros
    from mpl_toolkits.mplot3d import Axes3D

    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    alpha1_vals = df["param1_alpha1"].values
    alpha2_vals = df["param2_alpha2"].values
    alpha3_vals = df["param3_alpha3"].values
    custos_vals = df["custo_total"].values

    scatter_3d = ax_3d.scatter(
        alpha1_vals,
        alpha2_vals,
        alpha3_vals,
        c=cores_rgb,
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    ax_3d.set_xlabel("α1 (Vermelho)", fontsize=11, labelpad=10)
    ax_3d.set_ylabel("α2 (Verde)", fontsize=11, labelpad=10)
    ax_3d.set_zlabel("α3 (Azul)", fontsize=11, labelpad=10)
    ax_3d.set_title(
        "Distribuição 3D dos Parâmetros\n(Tamanho proporcional ao custo)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(
        "python\\outputs\\distribuicao_3d_parametros.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Gráfico 3D salvo: python\\outputs\\distribuicao_3d_parametros.png")
else:
    print("Nenhum resultado válido obtido nas simulações.")
    # Cria um DataFrame vazio mesmo sem resultados
    df_empty = pd.DataFrame(
        columns=["param1_alpha1", "param2_alpha2", "param3_alpha3", "custo_total"]
    )
    df_empty.to_excel("resultados_simulacao_sdre_discreto_vazio.xlsx", index=False)
    print("Arquivo Excel vazio criado: resultados_simulacao_sdre_discreto_vazio.xlsx")
