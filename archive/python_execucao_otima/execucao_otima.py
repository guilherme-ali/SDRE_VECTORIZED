# ESTE CÓDIGO IMPLEMENTA E COMPARA DUAS ABORDAGENS PARA SIMULAÇÃO DE CONTROLE ÓTIMO
# DE UM SISTEMA DINÂMICO NÃO LINEAR (QUADRICÓPTERO) USANDO O MÉTODO SDRE.
# A PRIMEIRA ABORDAGEM UTILIZA CÁLCULO SIMBÓLICO PRÉVIO PARA GERAR FUNÇÕES DE MATRIZES
# DISCRETAS, ENQUANTO A SEGUNDA CALCULA AS MATRIZES NUMERICAMENTE EM TEMPO REAL.

import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# --- Definição dos Parâmetros do Sistema ---
# Momentos de inércia (valores de exemplo para um quadricóptero)
Ixx = 0.082
Iyy = 0.082
Izz = 0.149
Ir = 0.003  # Inércia do rotor
Omega_r = 0  # Velocidade do rotor (rad/s)

# Parâmetros de simulação
Ts = 0.01  # Período de amostragem (10ms)
T_sim = 10.0  # Tempo total de simulação (10s)
N = int(T_sim / Ts)  # Número de passos

# Estado inicial x = [p, q, r, phi, theta, psi]
x0 = np.array([0, 0, 0, np.deg2rad(10), np.deg2rad(-15), np.deg2rad(5)])

# Matrizes de ponderação para o controlador LQR (SDRE)
# Q penaliza o desvio do estado, R penaliza o esforço de controle
Q = np.diag([10, 10, 10, 20, 20, 5])
R = np.diag([0.1, 0.1, 0.1])

# Matriz B (constante)
B_cont = np.array([
    [1/Ixx, 0, 0],
    [0, 1/Iyy, 0],
    [0, 0, 1/Izz],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

# --- Método 1: Abordagem Algébrica (Pré-cálculo com SymPy) ---
def run_simulation_algebraic():
    """
    Executa a simulação usando funções pré-calculadas para Ad e Bd.
    """
    print("--- Iniciando Simulação com Método Algébrico (Pré-cálculo) ---")
    
    # Início da medição de tempo de setup
    setup_start_time = time.time()

    # Definir variáveis simbólicas
    p, q, r, phi, theta = sp.symbols('p q r phi theta')
    _Ixx, _Iyy, _Izz, _Ir, _Omega_r, _Ts = sp.symbols('Ixx Iyy Izz Ir Omega_r Ts')
    
    # Definir os elementos a_ij com alpha = 0
    a12 = -_Ir * _Omega_r / _Ixx
    a13 = (_Iyy - _Izz) / _Ixx * q
    a21 = _Ir * _Omega_r / _Iyy
    a23 = (_Izz - _Ixx) / _Iyy * p
    a31 = (_Ixx - _Iyy) / _Izz * q
    a32 = (_Ixx - _Iyy) / _Izz * p

    # Construir a matriz A(x) simbólica
    A_sym = sp.Matrix([
        [0, a12, a13, 0, 0, 0],
        [a21, 0, a23, 0, 0, 0],
        [a31, a32, 0, 0, 0, 0],
        [1, sp.sin(phi) * sp.tan(theta), sp.cos(phi) * sp.tan(theta), 0, 0, 0],
        [0, sp.cos(phi), -sp.sin(phi), 0, 0, 0],
        [0, sp.sin(phi) / sp.cos(theta), sp.cos(phi) / sp.cos(theta), 0, 0, 0]
    ])

    # Matriz B simbólica (é constante)
    B_sym = sp.Matrix(B_cont)
    I_sym = sp.eye(6)

    # Calcular Ad e Bd simbólicos usando a expansão de Taylor de 2ª ordem
    Ad_sym = I_sym + A_sym * _Ts + (A_sym @ A_sym * _Ts**2) / 2
    Bd_sym = (I_sym * _Ts + (A_sym * _Ts**2) / 2) @ B_sym

    # Usar sp.lambdify para converter as matrizes simbólicas em funções numéricas rápidas
    # Note que psi não está em A(x), então não é um argumento
    vars_list = [p, q, r, phi, theta, _Ixx, _Iyy, _Izz, _Ir, _Omega_r, _Ts]
    
    Ad_func = sp.lambdify(vars_list, Ad_sym, 'numpy')
    Bd_func = sp.lambdify(vars_list, Bd_sym, 'numpy')
    
    setup_end_time = time.time()
    setup_time = setup_end_time - setup_start_time
    print(f"Tempo de Setup (Pré-cálculo): {setup_time:.6f} segundos")

    # --- Loop de Simulação ---
    sim_start_time = time.time()
    
    x_hist = np.zeros((6, N + 1))
    u_hist = np.zeros((3, N))
    x_hist[:, 0] = x0
    x = x0.copy()

    params = (Ixx, Iyy, Izz, Ir, Omega_r, Ts)

    for k in range(N):
        # Desempacotar estados para a função
        pk, qk, rk, phik, thetak, _ = x

        # 1. Calcular Ad e Bd usando as funções pré-calculadas
        Ad = Ad_func(pk, qk, rk, phik, thetak, *params)
        Bd = Bd_func(pk, qk, rk, phik, thetak, *params)

        # 2. Resolver a Equação de Riccati Discreta (DARE)
        P = solve_discrete_are(Ad, Bd, Q, R)

        # 3. Calcular o ganho de controle ótimo K
        K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

        # 4. Calcular a entrada de controle
        u = -K @ x
        u_hist[:, k] = u

        # 5. Atualizar o estado do sistema (usando o modelo discretizado)
        x = Ad @ x + Bd @ u
        x_hist[:, k + 1] = x
        
    sim_end_time = time.time()
    sim_time = sim_end_time - sim_start_time
    print(f"Tempo de Simulação (Loop): {sim_time:.6f} segundos")
    print(f"Tempo Total (Setup + Simulação): {setup_time + sim_time:.6f} segundos")
    
    return x_hist, u_hist, setup_time + sim_time

# --- Método 2: Abordagem Numérica (Cálculo em Tempo Real) ---
def run_simulation_numerical():
    """
    Executa a simulação calculando Ad e Bd numericamente a cada passo.
    """
    print("\n--- Iniciando Simulação com Método Numérico (Em Tempo Real) ---")
    
    setup_start_time = time.time()
    # Praticamente nenhum setup é necessário aqui
    I_mat = np.eye(6)
    
    def get_continuous_A(x_vec):
        p, q, r, phi, theta, _ = x_vec
        
        # Elementos a_ij com alpha = 0
        a12 = -Ir * Omega_r / Ixx
        a13 = (Iyy - Izz) / Ixx * q
        a21 = Ir * Omega_r / Iyy
        a23 = (Izz - Ixx) / Iyy * p
        a31 = (Ixx - Iyy) / Izz * q
        a32 = (Ixx - Iyy) / Izz * p
        
        # Termos trigonométricos
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        tan_theta = np.tan(theta)
        sec_theta = 1 / np.cos(theta)

        A = np.array([
            [0, a12, a13, 0, 0, 0],
            [a21, 0, a23, 0, 0, 0],
            [a31, a32, 0, 0, 0, 0],
            [1, sin_phi * tan_theta, cos_phi * tan_theta, 0, 0, 0],
            [0, cos_phi, -sin_phi, 0, 0, 0],
            [0, sin_phi * sec_theta, cos_phi * sec_theta, 0, 0, 0]
        ])
        return A

    setup_end_time = time.time()
    setup_time = setup_end_time - setup_start_time
    print(f"Tempo de Setup: {setup_time:.6f} segundos")
    
    # --- Loop de Simulação ---
    sim_start_time = time.time()
    
    x_hist = np.zeros((6, N + 1))
    u_hist = np.zeros((3, N))
    x_hist[:, 0] = x0
    x = x0.copy()

    for k in range(N):
        # 1. Obter a matriz A(x) contínua
        Ac = get_continuous_A(x)
        
        # Calcular Ad e Bd numericamente
        Ac_sq = Ac @ Ac
        Ad = I_mat + Ac * Ts + (Ac_sq * Ts**2) / 2
        Bd = (I_mat * Ts + (Ac * Ts**2) / 2) @ B_cont

        # 2. Resolver DARE
        P = solve_discrete_are(Ad, Bd, Q, R)

        # 3. Calcular ganho K
        K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

        # 4. Calcular controle u
        u = -K @ x
        u_hist[:, k] = u

        # 5. Atualizar estado
        x = Ad @ x + Bd @ u
        x_hist[:, k + 1] = x

    sim_end_time = time.time()
    sim_time = sim_end_time - sim_start_time
    print(f"Tempo de Simulação (Loop): {sim_time:.6f} segundos")
    print(f"Tempo Total (Setup + Simulação): {setup_time + sim_time:.6f} segundos")

    return x_hist, u_hist, setup_time + sim_time

# --- Execução Principal e Plotagem ---
if __name__ == '__main__':
    x_hist_alg, u_hist_alg, total_time_alg = run_simulation_algebraic()
    x_hist_num, u_hist_num, total_time_num = run_simulation_numerical()

    print("\n" + "="*50)
    print("           RESULTADO DA COMPARAÇÃO DE VELOCIDADE")
    print("="*50)
    print(f"Tempo Total - Método Algébrico (Pré-cálculo): {total_time_alg:.6f} s")
    print(f"Tempo Total - Método Numérico (Em Tempo Real): {total_time_num:.6f} s")
    
    if total_time_alg < total_time_num:
        print("\n🏆 Vencedor: O Método Algébrico (Pré-cálculo) foi mais rápido.")
    else:
        print("\n🏆 Vencedor: O Método Numérico (Em Tempo Real) foi mais rápido.")
    print("="*50 + "\n")


    # Plotagem dos resultados (usando os resultados de um dos métodos, pois são idênticos)
    time_vec = np.linspace(0, T_sim, N + 1)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle('Simulação do Controle SDRE - Evolução dos Estados', fontsize=16)

    labels_rad = ['p (rad/s)', 'q (rad/s)', 'r (rad/s)']
    labels_deg = [r'$\phi$ (graus)', r'$\theta$ (graus)', r'$\psi$ (graus)']

    # Plot de velocidades angulares
    for i in range(3):
        axes[i, 0].plot(time_vec, x_hist_alg[i, :], label=labels_rad[i])
        axes[i, 0].set_ylabel(labels_rad[i])
        axes[i, 0].grid(True)
        axes[i, 0].legend()

    # Plot de ângulos de Euler (convertidos para graus para melhor visualização)
    for i in range(3):
        axes[i, 1].plot(time_vec, np.rad2deg(x_hist_alg[i+3, :]), label=labels_deg[i])
        axes[i, 1].set_ylabel(labels_deg[i])
        axes[i, 1].grid(True)
        axes[i, 1].legend()

    axes[2, 0].set_xlabel('Tempo (s)')
    axes[2, 1].set_xlabel('Tempo (s)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()