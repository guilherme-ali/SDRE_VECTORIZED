import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import time

# --- Parâmetros do Sistema ---
# Parâmetros físicos do VANT
Ixx = 0.00184  # Momento de inércia em x
Iyy = 0.00225  # Momento de inércia em y  
Izz = 0.00338  # Momento de inércia em z
Ir = 0.00001   # Momento de inércia do rotor
Omega_r = 0.0  # Velocidade angular do rotor

# Funções trigonométricas
def C_phi(phi):
    return np.cos(phi)

def S_phi(phi):
    return np.sin(phi)

def C_theta(theta):
    return np.cos(theta)

# Condições iniciais [p, q, r, phi, theta, psi]
x0 = [0.0, 0.0, 0.0, 1, 0.0, 0.0]

# Matrizes de ponderação para a função de custo
Q = np.diag([1, 1, 1, 50.0, 50.0, 50.0])  # 6x6
R = np.eye(3)*0.1  # 3x3

# Matriz de entrada do sistema B (6x3)
B = np.array([[1.0/Ixx, 0.0, 0.0],
              [0.0, 1.0/Iyy, 0.0],
              [0.0, 0.0, 1.0/Izz],
              [0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]])

# Parâmetros de simulação
t_span = [0, 1]
Ts = 0.01  # Período de amostragem
num_steps = int((t_span[1] - t_span[0]) / Ts) + 1
t_eval = np.linspace(t_span[0], t_span[1], num_steps)

# Função de simulação e cálculo de custo para os parâmetros fornecidos
def simulate_and_get_cost(params):
    param1, param2, param3 = params
    
    # Inicializa vetor de estados e custo
    states = np.zeros((7, num_steps))  # 6 estados + custo
    states[:6, 0] = x0
    states[6, 0] = 0.0  # Custo inicial
    
    # Simulação discretizada
    for k in range(1, num_steps):
        # Estado atual
        x = states[:6, k-1].reshape(-1, 1)
        p, q, r, phi, theta, psi = x.flatten()
        
        # Calcula as funções trigonométricas
        s_phi = S_phi(phi)
        c_phi = C_phi(phi)
        c_theta = C_theta(theta)
        
        # Evitar divisão por zero para tan(theta) e 1/cos(theta)
        if abs(c_theta) < 1e-6:
            c_theta = 1e-6 * np.sign(c_theta)
        t_theta = np.sin(theta) / c_theta
        
        # Matriz A(x) contínua
        # Linha 1: p_dot
        a12 = param1 * ((Iyy - Izz) / Ixx) * r - (Ir * Omega_r / Ixx) 
        a13 = (1 - param1) * ((Iyy - Izz) / Ixx) * q

        # Linha 2: q_dot
        a21 = param2 * ((Izz - Ixx) / Iyy) * r + (Ir * Omega_r / Iyy)
        a23 = (1 - param2) * ((Izz - Ixx) / Iyy) * p

        # Linha 3: r_dot
        a31 = param3 * ((Ixx - Iyy) / Izz) * q
        a32 = (1 - param3) * ((Ixx - Iyy) / Izz) * p
        
        A_c = np.array([
            [0.0, a12, a13, 0.0, 0.0, 0.0],
            [a21, 0.0, a23, 0.0, 0.0, 0.0],
            [a31, a32, 0.0, 0.0, 0.0, 0.0],
            [1.0, s_phi * t_theta, c_phi * t_theta, 0.0, 0.0, 0.0],
            [0.0, c_phi, -s_phi, 0.0, 0.0, 0.0],
            [0.0, s_phi / c_theta, c_phi / c_theta, 0.0, 0.0, 0.0]
        ])
        
        # Discretização do sistema (aproximação de segunda ordem - Taylor)
        A_d = np.eye(6) + A_c * Ts + (A_c @ A_c * Ts**2) / 2.0
        B_d = (np.eye(6) * Ts + A_c * Ts**2 / 2.0) @ B
        
        # Ajuste das matrizes de custo para o sistema discreto
        Q_d = Q * Ts
        R_d = R
        
        # Tenta resolver a Equação de Riccati discreta
        try:
            P = solve_discrete_are(A_d, B_d, Q_d, R_d)
        except (np.linalg.LinAlgError, ValueError):
            # Penalização para soluções que não convergem
            return 1e10
            
        # Calcula a entrada de controle SDRE discreta
        K = np.linalg.inv(B_d.T @ P @ B_d + R_d) @ B_d.T @ P @ A_d
        u = -K @ x
        
        # Próximo estado usando o modelo discreto
        x_next = A_d @ x + B_d @ u
        states[:6, k] = x_next.flatten()
        
        # Calcula o custo incremental
        cost_increment = (x.T @ Q @ x + u.T @ R @ u)[0, 0] * Ts
        states[6, k] = states[6, k-1] + cost_increment
    
    # Retorna o custo final
    return states[6, -1]

# Função para otimização que imprime o progresso
def objective_function(params):
    cost = simulate_and_get_cost(params)
    print(f"Parâmetros: α₁={params[0]:.4f}, α₂={params[1]:.4f}, α₃={params[2]:.4f}, Custo: {cost:.8f}")
    return cost

# --- Algoritmo de Busca ---
def busca_parametros_otimos():
    print("Iniciando busca de parâmetros ótimos para o controle SDRE Discreto...")
    
    # 1. Primeiro método: Nelder-Mead (simplex)
    print("\n=== Método 1: Otimização Nelder-Mead ===")
    start_time = time.time()
    
    # Ponto inicial
    initial_guess = [0.0, 0.0, 0.0]
    
    # Limites dos parâmetros
    bounds = [(-10000, 10000), (-10000, 10000), (-10000, 10000)]
      
    # Executa otimização
    result = minimize(
        objective_function, 
        initial_guess,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 10000000, 'disp': True}
    )
    
    # Resultados da otimização
    optimal_params = result.x
    optimal_cost = result.fun
    
    print(f"\nTempo de execução: {time.time() - start_time:.2f} segundos")
    print(f"Melhor solução encontrada:")
    print(f"α₁ = {optimal_params[0]:.6f}")
    print(f"α₂ = {optimal_params[1]:.6f}")
    print(f"α₃ = {optimal_params[2]:.6f}")
    print(f"Custo mínimo: {optimal_cost:.8f}")
    
    # 2. Segundo método: Busca em grade (para comparação)
    print("\n=== Método 2: Busca em Grade Refinada ===")
    start_time = time.time()
    
    # Cria uma busca em grade mais refinada em torno da solução encontrada
    margin = 1.0  # Margem para refinamento
    amostras_refinadas = 10
    
    param1_values = np.linspace(max(-10, optimal_params[0] - margin), 
                               min(10, optimal_params[0] + margin), 
                               amostras_refinadas)
    param2_values = np.linspace(max(-10, optimal_params[1] - margin), 
                               min(10, optimal_params[1] + margin), 
                               amostras_refinadas)
    param3_values = np.linspace(max(-10, optimal_params[2] - margin), 
                               min(10, optimal_params[2] + margin), 
                               amostras_refinadas)
    
    best_cost = float('inf')
    best_params = None
    
    total_combinations = len(param1_values) * len(param2_values) * len(param3_values)
    counter = 0
    
    print(f"Executando busca em grade refinada com {total_combinations} combinações...")
    
    # Loop para todas as combinações
    for p1 in param1_values:
        for p2 in param2_values:
            for p3 in param3_values:
                counter += 1
                params = [p1, p2, p3]
                cost = simulate_and_get_cost(params)
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = params
                
                # Imprime progresso
                if counter % 10 == 0 or counter == total_combinations:
                    print(f"Progresso: {counter}/{total_combinations} ({counter/total_combinations*100:.1f}%) - Melhor custo até agora: {best_cost:.8f}")
    
    print(f"\nTempo de execução: {time.time() - start_time:.2f} segundos")
    print(f"Melhor solução encontrada pela busca em grade:")
    print(f"α₁ = {best_params[0]:.6f}")
    print(f"α₂ = {best_params[1]:.6f}")
    print(f"α₃ = {best_params[2]:.6f}")
    print(f"Custo mínimo: {best_cost:.8f}")
    
    # Comparação dos resultados
    print("\n=== Comparação dos Resultados ===")
    print(f"Método Nelder-Mead: Custo = {optimal_cost:.8f}")
    print(f"Busca em Grade Refinada: Custo = {best_cost:.8f}")
    
    # Retorna os resultados
    return {
        'nelder_mead': {
            'params': optimal_params,
            'cost': optimal_cost
        },
        'grid_search': {
            'params': best_params,
            'cost': best_cost
        }
    }

# Executa a busca
if __name__ == "__main__":
    results = busca_parametros_otimos()
    
    # Exporta resultados para Excel
    df = pd.DataFrame([
        {
            'metodo': 'Nelder-Mead',
            'param1_alpha1': results['nelder_mead']['params'][0],
            'param2_alpha2': results['nelder_mead']['params'][1],
            'param3_alpha3': results['nelder_mead']['params'][2],
            'custo_total': results['nelder_mead']['cost']
        },
        {
            'metodo': 'Busca em Grade Refinada',
            'param1_alpha1': results['grid_search']['params'][0],
            'param2_alpha2': results['grid_search']['params'][1],
            'param3_alpha3': results['grid_search']['params'][2],
            'custo_total': results['grid_search']['cost']
        }
    ])
    
    # Salva no Excel
    excel_filename = 'python\\outputs\\resultados_busca_parametros_otimos.xlsx'
    df.to_excel(excel_filename, index=False)
    print(f"\nResultados salvos em: {excel_filename}")