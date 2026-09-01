import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import pandas as pd

def runge_kutta(f, y0, t0, tf, h, order=4, show_progress=False, desc="RK"):
    """
    Método de Runge-Kutta para resolver sistemas de equações diferenciais.
    Esse método pode ser implementado tanto forward quanto backward, dependendo do valor de h.
    O método backward é utilizado quando h < 0, assim tf < t0.
    O método forward é utilizado quando h > 0, assim tf > t0.

    Parâmetros:
    f            : função que define o sistema de EDOs (dy/dt = f(t, y)).
    y0           : vetor de condições iniciais.
    t0           : tempo inicial.
    tf           : tempo final.
    h            : passo de integração.
    order        : ordem do método de Runge-Kutta (inteiro >= 1).
    show_progress: se True, mostra barra de progresso para os passos RK.
    desc         : descrição para a barra de progresso.

    Retorna:
    t  : array com os tempos.
    y  : array com as soluções para cada variável no sistema.
    """
    if order < 1 or not isinstance(order, int):
        raise ValueError("A ordem do método deve ser um número inteiro maior ou igual a 1.")
    
    if h == 0:
        raise ValueError("O passo de integração não pode ser zero.")
    elif h < 0 and tf > t0:
        raise ValueError("Para h < 0, tf deve ser menor que t0.")
    elif h > 0 and tf < t0:
        raise ValueError("Para h > 0, tf deve ser maior que t0.")
    else:
        pass

    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    # Cria iterator com ou sem barra de progresso
    if show_progress:
        iterator = tqdm(range(1, len(t)), desc=desc, leave=False, unit="passo")
    else:
        iterator = range(1, len(t))

    for i in iterator:
        ti = t[i - 1]
        yi = y[i - 1]

        k = []
        k.append(h * np.array(f(ti, yi)))
        for j in range(1, order):
            k.append(h * np.array(f(ti + j * h / order, yi + sum(k) / order)))
        y[i] = yi + sum(k) / order

        # Atualiza informações da barra de progresso se ativada
        if show_progress:
            current_time = t[i]
            iterator.set_postfix(t=f"{current_time:.3f}")

    return t, y

# --- Parâmetros do Sistema ---
# Parâmetros físicos do VANT
Ixx = 0.00184  # Momento de inércia em x
Iyy = 0.00225  # Momento de inércia em y  
Izz = 0.00338  # Momento de inércia em z
Ir = 0.00001  # Momento de inércia do rotor
Omega_r = 0.0  # Velocidade angular do rotor
T_theta = 0.1  # Constante de torque

# Funções trigonométricas (assumindo pequenos ângulos inicialmente)
def C_phi(phi):
    return np.cos(phi)

def S_phi(phi):
    return np.sin(phi)

def C_theta(theta):
    return np.cos(theta)

# Condições iniciais [p, q, r, phi, theta, psi]
x0 = [0.0, 0.0, 0.0, 1, 0.0, 0.0]

# Matrizes de ponderação para a função de custo J = integral(x'Qx + u'Ru)dt
Q = np.diag([1, 1, 1, 50.0, 50.0, 50.0])  # 6x6

# Matriz R para 4 entradas de controle (u1, u2, u3, u4)
R = np.eye(3)*0.1

# Matriz de entrada do sistema B (6x4)
B = np.array([[1.0/Ixx, 0.0, 0.0],
              [0.0, 1.0/Iyy, 0.0],
              [0.0, 0.0, 1.0/Izz],
              [0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]])

# Intervalo de tempo para a simulação
t_span = [0, 1]
amostras_por_segundo = 4000
t_eval = np.linspace(t_span[0], t_span[1], t_span[1]*amostras_por_segundo) 

# Valores dos três parâmetros para testar (substituindo gama e epsilon)
amostras = 2  # Reduzido para 2, resultando em (2+1)³ = 27 simulações
param1_values = np.linspace(-10, 10, amostras)  # Parâmetro 1 (para p)
param2_values = np.linspace(-10, 10, amostras)  # Parâmetro 2 (para q) 
param3_values = np.linspace(-10, 10, amostras)  # Parâmetro 3 (para r)

# Calcula o total de simulações
total_simulations = len(param1_values) * len(param2_values) * len(param3_values)

# --- Simulação e Geração de Gráficos ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
results = {}

print("Iniciando simulações do controle SDRE para sistema 6x6...")
print(f"Total de simulações: {total_simulations}")

# Cria todas as combinações de parâmetros
param_combinations = list(itertools.product(param1_values, param2_values, param3_values))

# Loop através dos três parâmetros com barra de progresso
with tqdm(param_combinations, desc="Simulações", unit="sim", position=0) as pbar:
    for param1, param2, param3 in pbar:
        # Atualiza a descrição da barra de progresso
        pbar.set_description(f"α₁={param1:.2f}, α₂={param2:.2f}, α₃={param3:.2f}")
        
        # Variável para rastrear o maior autovalor durante esta simulação específica
        max_real_eigenvalue_for_sim = [-np.inf]
        # Lista para armazenar todos os autovalores ao longo do tempo
        eigenvalues_over_time = []

        def system_dynamics(t, y):
            
            # Extrai o vetor de estado x = [p, q, r, phi, theta, psi, J]
            x = y[:6].reshape(-1, 1)
            p, q, r, phi, theta, psi = x.flatten()

            # Calcula as funções trigonométricas
            c_phi = C_phi(phi)
            s_phi = S_phi(phi) 
            c_theta = C_theta(theta)

            s_phi = S_phi(phi)
            c_phi = C_phi(phi)
            c_theta = C_theta(theta)
            # Evitar divisão por zero para tan(theta) e 1/cos(theta)
            if abs(c_theta) < 1e-6:
                c_theta = 1e-6 * np.sign(c_theta)
            t_theta = np.sin(theta) / c_theta

            # Linha 1: p_dot
            a12 = param1 * ((Iyy - Izz) / Ixx) * r - (Ir * Omega_r / Ixx)
            a13 = (1 - param1) * ((Iyy - Izz) / Ixx) * q

            # Linha 2: q_dot
            a21 = param2 * ((Izz - Ixx) / Iyy) * r + (Ir * Omega_r / Iyy)
            a23 = (1 - param2) * ((Izz - Ixx) / Iyy) * p

            # Linha 3: r_dot
            a31 = param3 * ((Ixx - Iyy) / Izz) * q
            a32 = (1 - param3) * ((Ixx - Iyy) / Izz) * p

            # Matriz A(x) 6x6
            A = np.array([
                [0.0, a12, a13, 0.0, 0.0, 0.0],
                [a21, 0.0, a23, 0.0, 0.0, 0.0],
                [a31, a32, 0.0, 0.0, 0.0, 0.0],
                [1.0, s_phi * t_theta, c_phi * t_theta, 0.0, 0.0, 0.0],
                [0.0, c_phi, -s_phi, 0.0, 0.0, 0.0],
                [0.0, s_phi / c_theta, c_phi / c_theta, 0.0, 0.0, 0.0]
            ])

            # Tenta resolver a Equação de Riccati
            try:
                P = solve_continuous_are(A, B, Q, R)
            except (np.linalg.LinAlgError, ValueError):
                eigenvalues_over_time.append(0.0)  # Adiciona 0 em caso de erro
                return [0, 0, 0, 0, 0, 0, 0]

            # Calcula a entrada de controle SDRE
            K = np.linalg.inv(R) @ B.T @ P
            u = -K @ x

            # Calcula os autovalores de A(x) e atualiza o máximo da simulação
            eigenvalues = np.linalg.eigvals(A - B @ K)
            current_max_real_eig = np.max(np.real(eigenvalues))
            if current_max_real_eig > max_real_eigenvalue_for_sim[0]:
                max_real_eigenvalue_for_sim[0] = current_max_real_eig
            
            # Armazena o maior autovalor atual
            eigenvalues_over_time.append(current_max_real_eig)

            # Calcula as derivadas de estado
            dxdt = (A @ x + B @ u).flatten()

            # Calcula a derivada do custo
            dJdt = (x.T @ Q @ x + u.T @ R @ u)[0, 0]

            return [dxdt[0], dxdt[1], dxdt[2], dxdt[3], dxdt[4], dxdt[5], dJdt]

        # Condições iniciais: [p, q, r, phi, theta, psi, J]
        y0 = x0 + [0.0]

        # Resolve a equação diferencial
        # Calcula o passo de integração h para o método Runge-Kutta
        h = (t_span[1] - t_span[0]) / (len(t_eval) - 1)

        # Chama a função runge_kutta com barra de progresso
        t, y = runge_kutta(
            f=system_dynamics,
            y0=y0,
            t0=t_span[0],
            tf=t_span[1],
            h=h,
            order=4,
            show_progress=True,
            desc=f"RK α₁={param1:.1f},α₂={param2:.1f},α₃={param3:.1f}"
        )

        # A função runge_kutta não tem um indicador de sucesso como o solve_ivp.
        # Assumimos que a integração foi bem-sucedida se não houver exceção.
        # A saída 'y' de runge_kutta precisa ser transposta para corresponder ao formato de sol.y.
        sol_y_transposed = y.T

        # Calcula o custo final como a soma de todos os dJdt (aproximação da integral)
        dt = h  # Passo de integração
        dJdt_values = sol_y_transposed[6, :]  # Todos os valores de dJ/dt
        final_cost = np.trapezoid(dJdt_values, dx=dt)  # Integração numérica usando regra trapezoidal
        
        key = (param1, param2, param3)
        results[key] = {
            't': t,
            'states': sol_y_transposed[:6],
            'cost': final_cost,
            'max_eig': max_real_eigenvalue_for_sim[0],
            'eigenvalues_time': np.array(eigenvalues_over_time)  # Armazena todos os autovalores
        }
        
        # Atualiza o sufixo da barra de progresso com o custo e o maior autovalor
        pbar.set_postfix(J=f"{final_cost:.8f}", λ_max=f"{max_real_eigenvalue_for_sim[0]:.4f}")

# --- Exportação para Excel (ANTES dos gráficos) ---
if results:
    print("\n--- Processando resultados e exportando para Excel ---")
    
    # Prepara os dados para o DataFrame
    data = []
    for params, result in results.items():
        data.append({
            'param1_alpha1': params[0],
            'param2_alpha2': params[1], 
            'param3_alpha3': params[2],
            'custo_total': result['cost'],
            'maior_autovalor_real': result['max_eig']
        })
    
    # Cria o DataFrame
    df = pd.DataFrame(data)
    
    # Ordena por custo total (menor para maior)
    df = df.sort_values('custo_total')
    
    # Define o nome do arquivo Excel
    excel_filename = 'resultados_simulacao_sdre.xlsx'
    
    try:
        # Exporta para Excel
        df.to_excel(excel_filename, index=False, sheet_name='Resultados')
        print(f"Arquivo Excel salvo como: {excel_filename}")
        print(f"Total de registros exportados: {len(df)}")

        
    except Exception as e:
        print(f"Erro ao exportar para Excel: {e}")
        print("Verifique se você tem a biblioteca openpyxl instalada: pip install openpyxl")
        
        # Como alternativa, salva em CSV
        csv_filename = 'resultados_simulacao_sdre.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Arquivo CSV salvo como alternativa: {csv_filename}")

    # Encontra os melhores e piores parâmetros
    cost_list = [results[key]['cost'] for key in results.keys()]
    min_cost = min(cost_list)
    max_cost = max(cost_list)

    best_params = None
    worst_params = None

    for params, result in results.items():
        if result['cost'] == min_cost:
            best_params = params
        if result['cost'] == max_cost:
            worst_params = params

    # Resumo dos resultados
    print("\n--- Resumo dos Custos Finais ---")
    print(f"Melhor caso: α₁={best_params[0]:.2f}, α₂={best_params[1]:.2f}, α₃={best_params[2]:.2f}, J={min_cost:.8f}, λ_max={results[best_params]['max_eig']:.6f}")
    print(f"Pior caso: α₁={worst_params[0]:.2f}, α₂={worst_params[1]:.2f}, α₃={worst_params[2]:.2f}, J={max_cost:.8f}, λ_max={results[worst_params]['max_eig']:.6f}")
    
    print("\nTodos os resultados:")
    for params in sorted(results.keys()):
        cost = results[params]['cost']
        max_eig = results[params]['max_eig']
        print(f"α₁={params[0]:.2f}, α₂={params[1]:.2f}, α₃={params[2]:.2f}: J={cost:.8f}, λ_max={max_eig:.6f}")

else:
    print("Nenhum resultado válido obtido nas simulações.")
    # Cria um DataFrame vazio mesmo sem resultados
    df_empty = pd.DataFrame(columns=['param1_alpha1', 'param2_alpha2', 'param3_alpha3', 'custo_total'])
    df_empty.to_excel('resultados_simulacao_sdre_vazio.xlsx', index=False)
    print("Arquivo Excel vazio criado: resultados_simulacao_sdre_vazio.xlsx")

# --- Geração dos Gráficos (DEPOIS da exportação) ---
print("\n--- Gerando gráficos ---")

if results:
    # Plota cada estado em um subplot
    state_names = ['p(t)', 'q(t)', 'r(t)', 'φ(t)', 'θ(t)', 'ψ(t)']
    for params, result in results.items():
        for i in range(6):
            axes[i].plot(result['t'], result['states'][i], alpha=0.7,
                       label=f'α_1={params[0]:.1f}, α_2={params[1]:.1f}, α_3={params[2]:.1f}')
            axes[i].set_title(f'Estado {state_names[i]}', fontsize=12)
            axes[i].set_xlabel('Tempo (s)', fontsize=10)
            axes[i].grid(True)

    # Finaliza os gráficos
    for ax in axes:
        if len(results) <= 10:  # Só mostra legenda se não houver muitas curvas
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # Gráfico de comparação dos melhores vs piores resultados
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    axes2 = axes2.flatten()
    
    for i in range(6):
        axes2[i].plot(results[best_params]['t'], results[best_params]['states'][i], 
                     'b-', linewidth=2, 
                     label=f'Melhor: α_1={best_params[0]:.2f}, α_2={best_params[1]:.2f}, α_3={best_params[2]:.2f} (J={min_cost:.8f}, λ_max={results[best_params]["max_eig"]:.4f})')
        axes2[i].plot(results[worst_params]['t'], results[worst_params]['states'][i], 
                     'r-', linewidth=2,
                     label=f'Pior: α_1={worst_params[0]:.2f}, α_2={worst_params[1]:.2f}, α_3={worst_params[2]:.2f} (J={max_cost:.8f}, λ_max={results[worst_params]["max_eig"]:.4f})')
        axes2[i].set_title(f'Comparação - {state_names[i]}', fontsize=12)
        axes2[i].set_xlabel('Tempo (s)', fontsize=10)
        axes2[i].legend(fontsize=8)
        axes2[i].grid(True)

    plt.tight_layout()
    plt.show()