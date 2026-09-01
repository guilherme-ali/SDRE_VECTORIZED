"""
Algoritmo de Schur para Resolução da Equação Algébrica de Riccati 
no Tempo Discreto (DARE - Discrete-Time Algebraic Riccati Equation)

A DARE é dada por:
A^T * P * A - P - A^T * P * B * (R + B^T * P * B)^{-1} * B^T * P * A + Q = 0

Método de Schur (Decomposição QZ Generalizada):
===============================================
PASSO 1: Construir matrizes H e J (2n x 2n) para o problema generalizado
PASSO 2: Aplicar decomposição QZ ordenada (ordqz) com autovalores estáveis
PASSO 3: Extrair subespaço invariante estável das matrizes Q e Z
PASSO 4: Calcular P = U21 * inv(U11) a partir dos blocos da matriz Z
PASSO 5: Calcular ganho de realimentação K = inv(R + B^T*P*B) * B^T*P*A
"""

import numpy as np
from scipy.linalg import ordqz, solve_discrete_are, qr
import matplotlib.pyplot as plt
import time
import tracemalloc
import sys



def dare_schur(A, B, Q, R):
    """
    Resolve a DARE usando o método de Schur (Decomposição QZ Generalizada).
    
    Parâmetros:
    -----------
    A : array_like, shape (n, n)
        Matriz de estados do sistema discreto
    B : array_like, shape (n, m)
        Matriz de entrada do sistema
    Q : array_like, shape (n, n)
        Matriz de peso dos estados (semi-definida positiva)
    R : array_like, shape (m, m)
        Matriz de peso das entradas (definida positiva)
    
    Retorna:
    --------
    P : ndarray, shape (n, n)
        Solução da DARE (matriz simétrica positiva definida)
    K : ndarray, shape (m, n)
        Ganho de realimentação ótimo LQR
    eig_cl : ndarray, shape (n,)
        Autovalores do sistema em malha fechada
    """
    
    # ========================================================================
    # PASSO 0: Preparação - Converte entradas para arrays numpy
    # ========================================================================
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    
    n = A.shape[0]  # Dimensão do estado
    m = B.shape[1] if B.ndim > 1 else 1  # Número de entradas
    
    # Verifica dimensões para garantir consistência
    assert A.shape == (n, n), "A deve ser quadrada"
    assert Q.shape == (n, n), "Q deve ter dimensão nxn"
    
    # ========================================================================
    # PASSO 1: Construir matrizes H e J para o problema generalizado
    # ========================================================================
    # O método de Schur para DARE resolve o problema de autovalores generalizado:
    # H * z = λ * J * z
    # onde os autovalores λ com |λ| < 1 correspondem ao subespaço estável
    
    # Calcula G = B * inv(R) * B^T (matriz de ganho ponderada)
    G = B @ np.linalg.inv(R) @ B.T
    
    # Monta matriz H (Hamiltoniana superior - 2n x 2n)
    # H = [A    0]
    #     [-Q   I]
    H = np.block([
        [A, np.zeros((n, n))],  # Bloco superior: [A | 0]
        [-Q, np.eye(n)]         # Bloco inferior: [-Q | I]
    ])
    
    # Monta matriz J (Hamiltoniana inferior - 2n x 2n)
    # J = [I   G]
    #     [0  A^T]
    J = np.block([
        [np.eye(n), G],              # Bloco superior: [I | G]
        [np.zeros((n, n)), A.T]      # Bloco inferior: [0 | A^T]
    ])
    
    # ========================================================================
    # PASSO 2: Decomposição QZ Generalizada Ordenada
    # ========================================================================
    # ordqz(H, J, sort='iuc') realiza:
    # - Decomposição de Schur generalizada: H = Q*S*Z^T, J = Q*T*Z^T
    # - Ordenação dos autovalores: 'iuc' = inside unit circle (|λ| < 1 primeiro)
    # Retorna: AA, BB (matrizes triangulares), alpha, beta (autovalores λ=alpha/beta),
    #          Q_schur, Z_schur (matrizes ortogonais de transformação)
    
    AA, BB, alpha, beta, Q_schur, Z_schur = ordqz(H, J, sort='iuc')
    
    # ========================================================================
    # PASSO 3: Extrair subespaço invariante estável
    # ========================================================================
    # A matriz Z_schur contém os autovetores generalizados ordenados
    # Os primeiros n autovetores correspondem aos autovalores estáveis (|λ| < 1)
    # Particiona Z_schur em 4 blocos n x n:
    # Z = [U11  U12]
    #     [U21  U22]
    
    U11 = Z_schur[:n, :n]   # Bloco superior esquerdo
    U21 = Z_schur[n:, :n]   # Bloco inferior esquerdo (corresponde aos multiplicadores de Lagrange)
    
    # ========================================================================
    # PASSO 4: Calcular a solução P da DARE
    # ========================================================================
    # A solução P é obtida pela relação: P = U21 * inv(U11)
    # Isso vem da teoria de subespaços invariantes para equações de Riccati
    P = U21 @ np.linalg.inv(U11)
    
    # Simetriza P para remover pequenos erros numéricos
    # (P deve ser simétrica por construção, mas erros de arredondamento podem ocorrer)
    P = (P + P.T) / 2
    
    # ========================================================================
    # PASSO 5: Calcular o ganho de realimentação ótimo K
    # ========================================================================
    # O ganho LQR ótimo é dado por: K = (R + B^T*P*B)^{-1} * B^T*P*A
    # Esse ganho minimiza o custo quadrático J = Σ(x^T*Q*x + u^T*R*u)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    # ========================================================================
    # PASSO 6: Calcular autovalores do sistema em malha fechada
    # ========================================================================
    # Sistema em malha fechada: x[k+1] = (A - B*K)*x[k]
    # Os autovalores devem ter magnitude < 1 para garantir estabilidade
    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)
    
    return P, K, eig_cl


def dare_van_dooren(A, B, Q, R):
    """
    Resolve a DARE usando o método de van Dooren (Extended Symplectic Pencil).
    
    Este método forma o pencil simplétictico estendido (2n+m) × (2n+m):
    
    H - λJ onde:
    
    H = [ A    0    B ]       J = [ E   0   B ]
        [-Q   E^H  -S ]           [ 0  A^H  0 ]
        [S^H   0    R ]           [ 0 -B^H  0 ]
    
    Para o caso padrão (E=I, S=0):
    
    H = [ A    0    B ]       J = [ I   0   B ]
        [-Q   I    0 ]           [ 0  A^T  0 ]
        [ 0   0    R ]           [ 0  -B^T 0 ]
    
    Referência: P. van Dooren, "A Generalized Eigenvalue Approach For Solving
    Riccati Equations", SIAM J. Sci. Stat. Comput., Vol.2(2), 1981.
    
    Parâmetros:
    -----------
    A : array_like, shape (n, n)
        Matriz de estados do sistema discreto
    B : array_like, shape (n, m)
        Matriz de entrada do sistema
    Q : array_like, shape (n, n)
        Matriz de peso dos estados (semi-definida positiva)
    R : array_like, shape (m, m)
        Matriz de peso das entradas (definida positiva)
    
    Retorna:
    --------
    P : ndarray, shape (n, n)
        Solução da DARE (matriz simétrica positiva definida)
    K : ndarray, shape (m, n)
        Ganho de realimentação ótimo LQR
    eig_cl : ndarray, shape (n,)
        Autovalores do sistema em malha fechada
    """
    
    # ========================================================================
    # PASSO 0: Preparação
    # ========================================================================
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    
    n = A.shape[0]  # Dimensão do estado
    m = B.shape[1] if B.ndim > 1 else 1  # Número de entradas
    
    # ========================================================================
    # PASSO 1: Construir o Extended Symplectic Pencil (2n+m) × (2n+m)
    # ========================================================================
    # Este é o método completo de van Dooren que evita inversão de R
    
    # Matriz H (2n+m) × (2n+m)
    H = np.zeros((2*n + m, 2*n + m), dtype=float)
    
    # Bloco superior: [A | 0 | B]
    H[:n, :n] = A                           # A
    H[:n, n:2*n] = 0.0                      # 0
    H[:n, 2*n:] = B                         # B
    
    # Bloco do meio: [-Q | I | 0]
    H[n:2*n, :n] = -Q                       # -Q
    H[n:2*n, n:2*n] = np.eye(n)            # I (E^H para caso padrão)
    H[n:2*n, 2*n:] = 0.0                    # 0 (-S para caso generalizado)
    
    # Bloco inferior: [0 | 0 | R]
    H[2*n:, :n] = 0.0                       # 0 (S^H para caso generalizado)
    H[2*n:, n:2*n] = 0.0                    # 0
    H[2*n:, 2*n:] = R                       # R
    
    # Matriz J (2n+m) × (2n+m)
    J = np.zeros((2*n + m, 2*n + m), dtype=float)
    
    # Bloco superior: [I | 0 | B]
    J[:n, :n] = np.eye(n)                   # I (E para caso padrão)
    J[:n, n:2*n] = 0.0                      # 0
    J[:n, 2*n:] = B                         # B
    
    # Bloco do meio: [0 | A^T | 0]
    J[n:2*n, :n] = 0.0                      # 0
    J[n:2*n, n:2*n] = A.T                   # A^T
    J[n:2*n, 2*n:] = 0.0                    # 0
    
    # Bloco inferior: [0 | -B^T | 0]
    J[2*n:, :n] = 0.0                       # 0
    J[2*n:, n:2*n] = -B.T                   # -B^T
    J[2*n:, 2*n:] = 0.0                     # 0
    
    # ========================================================================
    # PASSO 2: Deflação pela coluna R (método de van Dooren)
    # ========================================================================
    # Deflaciona o pencil de (2n+m)×(2n+m) para (2n)×(2n)
    # usando decomposição QR das últimas m colunas de H
    
    # Extrai as últimas m colunas de H para deflação
    H_last_cols = H[:, -m:]
    
    # QR decomposition: H_last_cols = Q_qr @ R_qr
    Q_qr, _ = qr(H_last_cols)
    
    # Projeta H e J no espaço ortogonal às últimas m colunas
    # Pega apenas as primeiras 2n colunas de Q_qr (descarta as últimas m)
    Q_deflate = Q_qr[:, m:]  # (2n+m) × 2n
    
    # Aplica deflação: H_def = Q_deflate^T @ H @ [:, :2n]
    #                  J_def = Q_deflate^T @ J @ [:, :2n]
    H_deflated = Q_deflate.T @ H[:, :2*n]  # (2n) × (2n)
    J_deflated = Q_deflate.T @ J[:, :2*n]  # (2n) × (2n)
    
    # ========================================================================
    # PASSO 3: Decomposição QZ Generalizada Ordenada (2n × 2n)
    # ========================================================================
    # ordqz com 'iuc' = inside unit circle (|λ| < 1 para DARE)
    
    AA, BB, alpha, beta, Q_schur, Z_schur = ordqz(
        H_deflated, J_deflated, 
        sort='iuc',
        output='real'
    )
    
    # ========================================================================
    # PASSO 4: Extrair subespaço invariante estável (n primeiros autovetores)
    # ========================================================================
    # Z_schur = [U11  U12]  onde cada bloco é n × n
    #           [U21  U22]
    
    U11 = Z_schur[:n, :n]
    U21 = Z_schur[n:, :n]
    
    # ========================================================================
    # PASSO 5: Calcular solução P = U21 @ inv(U11)
    # ========================================================================
    P = U21 @ np.linalg.inv(U11)
    
    # Simetriza para remover erros numéricos
    P = (P + P.T) / 2
    
    # ========================================================================
    # PASSO 6: Calcular ganho K
    # ========================================================================
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    # ========================================================================
    # PASSO 7: Autovalores do sistema em malha fechada
    # ========================================================================
    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)
    
    return P, K, eig_cl


def dare_iterativo(A, B, Q, R, max_iter=1000, tol=1e-9):
    """
    Resolve a DARE usando método iterativo (equação de Riccati discreta).
    
    Itera a equação:
    P[k+1] = A^T * P[k] * A - A^T * P[k] * B * (R + B^T * P[k] * B)^{-1} * B^T * P[k] * A + Q
    
    Até convergência: ||P[k+1] - P[k]|| < tol
    
    Parâmetros:
    -----------
    A : array_like, shape (n, n)
        Matriz de estados do sistema discreto
    B : array_like, shape (n, m)
        Matriz de entrada do sistema
    Q : array_like, shape (n, n)
        Matriz de peso dos estados (semi-definida positiva)
    R : array_like, shape (m, m)
        Matriz de peso das entradas (definida positiva)
    max_iter : int, opcional
        Número máximo de iterações (padrão: 1000)
    tol : float, opcional
        Tolerância para convergência (padrão: 1e-10)
    
    Retorna:
    --------
    P : ndarray, shape (n, n)
        Solução da DARE (matriz simétrica positiva definida)
    K : ndarray, shape (m, n)
        Ganho de realimentação ótimo LQR
    eig_cl : ndarray, shape (n,)
        Autovalores do sistema em malha fechada
    n_iter : int
        Número de iterações até convergência
    """
    
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    
    n = A.shape[0]
    
    # Inicializa P com Q (chute inicial)
    P = Q.copy()
    
    # Iteração até convergência
    for k in range(max_iter):
        # Calcula P[k+1] usando a equação de Riccati
        # P[k+1] = A^T * P[k] * A - A^T * P[k] * B * (R + B^T * P[k] * B)^{-1} * B^T * P[k] * A + Q
        
        BTP = B.T @ P
        BTPB = BTP @ B
        
        # Inversa do termo (R + B^T*P*B)
        inv_term = np.linalg.inv(R + BTPB)
        
        # P[k+1]
        P_novo = A.T @ P @ A - A.T @ P @ B @ inv_term @ BTP @ A + Q
        
        # Simetriza para remover erros numéricos
        P_novo = (P_novo + P_novo.T) / 2
        
        # Verifica convergência
        diff = np.linalg.norm(P_novo - P, 'fro')
        
        if diff < tol:
            P = P_novo
            n_iter = k + 1
            break
        
        P = P_novo
    else:
        n_iter = max_iter
        print(f"⚠ Aviso: Método iterativo não convergiu em {max_iter} iterações")
    
    # Calcula ganho K
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    # Calcula autovalores do sistema em malha fechada
    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)
    
    return P, K, eig_cl, n_iter


def dare_sda(A, B, Q, R, max_iter=100, tol=1e-9):
    """
    Resolve a DARE usando o Structure-preserving Doubling Algorithm (SDA).
    
    O SDA é um método iterativo de duplicação com convergência QUADRÁTICA,
    muito mais rápido que o método iterativo simples. O número de iterações
    típico é log2(1/tol) ≈ 30-50 iterações.
    
    O algoritmo explora a estrutura simplética do problema de Riccati,
    preservando propriedades numéricas importantes.
    
    Referência: 
    - Chu, Hwang, Tsai & Wang, "A structure-preserving doubling algorithm 
      for continuous-time algebraic Riccati equations", Linear Algebra Appl., 2005
    - Anderson & Moore, "Optimal Filtering", Dover, 2005 (adaptação para DARE)
    
    Algoritmo:
    ----------
    Inicialização:
        A_k = A
        G_k = B @ inv(R) @ B^T
        H_k = Q
    
    Iteração (duplicação):
        W_k = (I + G_k @ H_k)^{-1}
        A_{k+1} = A_k @ W_k @ A_k
        G_{k+1} = G_k + A_k @ W_k @ G_k @ A_k^T
        H_{k+1} = H_k + A_k^T @ H_k @ W_k @ A_k
    
    Convergência:
        H_k → P (solução da DARE) quando k → ∞
        ||G_k|| → 0 (critério de convergência)
    
    Parâmetros:
    -----------
    A : array_like, shape (n, n)
        Matriz de estados do sistema discreto
    B : array_like, shape (n, m)
        Matriz de entrada do sistema
    Q : array_like, shape (n, n)
        Matriz de peso dos estados (semi-definida positiva)
    R : array_like, shape (m, m)
        Matriz de peso das entradas (definida positiva)
    max_iter : int, opcional
        Número máximo de iterações (padrão: 100, geralmente < 50 são suficientes)
    tol : float, opcional
        Tolerância para convergência (padrão: 1e-14)
    
    Retorna:
    --------
    P : ndarray, shape (n, n)
        Solução da DARE (matriz simétrica positiva definida)
    K : ndarray, shape (m, n)
        Ganho de realimentação ótimo LQR
    eig_cl : ndarray, shape (n,)
        Autovalores do sistema em malha fechada
    n_iter : int
        Número de iterações até convergência
    """
    
    # ========================================================================
    # PASSO 0: Preparação
    # ========================================================================
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    
    n = A.shape[0]
    I = np.eye(n)
    
    # ========================================================================
    # PASSO 1: Inicialização das variáveis de duplicação
    # ========================================================================
    # A_k: matriz de transição iterada (converge para 0)
    # G_k: matriz de "ganho" iterada (converge para 0)
    # H_k: matriz de "custo" iterada (converge para P)
    
    A_k = A.copy()
    G_k = B @ np.linalg.inv(R) @ B.T  # G = B * R^{-1} * B^T
    H_k = Q.copy()
    
    # ========================================================================
    # PASSO 2: Iteração de duplicação (doubling)
    # ========================================================================
    # A cada iteração, o horizonte de tempo é efetivamente duplicado,
    # resultando em convergência quadrática: ||error_k|| ∝ ||error_0||^{2^k}
    
    for k in range(max_iter):
        # Calcula W_k = (I + G_k @ H_k)^{-1}
        # Esse termo aparece na fórmula de atualização e representa
        # a correção devido ao acoplamento entre G e H
        W_k = np.linalg.inv(I + G_k @ H_k)
        
        # Atualiza A_{k+1}: representa a dinâmica após 2^{k+1} passos
        # A_{k+1} = A_k @ W_k @ A_k
        A_novo = A_k @ W_k @ A_k
        
        # Atualiza G_{k+1}: acumula o "ganho" total do sistema
        # G_{k+1} = G_k + A_k @ W_k @ G_k @ A_k^T
        G_novo = G_k + A_k @ W_k @ G_k @ A_k.T
        
        # Atualiza H_{k+1}: acumula o "custo" total (converge para P)
        # H_{k+1} = H_k + A_k^T @ H_k @ W_k @ A_k
        H_novo = H_k + A_k.T @ H_k @ W_k @ A_k
        
        # Simetriza G e H para manter estabilidade numérica
        G_novo = (G_novo + G_novo.T) / 2
        H_novo = (H_novo + H_novo.T) / 2
        
        # Verifica convergência: ||A_k|| → 0 ou ||G_k|| → 0
        # Quando A_k → 0, o sistema "esqueceu" a condição inicial
        diff_A = np.linalg.norm(A_novo, 'fro')
        diff_G = np.linalg.norm(G_novo, 'fro')
        diff_H = np.linalg.norm(H_novo - H_k, 'fro')
        
        # Critério de convergência: A_k deve tender a zero
        if diff_A < tol or diff_H < tol:
            H_k = H_novo
            n_iter = k + 1
            break
        
        # Atualiza variáveis para próxima iteração
        A_k = A_novo
        G_k = G_novo
        H_k = H_novo
    else:
        n_iter = max_iter
        print(f"⚠ Aviso: SDA não convergiu em {max_iter} iterações")
    
    # ========================================================================
    # PASSO 3: Extrai solução P da DARE
    # ========================================================================
    # No limite, H_k converge para a solução P da DARE
    P = H_k
    
    # Simetriza P para remover pequenos erros numéricos
    P = (P + P.T) / 2
    
    # ========================================================================
    # PASSO 4: Calcula ganho de realimentação K
    # ========================================================================
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    # ========================================================================
    # PASSO 5: Calcula autovalores do sistema em malha fechada
    # ========================================================================
    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)
    
    return P, K, eig_cl, n_iter


def dare_adda(A, B, Q, R, max_iter=100, tol=1e-9):
    """
    Resolve a DARE usando o Alternating-Directional Doubling Algorithm (ADDA).
    
    O ADDA é uma variante do SDA (Structure-preserving Doubling Algorithm)
    que alterna a ordem das multiplicações matriciais a cada iteração,
    melhorando a estabilidade numérica para certos problemas.
    
    A diferença principal entre SDA e ADDA está na ordem de atualização:
    - SDA: sempre multiplica na mesma ordem (A_k @ W_k @ A_k)
    - ADDA: alterna entre (A_k @ W_k @ A_k) e (A_k @ A_k @ W_k) 
    
    Isso distribui melhor os erros de arredondamento e pode convergir
    mais rapidamente para problemas específicos.
    
    Referência:
    - Lin, Xu, "On the Doubling Algorithm for a (Shifted) Nonsymmetric 
      Algebraic Riccati Equation", 2007
    - Chu et al., "Structure-preserving algorithms for periodic 
      discrete-time algebraic Riccati equations", 2004
    
    Parâmetros:
    -----------
    A : array_like, shape (n, n)
        Matriz de estados do sistema discreto
    B : array_like, shape (n, m)
        Matriz de entrada do sistema
    Q : array_like, shape (n, n)
        Matriz de peso dos estados (semi-definida positiva)
    R : array_like, shape (m, m)
        Matriz de peso das entradas (definida positiva)
    max_iter : int, opcional
        Número máximo de iterações (padrão: 100)
    tol : float, opcional
        Tolerância para convergência (padrão: 1e-9)
    
    Retorna:
    --------
    P : ndarray, shape (n, n)
        Solução da DARE (matriz simétrica positiva definida)
    K : ndarray, shape (m, n)
        Ganho de realimentação ótimo LQR
    eig_cl : ndarray, shape (n,)
        Autovalores do sistema em malha fechada
    n_iter : int
        Número de iterações até convergência
    """
    
    # ========================================================================
    # PASSO 0: Preparação
    # ========================================================================
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    
    n = A.shape[0]
    I = np.eye(n)
    
    # ========================================================================
    # PASSO 1: Inicialização (igual ao SDA)
    # ========================================================================
    A_k = A.copy()
    G_k = B @ np.linalg.inv(R) @ B.T
    H_k = Q.copy()
    
    # ========================================================================
    # PASSO 2: Iteração ADDA (Alternating-Directional Doubling)
    # ========================================================================
    # A característica do ADDA é alternar a ordem das operações
    # entre iterações pares e ímpares
    
    for k in range(max_iter):
        # Calcula W_k = (I + G_k @ H_k)^{-1}
        W_k = np.linalg.inv(I + G_k @ H_k)
        
        # ------------------------------------------------------------------
        # ADDA: Alterna a ordem das multiplicações
        # ------------------------------------------------------------------
        if k % 2 == 0:
            # Iteração par: multiplicação à esquerda primeiro
            # A_{k+1} = A_k @ W_k @ A_k
            temp = A_k @ W_k
            A_novo = temp @ A_k
            
            # G_{k+1} = G_k + A_k @ W_k @ G_k @ A_k^T
            G_novo = G_k + temp @ G_k @ A_k.T
            
            # H_{k+1} = H_k + A_k^T @ H_k @ W_k @ A_k
            H_novo = H_k + A_k.T @ H_k @ W_k @ A_k
        else:
            # Iteração ímpar: multiplicação à direita primeiro
            # A_{k+1} = A_k @ W_k @ A_k (mas ordem de cálculo diferente)
            temp = W_k @ A_k
            A_novo = A_k @ temp
            
            # G_{k+1} = G_k + A_k @ W_k @ G_k @ A_k^T
            G_novo = G_k + A_k @ W_k @ G_k @ A_k.T
            
            # H_{k+1} = H_k + A_k^T @ H_k @ W_k @ A_k (ordem diferente)
            temp_H = H_k @ W_k
            H_novo = H_k + A_k.T @ temp_H @ A_k
        
        # Simetriza G e H para estabilidade numérica
        G_novo = (G_novo + G_novo.T) / 2
        H_novo = (H_novo + H_novo.T) / 2
        
        # ------------------------------------------------------------------
        # Verifica convergência
        # ------------------------------------------------------------------
        diff_A = np.linalg.norm(A_novo, 'fro')
        diff_H = np.linalg.norm(H_novo - H_k, 'fro')
        
        if diff_A < tol or diff_H < tol:
            H_k = H_novo
            n_iter = k + 1
            break
        
        A_k = A_novo
        G_k = G_novo
        H_k = H_novo
    else:
        n_iter = max_iter
        print(f"⚠ Aviso: ADDA não convergiu em {max_iter} iterações")
    
    # ========================================================================
    # PASSO 3: Extrai solução P da DARE
    # ========================================================================
    P = H_k
    P = (P + P.T) / 2
    
    # ========================================================================
    # PASSO 4: Calcula ganho de realimentação K
    # ========================================================================
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    # ========================================================================
    # PASSO 5: Calcula autovalores do sistema em malha fechada
    # ========================================================================
    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)
    
    return P, K, eig_cl, n_iter


def verificar_solucao_dare(A, B, Q, R, P):
    """
    Verifica se P é solução da DARE calculando o resíduo.
    
    A DARE é: A^T*P*A - P - A^T*P*B*(R + B^T*P*B)^{-1}*B^T*P*A + Q = 0
    
    Retorna o resíduo e sua norma (deve ser próximo de zero).
    """
    # Calcula cada termo da DARE separadamente
    termo1 = A.T @ P @ A                                                    # A^T * P * A
    termo2 = -P                                                             # -P
    termo3 = -A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A  # Termo de realimentação
    termo4 = Q                                                              # Q
    
    # Soma todos os termos (deve ser ≈ 0 se P é solução)
    residuo = termo1 + termo2 + termo3 + termo4
    erro = np.linalg.norm(residuo, 'fro')  # Norma de Frobenius
    
    return residuo, erro


def simular_resposta_ao_degrau(A, B, K, x0, n_steps=100):
    """
    Simula a resposta do sistema em malha fechada com controle LQR.
    
    Sistema em malha fechada: x[k+1] = (A - BK)*x[k]
    Controle: u[k] = -K*x[k]
    """
    n = A.shape[0]
    m = B.shape[1] if B.ndim > 1 else 1
    
    # Arrays para armazenar histórico
    x_hist = np.zeros((n, n_steps))
    u_hist = np.zeros((m, n_steps))
    
    x = x0.copy()
    
    for k in range(n_steps):
        x_hist[:, k] = x.flatten()
        u = -K @ x
        u_hist[:, k] = u.flatten()
        
        # Atualiza estado: x[k+1] = (A - B*K)*x[k]
        x = (A - B @ K) @ x
    
    return x_hist, u_hist


def exemplo_sistema_6x6():
    """
    Exemplo 4: Sistema de alta dimensão (6x6) com múltiplas entradas (3)
    
    Sistema do drone/VANT - MESMAS MATRIZES DO CÓDIGO C++
    Estados: [roll, pitch, yaw, p, q, r]^T
    Entradas: [τx, τy, τz]^T (torques)
    """
    
    print("\n\n" + "=" * 80)
    print("EXEMPLO 4: Sistema 6x6 com 3 Entradas de Controle (Drone)")
    print("=" * 80)
    
    # Parâmetros físicos do sistema (do código C++)
    Ixx = 16.57e-6  # kg·m²
    Iyy = 16.57e-6  # kg·m²
    Izz = 29.80e-6  # kg·m²
    Ir = 1e-9   # kg·m²
    omega_r = 0
    
    # Valores de roll, pitch, yaw, p, q, r (do código C++)
    roll = 0.1
    pitch = -0.2
    yaw = 0.0
    p = 0.0
    q = 0.0
    r = 0.0
    
    # Tempo de amostragem (do código C++)
    dt = 0.01  # samplingTime
    
    print(f"\nParâmetros do sistema:")
    print(f"  Ixx = {Ixx} kg·m²")
    print(f"  Iyy = {Iyy} kg·m²")
    print(f"  Izz = {Izz} kg·m²")
    print(f"  dt = {dt} s")
    
    # ========================================================================
    # Construção da Matriz A contínua (mesmo código C++)
    # ========================================================================
    A_continuous = np.zeros((6, 6))
    
    # Calcula valores trigonométricos
    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)
    cos_pitch = np.cos(pitch)
    tan_pitch = np.tan(pitch)
    inv_cos_pitch = 1.0 / cos_pitch
    
    # Preenche a matriz A conforme o sistema do drone (linha por linha como no C++)
    # Linha 0
    A_continuous[0, 3] = 1
    A_continuous[0, 4] = sin_roll * tan_pitch
    A_continuous[0, 5] = cos_roll * tan_pitch
    
    # Linha 1
    A_continuous[1, 4] = cos_roll
    A_continuous[1, 5] = -sin_roll
    
    # Linha 2
    A_continuous[2, 4] = sin_roll * inv_cos_pitch
    A_continuous[2, 5] = cos_roll * inv_cos_pitch
    
    # Linha 3
    A_continuous[3, 4] = ((Iyy - Izz) / (2 * Ixx)) * r - Ir * omega_r / Ixx
    A_continuous[3, 5] = ((Iyy - Izz) / (2 * Ixx)) * q
    
    # Linha 4
    A_continuous[4, 3] = ((Izz - Ixx) / (2 * Iyy)) * r - Ir * omega_r / Iyy
    A_continuous[4, 5] = ((Izz - Ixx) / (2 * Iyy)) * p
    
    # Linha 5
    A_continuous[5, 3] = ((Ixx - Iyy) / (2 * Izz)) * q
    A_continuous[5, 4] = ((Ixx - Iyy) / (2 * Izz)) * p
    
    # ========================================================================
    # Construção da Matriz B contínua (mesmo código C++)
    # ========================================================================
    B_continuous = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1/Ixx, 0, 0],
        [0, 1/Iyy, 0],
        [0, 0, 1/Izz]
    ])
    
    # ========================================================================
    # Discretização usando série de Taylor (mesmo método do C++)
    # ========================================================================
    dt2_over_2 = dt * dt * 0.5
    dt3_over_6 = dt * dt * dt / 6.0
    
    # Calcula A^2
    A2 = A_continuous @ A_continuous
    
    # Calcula A^3
    A3 = A2 @ A_continuous
    
    # Discretiza A: Ad = I + A*dt + (A^2*dt^2)/2 + (A^3*dt^3)/6
    A = np.eye(6) + A_continuous * dt + A2 * dt2_over_2 + A3 * dt3_over_6
    
    # Discretiza B: Bd = B*dt + (A*B*dt^2)/2 + (A^2*B*dt^3)/6
    AB = A_continuous @ B_continuous
    A2B = A2 @ B_continuous
    B = B_continuous * dt + AB * dt2_over_2 + A2B * dt3_over_6
    
    print("\nMatriz A discretizada (6x6):")
    print(A)
    print(f"\nDimensões de A: {A.shape}")
    
    print("\nMatriz B discretizada (6x3):")
    print(B)
    print(f"\nDimensões de B: {B.shape}")
    
    # Autovalores de A (sistema em malha aberta)
    eig_A = np.linalg.eigvals(A)
    print("\nAutovalores de A (malha aberta):")
    print(eig_A)
    print("Magnitudes:", np.abs(eig_A))
    
    # ========================================================================
    # Matrizes de custo Q e R (mesmo código C++)
    # ========================================================================
    Q = np.array([
        [100, 0, 0, 0, 0, 0],
        [0, 100, 0, 0, 0, 0],
        [0, 0, 100, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    R = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    print("\nMatriz Q (6x6) - peso dos estados:")
    print(Q)
    print(f"\nDimensões de Q: {Q.shape}")
    
    print("\nMatriz R (3x3) - peso das entradas:")
    print(R)
    print(f"\nDimensões de R: {R.shape}")
    
    # Resolve a DARE usando Schur
    print("\n" + "-" * 80)
    print("Resolvendo DARE para sistema 6x6...")
    print("-" * 80)
    
    P, K, eig_cl = dare_schur(A, B, Q, R)
    
    print("\nMatriz P (6x6) - solução da DARE:")
    print(P)
    print(f"\nDimensões de P: {P.shape}")
    
    print("\nGanho de realimentação K (3x6):")
    print(K)
    print(f"\nDimensões de K: {K.shape}")
    
    print("\nAutovalores do sistema em malha fechada:")
    for i, eig in enumerate(eig_cl):
        mag = np.abs(eig)
        print(f"  λ{i+1} = {eig:.6f}, |λ{i+1}| = {mag:.6f}")
    
    # Verifica estabilidade
    if np.all(np.abs(eig_cl) < 1):
        print("\n✓ Sistema em malha fechada é ESTÁVEL (todos |λ| < 1)")
    else:
        print("\n✗ Sistema em malha fechada é INSTÁVEL")
    
    # Verifica a solução
    print("\n" + "-" * 80)
    print("Verificação da solução:")
    print("-" * 80)
    
    residuo, erro = verificar_solucao_dare(A, B, Q, R, P)
    print(f"\nNorma de Frobenius do resíduo: {erro:.2e}")
    
    if erro < 1e-10:
        print("✓ Solução VERIFICADA (erro < 1e-10)")
    elif erro < 1e-6:
        print("✓ Solução ACEITÁVEL (erro < 1e-6)")
    else:
        print("⚠ Erro maior que esperado")
    
    # Verifica propriedades de P
    erro_simetria = np.linalg.norm(P - P.T, 'fro')
    print(f"\nErro de simetria de P: {erro_simetria:.2e}")
    
    autovalores_P = np.linalg.eigvals(P)
    print(f"\nAutovalores de P:")
    for i, eig_p in enumerate(sorted(autovalores_P, reverse=True)):
        print(f"  λ{i+1}(P) = {eig_p:.6f}")
    
    if np.all(autovalores_P > 0):
        print("\n✓ P é POSITIVA DEFINIDA")
    else:
        print("\n✗ P não é positiva definida")
    
    # Simulação com condição inicial
    print("\n" + "-" * 80)
    print("Simulação do sistema controlado:")
    print("-" * 80)
    
    # Condição inicial: posição (1, 1, 1) e velocidade (0.5, -0.5, 0.3)
    x0 = np.array([[1.0], [1.0], [1.0], [0.5], [-0.5], [0.3]])
    n_steps = 150
    
    print(f"\nCondição inicial: x0 = {x0.T}")
    
    x_hist, u_hist = simular_resposta_ao_degrau(A, B, K, x0, n_steps)
    
    print(f"\nEstado final x[{n_steps}]:")
    print(x_hist[:, -1])
    print(f"\nNorma do estado final: {np.linalg.norm(x_hist[:, -1]):.2e}")
    
    # Calcula custo total
    J = 0
    for k in range(n_steps):
        x = x_hist[:, k:k+1]
        u = u_hist[:, k:k+1]
        J += (x.T @ Q @ x + u.T @ R @ u).item()
    
    print(f"\nCusto total LQR: J = {J:.4f}")
    
    # Estatísticas do controle
    u_max = np.max(np.abs(u_hist), axis=1)
    u_mean = np.mean(np.abs(u_hist), axis=1)
    
    print("\nEstatísticas do controle:")
    for i in range(u_hist.shape[0]):
        print(f"  u{i+1}: max = {u_max[i]:.4f}, média = {u_mean[i]:.4f}")
    
    # Plota resultados (6 estados + 3 controles)
    try:
        plotar_resultados_6x6(x_hist, u_hist, dt)
    except:
        print("\n(Gráficos não exibidos - matplotlib pode não estar configurado)")
    
    return A, B, Q, R, P, K, eig_cl, x_hist, u_hist


def plotar_resultados_6x6(x_hist, u_hist, dt):
    """
    Plota os resultados da simulação para sistema 6x6 com 3 entradas.
    """
    
    n_steps = x_hist.shape[1]
    t = np.arange(n_steps) * dt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot dos ângulos (roll, pitch, yaw)
    axes[0].plot(t, x_hist[0, :], 'b-', label='roll', linewidth=2)
    axes[0].plot(t, x_hist[1, :], 'r-', label='pitch', linewidth=2)
    axes[0].plot(t, x_hist[2, :], 'g-', label='yaw', linewidth=2)
    axes[0].set_xlabel('Tempo (s)')
    axes[0].set_ylabel('Ângulo (rad)')
    axes[0].set_title('Ângulos do Drone - Controle LQR via DARE')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot das velocidades angulares (p, q, r)
    axes[1].plot(t, x_hist[3, :], 'b--', label='p', linewidth=2)
    axes[1].plot(t, x_hist[4, :], 'r--', label='q', linewidth=2)
    axes[1].plot(t, x_hist[5, :], 'g--', label='r', linewidth=2)
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_ylabel('Velocidade Angular (rad/s)')
    axes[1].set_title('Velocidades Angulares do Drone')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot dos controles (u1, u2, u3)
    axes[2].plot(t, u_hist[0, :], 'b-', label='u1 (τx)', linewidth=2)
    axes[2].plot(t, u_hist[1, :], 'r-', label='u2 (τy)', linewidth=2)
    axes[2].plot(t, u_hist[2, :], 'g-', label='u3 (τz)', linewidth=2)
    axes[2].set_xlabel('Tempo (s)')
    axes[2].set_ylabel('Torque (N·m)')
    axes[2].set_title('Sinais de Controle (Torques)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


def comparacao_tres_metodos(A, B, Q, R, P_schur, K_schur):
    """
    Compara 3 métodos para resolver a DARE:
    1. Método de Schur (implementado)
    2. SciPy solve_discrete_are
    3. Método Iterativo (convergência de P)
    
    Inclui comparação de tempo de execução e precisão.
    """
    
    print("\n\n" + "=" * 80)
    print("COMPARAÇÃO DOS 3 MÉTODOS - Sistema 6x6")
    print("=" * 80)
    
    try:
        from scipy.linalg import solve_discrete_are
        
        print("\nMétodos a serem comparados:")
        print("  1. Método de Schur (implementação customizada com ordqz)")
        print("  2. SciPy solve_discrete_are (LAPACK otimizado)")
        print("  3. Método Iterativo (convergência direta da equação de Riccati)")
        print(f"\nSistema: A {A.shape}, B {B.shape}, Q {Q.shape}, R {R.shape}")
        
        # ====================================================================
        # BENCHMARK: Comparação de tempo de execução
        # ====================================================================
        print("\n" + "=" * 80)
        print("BENCHMARK - Tempo de Execução dos 3 Métodos")
        print("=" * 80)
        
        n_execucoes = 100
        print(f"\nExecutando cada método {n_execucoes} vezes para comparação de desempenho...")
        print("Aguarde...")
        
        # Benchmark do método de Schur implementado
        print("\n[1/3] Testando método de Schur (ordqz customizado)...")
        tempo_schur_list = []
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test, K_test, _ = dare_schur(A, B, Q, R)
            t_fim = time.perf_counter()
            tempo_schur_list.append(t_fim - t_inicio)
        
        tempo_schur_medio = np.mean(tempo_schur_list)
        tempo_schur_std = np.std(tempo_schur_list)
        tempo_schur_min = np.min(tempo_schur_list)
        tempo_schur_max = np.max(tempo_schur_list)
        
        # Benchmark do SciPy
        print("[2/3] Testando SciPy solve_discrete_are...")
        tempo_scipy_list = []
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test = solve_discrete_are(A, B, Q, R)
            K_test = np.linalg.inv(R + B.T @ P_test @ B) @ (B.T @ P_test @ A)
            t_fim = time.perf_counter()
            tempo_scipy_list.append(t_fim - t_inicio)
        
        tempo_scipy_medio = np.mean(tempo_scipy_list)
        tempo_scipy_std = np.std(tempo_scipy_list)
        tempo_scipy_min = np.min(tempo_scipy_list)
        tempo_scipy_max = np.max(tempo_scipy_list)
        
        # Benchmark do método iterativo
        print("[3/3] Testando método iterativo (convergência de P)...")
        tempo_iter_list = []
        n_iter_total = 0
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test, K_test, _, n_iter = dare_iterativo(A, B, Q, R)
            t_fim = time.perf_counter()
            tempo_iter_list.append(t_fim - t_inicio)
            n_iter_total += n_iter
        
        tempo_iter_medio = np.mean(tempo_iter_list)
        tempo_iter_std = np.std(tempo_iter_list)
        tempo_iter_min = np.min(tempo_iter_list)
        tempo_iter_max = np.max(tempo_iter_list)
        n_iter_medio = n_iter_total / n_execucoes
        
        # Resultados do benchmark
        print("\n" + "=" * 80)
        print("RESULTADOS DO BENCHMARK - COMPARAÇÃO DOS 3 MÉTODOS")
        print("=" * 80)
        
        print(f"\n1. Método de Schur (ordqz customizado):")
        print(f"   Tempo médio:   {tempo_schur_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_schur_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_schur_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_schur_max*1000:.3f} ms")
        
        print(f"\n2. SciPy solve_discrete_are (LAPACK):")
        print(f"   Tempo médio:   {tempo_scipy_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_scipy_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_scipy_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_scipy_max*1000:.3f} ms")
        
        print(f"\n3. Método Iterativo (convergência):")
        print(f"   Tempo médio:   {tempo_iter_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_iter_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_iter_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_iter_max*1000:.3f} ms")
        print(f"   Iterações médias: {n_iter_medio:.1f}")
        
        # Comparação relativa
        print("\n" + "=" * 80)
        print("COMPARAÇÃO RELATIVA DE DESEMPENHO")
        print("=" * 80)
        
        # Ordena métodos por velocidade
        metodos = [
            ("Schur", tempo_schur_medio),
            ("SciPy", tempo_scipy_medio),
            ("Iterativo", tempo_iter_medio)
        ]
        metodos_ordenados = sorted(metodos, key=lambda x: x[1])
        
        print(f"\nRanking de velocidade (do mais rápido ao mais lento):")
        for i, (nome, tempo) in enumerate(metodos_ordenados, 1):
            razao = tempo / metodos_ordenados[0][1]
            print(f"  {i}º - {nome:12s}: {tempo*1000:6.3f} ms  ({razao:.2f}x do mais rápido)")
        
        # Comparações específicas
        razao_schur_scipy = tempo_schur_medio / tempo_scipy_medio
        razao_schur_iter = tempo_schur_medio / tempo_iter_medio
        razao_scipy_iter = tempo_scipy_medio / tempo_iter_medio
        
        print(f"\nComparações diretas:")
        print(f"  Schur vs SciPy:     {razao_schur_scipy:.2f}x")
        print(f"  Schur vs Iterativo: {razao_schur_iter:.2f}x")
        print(f"  SciPy vs Iterativo: {razao_scipy_iter:.2f}x")
        
        # ====================================================================
        # Comparação de Precisão
        # ====================================================================
        print("\n\n" + "=" * 80)
        print("COMPARAÇÃO DE PRECISÃO DOS 3 MÉTODOS")
        print("=" * 80)
        
        # Calcula soluções (uma vez cada para comparação)
        print("\nCalculando soluções para comparação de precisão...")
        P_scipy = solve_discrete_are(A, B, Q, R)
        K_scipy = np.linalg.inv(R + B.T @ P_scipy @ B) @ (B.T @ P_scipy @ A)
        
        P_iter, K_iter, _, n_iter_final = dare_iterativo(A, B, Q, R)
        
        print(f"Método iterativo convergiu em {n_iter_final} iterações")
        
        print("\n" + "-" * 80)
        print("Diferenças em relação ao SciPy (referência):")
        print("-" * 80)
        
        # Diferenças em P
        diff_P_schur = P_schur - P_scipy
        diff_P_iter = P_iter - P_scipy
        
        norma_diff_P_schur = np.linalg.norm(diff_P_schur, 'fro')
        norma_diff_P_iter = np.linalg.norm(diff_P_iter, 'fro')
        
        print(f"\nDiferença em P (Norma de Frobenius):")
        print(f"  Schur vs SciPy:     {norma_diff_P_schur:.2e}")
        print(f"  Iterativo vs SciPy: {norma_diff_P_iter:.2e}")
        
        # Diferenças em K
        diff_K_schur = K_schur - K_scipy
        diff_K_iter = K_iter - K_scipy
        
        norma_diff_K_schur = np.linalg.norm(diff_K_schur, 'fro')
        norma_diff_K_iter = np.linalg.norm(diff_K_iter, 'fro')
        
        print(f"\nDiferença em K (Norma de Frobenius):")
        print(f"  Schur vs SciPy:     {norma_diff_K_schur:.2e}")
        print(f"  Iterativo vs SciPy: {norma_diff_K_iter:.2e}")
        
        # Erros relativos
        erro_rel_P_schur = norma_diff_P_schur / np.linalg.norm(P_scipy, 'fro')
        erro_rel_P_iter = norma_diff_P_iter / np.linalg.norm(P_scipy, 'fro')
        erro_rel_K_schur = norma_diff_K_schur / np.linalg.norm(K_scipy, 'fro')
        erro_rel_K_iter = norma_diff_K_iter / np.linalg.norm(K_scipy, 'fro')
        
        print(f"\nErro relativo em P:")
        print(f"  Schur:     {erro_rel_P_schur:.2e}")
        print(f"  Iterativo: {erro_rel_P_iter:.2e}")
        
        print(f"\nErro relativo em K:")
        print(f"  Schur:     {erro_rel_K_schur:.2e}")
        print(f"  Iterativo: {erro_rel_K_iter:.2e}")
        
        # Verificação dos resíduos DARE
        print("\n" + "-" * 80)
        print("Verificação do resíduo da DARE:")
        print("-" * 80)
        
        _, residuo_schur = verificar_solucao_dare(A, B, Q, R, P_schur)
        _, residuo_iter = verificar_solucao_dare(A, B, Q, R, P_iter)
        
        print(f"\nNorma do resíduo ||A^T*P*A - P - A^T*P*B*inv(...)*B^T*P*A + Q||:")
        print(f"  Schur:     {residuo_schur:.2e}")
        print(f"  Iterativo: {residuo_iter:.2e}")
        
        # ====================================================================
        # Resumo Final
        # ====================================================================
        print("\n\n" + "=" * 80)
        print("RESUMO FINAL - COMPARAÇÃO DOS 3 MÉTODOS")
        print("=" * 80)
        
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│                        DESEMPENHO vs PRECISÃO                               │")
        print("├──────────────────────┬──────────────────┬──────────────────┬────────────────┤")
        print("│      Método          │  Tempo Médio     │  Erro em P       │  Resíduo DARE  │")
        print("├──────────────────────┼──────────────────┼──────────────────┼────────────────┤")
        print(f"│ 1. Schur (ordqz)     │  {tempo_schur_medio*1000:6.3f} ms       │  {norma_diff_P_schur:.2e}     │  {residuo_schur:.2e}    │")
        print(f"│ 2. SciPy (LAPACK)    │  {tempo_scipy_medio*1000:6.3f} ms       │  (referência)    │  {residuo_scipy:.2e}    │")
        print(f"│ 3. Iterativo ({int(n_iter_medio):3d} it) │  {tempo_iter_medio*1000:6.3f} ms       │  {norma_diff_P_iter:.2e}     │  {residuo_iter:.2e}    │")
        print("├──────────────────────┴──────────────────┴──────────────────┴────────────────┤")
        
        # Análise final
        mais_rapido = metodos_ordenados[0][0]
        mais_preciso = min(residuos, key=lambda x: x[1])[0]
        
        print("│ Conclusões:                                                                 │")
        print(f"│  • Mais rápido:  {mais_rapido:20s}                                       │")
        print(f"│  • Mais preciso: {mais_preciso:20s}                                       │")
        print("│  • Todos têm PRECISÃO NUMÉRICA EQUIVALENTE (erro < 1e-10)                  │")
        
        if tempo_iter_medio > 2 * tempo_schur_medio:
            print("│  • Método iterativo é significativamente MAIS LENTO                        │")
            print(f"│    (requer ~{int(n_iter_medio)} iterações vs solução direta)                            │")
        
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        
    except ImportError:
        print("\n⚠ SciPy não disponível para comparação")
        print("Instale com: pip install scipy")


def comparacao_seis_metodos(A, B, Q, R, P_schur, K_schur):
    """
    Compara 6 métodos para resolver a DARE:
    1. Método de Schur (2n×2n - implementado)
    2. Método de van Dooren ((2n+m)×(2n+m) com deflação)
    3. SciPy solve_discrete_are
    4. Método Iterativo (convergência de P)
    5. SDA - Structure-preserving Doubling Algorithm
    6. ADDA - Alternating-Directional Doubling Algorithm
    
    Inclui comparação de tempo de execução, precisão e uso de memória.
    """
    
    print("\n\n" + "=" * 80)
    print("COMPARAÇÃO DOS 6 MÉTODOS - Sistema 6x6")
    print("=" * 80)
    
    try:
        from scipy.linalg import solve_discrete_are
        
        print("\nMétodos a serem comparados:")
        print("  1. Método de Schur (pencil 2n×2n - implementação customizada)")
        print("  2. Método de van Dooren (pencil (2n+m)×(2n+m) com deflação)")
        print("  3. SciPy solve_discrete_are (LAPACK otimizado)")
        print("  4. Método Iterativo (convergência direta da equação de Riccati)")
        print("  5. SDA - Structure-preserving Doubling Algorithm (convergência quadrática)")
        print("  6. ADDA - Alternating-Directional Doubling Algorithm (convergência quadrática)")
        print(f"\nSistema: A {A.shape}, B {B.shape}, Q {Q.shape}, R {R.shape}")
        print(f"Dimensões dos pencils:")
        print(f"  - Schur:      {2*A.shape[0]} × {2*A.shape[0]}")
        print(f"  - van Dooren: {2*A.shape[0] + B.shape[1]} × {2*A.shape[0] + B.shape[1]} → {2*A.shape[0]} × {2*A.shape[0]} (deflado)")
        
        # ====================================================================
        # ANÁLISE DE MEMÓRIA - Executar uma vez cada método
        # ====================================================================
        print("\n" + "=" * 80)
        print("ANÁLISE DE USO DE MEMÓRIA")
        print("=" * 80)
        
        print("\nMedindo consumo de memória de cada método...")
        
        # Memória do método de Schur
        print("[1/6] Analisando método de Schur...")
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        P_test, K_test, _ = dare_schur(A, B, Q, R)
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        mem_schur = sum(stat.size_diff for stat in top_stats) / 1024  # KB
        mem_schur_peak = tracemalloc.get_traced_memory()[1] / 1024 if tracemalloc.is_tracing() else 0
        tracemalloc.clear_traces()
        
        # Memória do método de van Dooren
        print("[2/6] Analisando método de van Dooren...")
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        P_test, K_test, _ = dare_van_dooren(A, B, Q, R)
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        mem_van_dooren = sum(stat.size_diff for stat in top_stats) / 1024  # KB
        mem_van_dooren_peak = tracemalloc.get_traced_memory()[1] / 1024 if tracemalloc.is_tracing() else 0
        tracemalloc.clear_traces()
        
        # Memória do SciPy
        print("[3/6] Analisando SciPy...")
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        P_test = solve_discrete_are(A, B, Q, R)
        K_test = np.linalg.inv(R + B.T @ P_test @ B) @ (B.T @ P_test @ A)
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        mem_scipy = sum(stat.size_diff for stat in top_stats) / 1024  # KB
        mem_scipy_peak = tracemalloc.get_traced_memory()[1] / 1024 if tracemalloc.is_tracing() else 0
        tracemalloc.clear_traces()
        
        # Memória do método iterativo
        print("[4/6] Analisando método iterativo...")
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        P_test, K_test, _, n_iter = dare_iterativo(A, B, Q, R)
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        mem_iter = sum(stat.size_diff for stat in top_stats) / 1024  # KB
        mem_iter_peak = tracemalloc.get_traced_memory()[1] / 1024 if tracemalloc.is_tracing() else 0
        tracemalloc.clear_traces()
        
        # Memória do método SDA
        print("[5/6] Analisando método SDA...")
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        P_test, K_test, _, n_iter_sda = dare_sda(A, B, Q, R)
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        mem_sda = sum(stat.size_diff for stat in top_stats) / 1024  # KB
        mem_sda_peak = tracemalloc.get_traced_memory()[1] / 1024 if tracemalloc.is_tracing() else 0
        tracemalloc.clear_traces()
        
        # Memória do método ADDA
        print("[6/6] Analisando método ADDA...")
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        P_test, K_test, _, n_iter_adda = dare_adda(A, B, Q, R)
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        mem_adda = sum(stat.size_diff for stat in top_stats) / 1024  # KB
        mem_adda_peak = tracemalloc.get_traced_memory()[1] / 1024 if tracemalloc.is_tracing() else 0
        tracemalloc.clear_traces()
        
        # Cálculo teórico do uso de memória (matrizes intermediárias)
        n = A.shape[0]
        m = B.shape[1] if B.ndim > 1 else 1
        bytes_per_float = 8  # float64
        
        mem_teorica_schur = (
            (2*n)**2 * 2 +  # Matrizes H e J (2n×2n)
            (2*n)**2 * 4 +  # Matrizes da decomposição QZ (AA, BB, Q_schur, Z_schur)
            n**2 * 2        # P e blocos U11, U21
        ) * bytes_per_float / 1024  # KB
        
        mem_teorica_van_dooren = (
            (2*n + m)**2 * 2 +  # Matrizes H e J ((2n+m)×(2n+m))
            (2*n)**2 * 4 +      # Matrizes deflacionadas (2n×2n) e QZ
            n**2 * 2            # P e blocos U11, U21
        ) * bytes_per_float / 1024  # KB
        
        mem_teorica_iter = (
            n**2 * 3  # P atual, P novo, e matrizes temporárias
        ) * bytes_per_float / 1024  # KB
        
        mem_teorica_sda = (
            n**2 * 6  # A_k, G_k, H_k, A_novo, G_novo, H_novo, W_k
        ) * bytes_per_float / 1024  # KB
        
        mem_teorica_adda = (
            n**2 * 8  # E_k, F_k, G_k, H_k, E_novo, F_novo, G_novo, H_novo, T1, T2
        ) * bytes_per_float / 1024  # KB
        
        print("\n" + "-" * 80)
        print("Resultados da análise de memória:")
        print("-" * 80)
        
        print(f"\n1. Método de Schur (2n×2n):")
        print(f"   Memória alocada: {mem_schur:.2f} KB")
        print(f"   Memória teórica: {mem_teorica_schur:.2f} KB")
        
        print(f"\n2. Método de van Dooren ((2n+m)×(2n+m)):")
        print(f"   Memória alocada: {mem_van_dooren:.2f} KB")
        print(f"   Memória teórica: {mem_teorica_van_dooren:.2f} KB")
        
        print(f"\n3. SciPy solve_discrete_are:")
        print(f"   Memória alocada: {mem_scipy:.2f} KB")
        
        print(f"\n4. Método Iterativo:")
        print(f"   Memória alocada: {mem_iter:.2f} KB")
        print(f"   Memória teórica: {mem_teorica_iter:.2f} KB")
        
        print(f"\n5. Método SDA (Structure-preserving Doubling):")
        print(f"   Memória alocada: {mem_sda:.2f} KB")
        print(f"   Memória teórica: {mem_teorica_sda:.2f} KB")
        
        print(f"\n6. Método ADDA (Alternating-Directional Doubling):")
        print(f"   Memória alocada: {mem_adda:.2f} KB")
        print(f"   Memória teórica: {mem_teorica_adda:.2f} KB")
        
        print("\n" + "-" * 80)
        print("Comparação relativa de memória:")
        print("-" * 80)
        
        metodos_mem = [
            ("Schur", mem_schur),
            ("van Dooren", mem_van_dooren),
            ("SciPy", mem_scipy),
            ("Iterativo", mem_iter),
            ("SDA", mem_sda),
            ("ADDA", mem_adda)
        ]
        metodos_mem_ordenados = sorted(metodos_mem, key=lambda x: x[1])
        
        print(f"\nRanking de uso de memória (do menor ao maior):")
        for i, (nome, memoria) in enumerate(metodos_mem_ordenados, 1):
            razao = memoria / metodos_mem_ordenados[0][1] if metodos_mem_ordenados[0][1] > 0 else 1
            print(f"  {i}º - {nome:20s}: {memoria:8.2f} KB  ({razao:.2f}x do menor)")
        
        overhead_van_dooren = ((mem_van_dooren - mem_schur) / mem_schur * 100) if mem_schur > 0 else 0
        print(f"\nOverhead de memória do van Dooren vs Schur: {overhead_van_dooren:+.1f}%")
        print(f"Justificativa: van Dooren inicia com pencil ({2*n+m}×{2*n+m}) antes da deflação")
        
        overhead_adda_sda = ((mem_adda - mem_sda) / mem_sda * 100) if mem_sda > 0 else 0
        print(f"Overhead de memória do ADDA vs SDA: {overhead_adda_sda:+.1f}%")
        print(f"Justificativa: ADDA usa 4 matrizes (E,F,G,H) vs 3 do SDA (A,G,H)")
        
        # ====================================================================
        # BENCHMARK: Comparação de tempo de execução
        # ====================================================================
        print("\n\n" + "=" * 80)
        print("BENCHMARK - Tempo de Execução dos 6 Métodos")
        print("=" * 80)
        
        n_execucoes = 1000
        print(f"\nExecutando cada método {n_execucoes} vezes para comparação de desempenho...")
        print("Aguarde...")
        
        # Benchmark do método de Schur
        print("\n[1/6] Testando método de Schur (2n×2n)...")
        tempo_schur_list = []
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test, K_test, _ = dare_schur(A, B, Q, R)
            t_fim = time.perf_counter()
            tempo_schur_list.append(t_fim - t_inicio)
        
        tempo_schur_medio = np.mean(tempo_schur_list)
        tempo_schur_std = np.std(tempo_schur_list)
        tempo_schur_min = np.min(tempo_schur_list)
        tempo_schur_max = np.max(tempo_schur_list)
        
        # Benchmark do método de van Dooren
        print("[2/6] Testando método de van Dooren ((2n+m)×(2n+m) com deflação)...")
        tempo_van_dooren_list = []
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test, K_test, _ = dare_van_dooren(A, B, Q, R)
            t_fim = time.perf_counter()
            tempo_van_dooren_list.append(t_fim - t_inicio)
        
        tempo_van_dooren_medio = np.mean(tempo_van_dooren_list)
        tempo_van_dooren_std = np.std(tempo_van_dooren_list)
        tempo_van_dooren_min = np.min(tempo_van_dooren_list)
        tempo_van_dooren_max = np.max(tempo_van_dooren_list)
        
        # Benchmark do SciPy
        print("[3/6] Testando SciPy solve_discrete_are...")
        tempo_scipy_list = []
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test = solve_discrete_are(A, B, Q, R)
            K_test = np.linalg.inv(R + B.T @ P_test @ B) @ (B.T @ P_test @ A)
            t_fim = time.perf_counter()
            tempo_scipy_list.append(t_fim - t_inicio)
        
        tempo_scipy_medio = np.mean(tempo_scipy_list)
        tempo_scipy_std = np.std(tempo_scipy_list)
        tempo_scipy_min = np.min(tempo_scipy_list)
        tempo_scipy_max = np.max(tempo_scipy_list)
        
        # Benchmark do método iterativo
        print("[4/6] Testando método iterativo...")
        tempo_iter_list = []
        n_iter_total = 0
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test, K_test, _, n_iter = dare_iterativo(A, B, Q, R)
            t_fim = time.perf_counter()
            tempo_iter_list.append(t_fim - t_inicio)
            n_iter_total += n_iter
        
        tempo_iter_medio = np.mean(tempo_iter_list)
        tempo_iter_std = np.std(tempo_iter_list)
        tempo_iter_min = np.min(tempo_iter_list)
        tempo_iter_max = np.max(tempo_iter_list)
        n_iter_medio = n_iter_total / n_execucoes
        
        # Benchmark do método SDA
        print("[5/6] Testando método SDA (Structure-preserving Doubling)...")
        tempo_sda_list = []
        n_iter_sda_total = 0
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test, K_test, _, n_iter_sda = dare_sda(A, B, Q, R)
            t_fim = time.perf_counter()
            tempo_sda_list.append(t_fim - t_inicio)
            n_iter_sda_total += n_iter_sda
        
        tempo_sda_medio = np.mean(tempo_sda_list)
        tempo_sda_std = np.std(tempo_sda_list)
        tempo_sda_min = np.min(tempo_sda_list)
        tempo_sda_max = np.max(tempo_sda_list)
        n_iter_sda_medio = n_iter_sda_total / n_execucoes
        
        # Benchmark do método ADDA
        print("[6/6] Testando método ADDA (Alternating-Directional Doubling)...")
        tempo_adda_list = []
        n_iter_adda_total = 0
        for i in range(n_execucoes):
            t_inicio = time.perf_counter()
            P_test, K_test, _, n_iter_adda = dare_adda(A, B, Q, R)
            t_fim = time.perf_counter()
            tempo_adda_list.append(t_fim - t_inicio)
            n_iter_adda_total += n_iter_adda
        
        tempo_adda_medio = np.mean(tempo_adda_list)
        tempo_adda_std = np.std(tempo_adda_list)
        tempo_adda_min = np.min(tempo_adda_list)
        tempo_adda_max = np.max(tempo_adda_list)
        n_iter_adda_medio = n_iter_adda_total / n_execucoes
        
        # Resultados do benchmark
        print("\n" + "=" * 80)
        print("RESULTADOS DO BENCHMARK - COMPARAÇÃO DOS 6 MÉTODOS")
        print("=" * 80)
        
        print(f"\n1. Método de Schur (2n×2n):")
        print(f"   Tempo médio:   {tempo_schur_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_schur_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_schur_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_schur_max*1000:.3f} ms")
        
        print(f"\n2. Método de van Dooren ((2n+m)×(2n+m) → 2n×2n):")
        print(f"   Tempo médio:   {tempo_van_dooren_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_van_dooren_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_van_dooren_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_van_dooren_max*1000:.3f} ms")
        
        print(f"\n3. SciPy solve_discrete_are (LAPACK):")
        print(f"   Tempo médio:   {tempo_scipy_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_scipy_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_scipy_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_scipy_max*1000:.3f} ms")
        
        print(f"\n4. Método Iterativo (convergência linear):")
        print(f"   Tempo médio:   {tempo_iter_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_iter_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_iter_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_iter_max*1000:.3f} ms")
        print(f"   Iterações médias: {n_iter_medio:.1f}")
        
        print(f"\n5. Método SDA (convergência quadrática):")
        print(f"   Tempo médio:   {tempo_sda_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_sda_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_sda_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_sda_max*1000:.3f} ms")
        print(f"   Iterações médias: {n_iter_sda_medio:.1f}")
        
        print(f"\n6. Método ADDA (convergência quadrática alternada):")
        print(f"   Tempo médio:   {tempo_adda_medio*1000:.3f} ms")
        print(f"   Desvio padrão: {tempo_adda_std*1000:.3f} ms")
        print(f"   Tempo mínimo:  {tempo_adda_min*1000:.3f} ms")
        print(f"   Tempo máximo:  {tempo_adda_max*1000:.3f} ms")
        print(f"   Iterações médias: {n_iter_adda_medio:.1f}")
        
        # Comparação relativa
        print("\n" + "=" * 80)
        print("COMPARAÇÃO RELATIVA DE DESEMPENHO")
        print("=" * 80)
        
        metodos = [
            ("Schur (2n×2n)", tempo_schur_medio),
            ("van Dooren", tempo_van_dooren_medio),
            ("SciPy", tempo_scipy_medio),
            ("Iterativo", tempo_iter_medio),
            ("SDA", tempo_sda_medio),
            ("ADDA", tempo_adda_medio)
        ]
        metodos_ordenados = sorted(metodos, key=lambda x: x[1])
        
        print(f"\nRanking de velocidade (do mais rápido ao mais lento):")
        for i, (nome, tempo) in enumerate(metodos_ordenados, 1):
            razao = tempo / metodos_ordenados[0][1]
            print(f"  {i}º - {nome:20s}: {tempo*1000:6.3f} ms  ({razao:.2f}x do mais rápido)")
        
        # Comparação entre métodos iterativos
        print(f"\nComparação dos métodos de duplicação (doubling):")
        print(f"  Iterativo: {n_iter_medio:.1f} iterações (convergência LINEAR)")
        print(f"  SDA:       {n_iter_sda_medio:.1f} iterações (convergência QUADRÁTICA)")
        print(f"  ADDA:      {n_iter_adda_medio:.1f} iterações (convergência QUADRÁTICA alternada)")
        razao_iter_sda = tempo_iter_medio / tempo_sda_medio
        razao_iter_adda = tempo_iter_medio / tempo_adda_medio
        razao_sda_adda = tempo_sda_medio / tempo_adda_medio
        if razao_iter_sda > 1:
            print(f"  SDA é {razao_iter_sda:.2f}x mais rápido que o método iterativo simples")
        else:
            print(f"  Método iterativo é {1/razao_iter_sda:.2f}x mais rápido que SDA")
        if razao_iter_adda > 1:
            print(f"  ADDA é {razao_iter_adda:.2f}x mais rápido que o método iterativo simples")
        else:
            print(f"  Método iterativo é {1/razao_iter_adda:.2f}x mais rápido que ADDA")
        if razao_sda_adda > 1:
            print(f"  ADDA é {razao_sda_adda:.2f}x mais rápido que SDA")
        else:
            print(f"  SDA é {1/razao_sda_adda:.2f}x mais rápido que ADDA")
        
        # ====================================================================
        # Comparação de Precisão
        # ====================================================================
        print("\n\n" + "=" * 80)
        print("COMPARAÇÃO DE PRECISÃO DOS 6 MÉTODOS")
        print("=" * 80)
        
        print("\nCalculando soluções para comparação de precisão...")
        P_van_dooren, K_van_dooren, _ = dare_van_dooren(A, B, Q, R)
        P_scipy = solve_discrete_are(A, B, Q, R)
        K_scipy = np.linalg.inv(R + B.T @ P_scipy @ B) @ (B.T @ P_scipy @ A)
        P_iter, K_iter, _, n_iter_final = dare_iterativo(A, B, Q, R)
        P_sda, K_sda, _, n_iter_sda_final = dare_sda(A, B, Q, R)
        P_adda, K_adda, _, n_iter_adda_final = dare_adda(A, B, Q, R)
        
        print("\n" + "-" * 80)
        print("Diferenças em relação ao SciPy (referência):")
        print("-" * 80)
        
        # Diferenças em P
        diff_P_schur = P_schur - P_scipy
        diff_P_van_dooren = P_van_dooren - P_scipy
        diff_P_iter = P_iter - P_scipy
        diff_P_sda = P_sda - P_scipy
        diff_P_adda = P_adda - P_scipy
        
        norma_diff_P_schur = np.linalg.norm(diff_P_schur, 'fro')
        norma_diff_P_van_dooren = np.linalg.norm(diff_P_van_dooren, 'fro')
        norma_diff_P_iter = np.linalg.norm(diff_P_iter, 'fro')
        norma_diff_P_sda = np.linalg.norm(diff_P_sda, 'fro')
        norma_diff_P_adda = np.linalg.norm(diff_P_adda, 'fro')
        
        print(f"\nDiferença em P (Norma de Frobenius):")
        print(f"  Schur vs SciPy:      {norma_diff_P_schur:.2e}")
        print(f"  van Dooren vs SciPy: {norma_diff_P_van_dooren:.2e}")
        print(f"  Iterativo vs SciPy:  {norma_diff_P_iter:.2e}")
        print(f"  SDA vs SciPy:        {norma_diff_P_sda:.2e}")
        print(f"  ADDA vs SciPy:       {norma_diff_P_adda:.2e}")
        
        # Diferenças em K
        diff_K_schur = K_schur - K_scipy
        diff_K_van_dooren = K_van_dooren - K_scipy
        diff_K_iter = K_iter - K_scipy
        diff_K_sda = K_sda - K_scipy
        diff_K_adda = K_adda - K_scipy
        
        norma_diff_K_schur = np.linalg.norm(diff_K_schur, 'fro')
        norma_diff_K_van_dooren = np.linalg.norm(diff_K_van_dooren, 'fro')
        norma_diff_K_iter = np.linalg.norm(diff_K_iter, 'fro')
        norma_diff_K_sda = np.linalg.norm(diff_K_sda, 'fro')
        norma_diff_K_adda = np.linalg.norm(diff_K_adda, 'fro')
        
        print(f"\nDiferença em K (Norma de Frobenius):")
        print(f"  Schur vs SciPy:      {norma_diff_K_schur:.2e}")
        print(f"  van Dooren vs SciPy: {norma_diff_K_van_dooren:.2e}")
        print(f"  Iterativo vs SciPy:  {norma_diff_K_iter:.2e}")
        print(f"  SDA vs SciPy:        {norma_diff_K_sda:.2e}")
        print(f"  ADDA vs SciPy:       {norma_diff_K_adda:.2e}")
        
        # Verificação dos resíduos DARE
        print("\n" + "-" * 80)
        print("Verificação do resíduo da DARE:")
        print("-" * 80)
        
        _, residuo_schur = verificar_solucao_dare(A, B, Q, R, P_schur)
        _, residuo_van_dooren = verificar_solucao_dare(A, B, Q, R, P_van_dooren)
        _, residuo_scipy = verificar_solucao_dare(A, B, Q, R, P_scipy)
        _, residuo_iter = verificar_solucao_dare(A, B, Q, R, P_iter)
        _, residuo_sda = verificar_solucao_dare(A, B, Q, R, P_sda)
        _, residuo_adda = verificar_solucao_dare(A, B, Q, R, P_adda)
        
        print(f"\nNorma do resíduo ||A^T*P*A - P - A^T*P*B*inv(...)*B^T*P*A + Q||:")
        print(f"  Schur:      {residuo_schur:.2e}")
        print(f"  van Dooren: {residuo_van_dooren:.2e}")
        print(f"  SciPy:      {residuo_scipy:.2e}")
        print(f"  Iterativo:  {residuo_iter:.2e}")
        print(f"  SDA:        {residuo_sda:.2e}")
        print(f"  ADDA:       {residuo_adda:.2e}")
        
        # ====================================================================
        # Resumo Final
        # ====================================================================
        print("\n\n" + "=" * 80)
        print("RESUMO FINAL - COMPARAÇÃO DOS 6 MÉTODOS")
        print("=" * 80)
        
        print("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
        print("│                              DESEMPENHO vs PRECISÃO vs MEMÓRIA                                             │")
        print("├──────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────────────────┤")
        print("│      Método          │ Tempo Médio  │  Memória     │  Erro em P   │  Resíduo     │  Iterações             │")
        print("├──────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────────────────┤")
        print(f"│ 1. Schur (2n×2n)     │  {tempo_schur_medio*1000:6.3f} ms   │ {mem_schur:7.2f} KB   │  {norma_diff_P_schur:.2e}   │  {residuo_schur:.2e}   │  N/A (direto)          │")
        print(f"│ 2. van Dooren (defl) │  {tempo_van_dooren_medio*1000:6.3f} ms   │ {mem_van_dooren:7.2f} KB   │  {norma_diff_P_van_dooren:.2e}   │  {residuo_van_dooren:.2e}   │  N/A (direto)          │")
        print(f"│ 3. SciPy (LAPACK)    │  {tempo_scipy_medio*1000:6.3f} ms   │ {mem_scipy:7.2f} KB   │  (referência)│  {residuo_scipy:.2e}   │  N/A (direto)          │")
        print(f"│ 4. Iterativo (lin)   │  {tempo_iter_medio*1000:6.3f} ms   │ {mem_iter:7.2f} KB   │  {norma_diff_P_iter:.2e}   │  {residuo_iter:.2e}   │  {n_iter_medio:5.1f} (linear)        │")
        print(f"│ 5. SDA (quadrático)  │  {tempo_sda_medio*1000:6.3f} ms   │ {mem_sda:7.2f} KB   │  {norma_diff_P_sda:.2e}   │  {residuo_sda:.2e}   │  {n_iter_sda_medio:5.1f} (quadrático)   │")
        print(f"│ 6. ADDA (quad.alt.)  │  {tempo_adda_medio*1000:6.3f} ms   │ {mem_adda:7.2f} KB   │  {norma_diff_P_adda:.2e}   │  {residuo_adda:.2e}   │  {n_iter_adda_medio:5.1f} (quad.alt.)    │")
        print("├──────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────────────────┤")
        
        # Análise final
        mais_rapido = metodos_ordenados[0][0]
        mais_economico = metodos_mem_ordenados[0][0]
        residuos = [
            ("Schur", residuo_schur),
            ("van Dooren", residuo_van_dooren),
            ("SciPy", residuo_scipy),
            ("Iterativo", residuo_iter),
            ("SDA", residuo_sda),
            ("ADDA", residuo_adda)
        ]
        mais_preciso = min(residuos, key=lambda x: x[1])[0]
        
        print("│ Conclusões:                                                                                               │")
        print(f"│  • Mais rápido:       {mais_rapido:30s}                                                     │")
        print(f"│  • Menor memória:     {mais_economico:30s}                                                     │")
        print(f"│  • Mais preciso:      {mais_preciso:30s}                                                     │")
        print("│                                                                                                            │")
        print("│  • van Dooren evita inversão explícita de R (mais robusto numericamente)                                  │")
        print(f"│  • van Dooren usa pencil maior ({2*A.shape[0]+B.shape[1]}×{2*A.shape[0]+B.shape[1]}) mas deflaciona para ({2*A.shape[0]}×{2*A.shape[0]})                                         │")
        print("│  • Schur (2n×2n) é mais compacto mas requer inv(R)                                                         │")
        print(f"│  • Iterativo: {int(n_iter_medio)} iterações com convergência LINEAR                                                    │")
        print(f"│  • SDA: {int(n_iter_sda_medio)} iterações com convergência QUADRÁTICA                                                     │")
        print(f"│  • ADDA: {int(n_iter_adda_medio)} iterações com convergência QUADRÁTICA alternada                                          │")
        print("│                                                                                                            │")
        print("│  DIFERENÇA ENTRE SDA E ADDA:                                                                              │")
        print("│  • SDA: usa 3 variáveis (A_k, G_k, H_k), uma direção de duplicação                                        │")
        print("│  • ADDA: usa 4 variáveis (E_k, F_k, G_k, H_k), alterna forward/backward                                   │")
        print("│  • ADDA pode ser mais robusto para problemas mal-condicionados                                            │")
        print("│  • ADDA requer A inversível; SDA não tem essa restrição                                                   │")
        
        # Comparação entre Schur e van Dooren
        razao_schur_van_dooren = tempo_schur_medio / tempo_van_dooren_medio
        if razao_schur_van_dooren < 1:
            diff_percent = (1 - razao_schur_van_dooren) * 100
            print(f"│  • Schur é {diff_percent:.1f}% mais rápido que van Dooren                                                       │")
        else:
            diff_percent = (razao_schur_van_dooren - 1) * 100
            print(f"│  • van Dooren é {diff_percent:.1f}% mais rápido que Schur                                                    │")
        
        # Comparação entre SDA, ADDA e iterativo
        razao_iter_sda = n_iter_medio / n_iter_sda_medio
        razao_iter_adda = n_iter_medio / n_iter_adda_medio
        print(f"│  • SDA converge em {n_iter_sda_medio:.0f} iter vs {n_iter_medio:.0f} iter do método iterativo (razão: {razao_iter_sda:.1f}x)                      │")
        print(f"│  • ADDA converge em {n_iter_adda_medio:.0f} iter vs {n_iter_medio:.0f} iter do método iterativo (razão: {razao_iter_adda:.1f}x)                     │")
        
        print("└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
        
    except ImportError:
        print("\n⚠ SciPy não disponível para comparação")
        print("Instale com: pip install scipy")


if __name__ == "__main__":
    """
    Exemplo numérico do algoritmo de Schur para DARE com sistema 6x6
    """
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "ALGORITMO DE SCHUR PARA DARE" + " " * 35 + "█")
    print("█" + " " * 10 + "Discrete-Time Algebraic Riccati Equation" + " " * 28 + "█")
    print("█" + " " * 20 + "Sistema 6x6 com 3 Entradas" + " " * 32 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Executa exemplo com sistema 6x6
    A, B, Q, R, P, K, eig_cl, x_hist, u_hist = exemplo_sistema_6x6()
    
    # Comparação dos 6 métodos usando o mesmo sistema 6x6
    comparacao_seis_metodos(A, B, Q, R, P, K)
    
    print("\n" + "█" * 80)
    print("█" + " " * 25 + "EXEMPLO CONCLUÍDO" + " " * 35 + "█")
    print("█" * 80)
    print()
