#ifndef AUTO_LQR_H
#define AUTO_LQR_H

#include <Arduino.h>
#include "MatrixOperations.h"

class AutoLQR : public MatrixOperations {
public:
    /**
     * @brief Construct a new AutoLQR controller
     * @param stateSize Number of state variables
     * @param controlSize Number of control inputs
     */
    AutoLQR(int stateSize, int controlSize);

    /**
     * @brief Destroy the AutoLQR controller and free memory
     */
    ~AutoLQR();

    /**
     * @brief Set the system state matrix A
     * @param A Pointer to state matrix (stateSize x stateSize)
     * @return true if successful, false otherwise
     */
    bool setStateMatrix(const float* A);

    /**
     * @brief Set the input matrix B
     * @param B Pointer to input matrix (stateSize x controlSize)
     * @return true if successful, false otherwise
     */
    bool setInputMatrix(const float* B);

    /**
     * @brief Set the cost matrices Q and R
     * @param Q Pointer to state cost matrix (stateSize x stateSize)
     * @param R Pointer to control cost matrix (controlSize x controlSize)
     * @return true if successful, false otherwise
     */
    bool setCostMatrices(const float* Q, const float* R);

    /**
     * @brief Compute optimal feedback gains
     * @return true if successful, false if computation fails
     */
    bool computeGains(const char* method = "SDA_FIXED");

    /**
     * @brief Update the controller with current state
     * @param currentState Pointer to current state vector (stateSize)
     */
    void updateState(const float* currentState);

    /**
     * @brief Calculate control inputs based on current state
     * @param controlOutput Pointer to control output vector (controlSize)
     */
    void calculateControl(float* controlOutput);

    /**
     * @brief Set pre-computed gain values
     * @param K Pointer to gain matrix (controlSize x stateSize)
     */
    void setGains(const float* K);

    /**
     * @brief Check if the system is controllable
     * @return true if controllable, false otherwise
     */
    bool isSystemControllable();

    /**
     * @brief Get the solution of the Riccati equation
     * @return Pointer to the P matrix (stateSize x stateSize)
     */
    const float* getRicattiSolution() const;

    /**
     * @brief Estimate feedforward gain for steady-state tracking
     * @param ffGain Pointer to feedforward gain vector (controlSize)
     * @param desiredState Pointer to desired state vector (stateSize)
     */
    void estimateFeedforwardGain(float* ffGain, const float* desiredState);

    /**
     * @brief Estimate time to convergence
     * @param convergenceThreshold Threshold for considering system converged (default: 0.05)
     * @return Estimated time in seconds, or -1 if estimation fails
     */
    float estimateConvergenceTime(float convergenceThreshold = 0.05f);

    /**
     * @brief Export computed gains to external array
     * @param exportedK Pointer to destination array (controlSize x stateSize)
     * @return true if successful, false otherwise
     */
    bool exportGains(float* exportedK);

    /**
     * @brief Calculate expected cost from current state
     * @return Expected cost value, or -1 if calculation fails
     */
    float calculateExpectedCost();

    /**
     * @brief Compute Kr gains
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixKr();

    /**
     * @brief Export Kr gains to external array
     * @param exportedKr Pointer to destination array
     * @return true if successful, false otherwise
     */
    bool exportKr(float* exportedKr);

    /**
     * @brief Update the reference values
     * @param newReference Pointer to new reference vector
     */
    void updateReference(const float* newReference);

    /**
     * @brief Get the number of iterations used in last computation
     * @return Number of iterations, or -1 if not available
     */
    int getLastIterations() const;

    /**
     * @brief Get the final residual from last computation
     * @return Norma do resíduo REAL da DARE, ‖A'PA-P-A'PB(R+B'PB)^-1B'PA+Q‖_F/‖Q‖_F,
     *         calculada preguiçosamente (só na primeira chamada após um solve — ver
     *         residualDirty) — não o critério interno de parada (ver getLastStepDelta()).
     *         -1 se P/A/B/Q/R indisponíveis, -2 se R+B'PB singular.
     * @note Chamar DEPOIS de computeGains() e ANTES de qualquer nova setStateMatrix()/
     *       setInputMatrix()/setCostMatrices() — o cálculo preguiçoso lê os A/B/Q/R/P
     *       atuais no momento da CHAMADA, não do solve; mutar as matrizes entre os dois
     *       invalida o resíduo reportado (mesma convenção que os benchmarks já seguem).
     */
    float getLastResidual() const;

    /**
     * @brief Get the internal stopping-criterion delta from last computation
     * @return ‖H_{k+1}-H_k‖_F/‖H_k‖_F da última iteração (critério interno de parada
     *         de cada solver — NÃO é o resíduo da DARE, ver getLastResidual()).
     */
    float getLastStepDelta() const;

    /**
     * @brief Maior |valor| real visto em qualquer produto matricial da última chamada
     *        bem-sucedida de um método `_FIXED` — mede a margem até o teto ±8192 do
     *        Q13.18 (ver docs/auditoria_solvers_riccati.md, Seção 9/10).
     * @return Magnitude em unidades reais (não o inteiro Q-format). 0 se o último método
     *         chamado não foi `_FIXED`, se falhou, ou se compilado sem -DFXQ_INSTRUMENT
     *         (o env de voo não paga esse custo por padrão — ver FixedPointQ.cpp).
     */
    float getLastFixedPointMaxAbsSeen() const;

    /**
     * @brief Get residuals history for first iterations
     * @param residuals Output array to store residuals (size MAX_RESIDUAL_HISTORY)
     * @return Number of valid residuals stored
     */
    int getResidualHistory(float* residuals) const;

    static const int MAX_RESIDUAL_HISTORY = 10; ///< Maximum number of residuals to store

private:
    int stateSize; ///< Number of state variables
    int controlSize; ///< Number of control inputs

    float* A; ///< State matrix
    float* B; ///< Input matrix
    float* Q; ///< State cost matrix
    float* R; ///< Control cost matrix
    float* K; ///< Control gain matrix
    float* state; ///< Current state
    float* P; ///< Riccati equation solution (publicado por qualquer método, via getRicattiSolution)
    float* P_warm; ///< Buffer de warm-start dedicado ao computeGainMatrixIterative() —
                   ///< não compartilha estado com P, para não herdar a solução de outro método
    float* Kr; ///< Kr gain matrix
    float* reference; ///< To store reference values
    
    int lastIterations; ///< Number of iterations in last computation
    mutable float lastResidual; ///< Resíduo REAL da DARE (ver getLastResidual()) — cache preguiçoso
    mutable bool residualDirty; ///< true = lastResidual precisa ser recalculado na próxima
                                 ///< getLastResidual(). Tira computeDareResidualNorm() (11
                                 ///< new[]/delete[] + 7 matmuls) do caminho de voo, que nunca lê
                                 ///< o resíduo — só quem chama getLastResidual() paga o custo,
                                 ///< e só uma vez por solve (ver docs/auditoria_solvers_riccati.md,
                                 ///< Seção 12).
    float lastStepDelta; ///< Critério interno de parada da última iteração (ver getLastStepDelta())
    float lastFixedPointMaxAbsSeen; ///< Maior |valor| real visto em qualquer matmul_q na última
                                     ///< chamada bem-sucedida de um método _FIXED (unidades reais,
                                     ///< não o inteiro Q-format; 0 se o método não é _FIXED, se
                                     ///< falhou, ou se compilado sem -DFXQ_INSTRUMENT). Mede a
                                     ///< margem até o teto ±8192 do Q13.18 — ver getLastFixedPointMaxAbsSeen().
    float residualHistory[10]; ///< Residuals for first 10 iterations
    int residualHistoryCount; ///< Number of valid entries in residualHistory

    /**
     * @brief Resíduo real da DARE para o (A,B,Q,R,P) atuais
     * @return ‖A'PA-P-A'PB(R+B'PB)^-1B'PA+Q‖_F/‖Q‖_F; -1 se dados indisponíveis,
     *         -2 se R+B'PB for singular.
     * @note Chamado uma vez ao final de cada computeGainMatrixXXX() bem-sucedido,
     *       para preencher lastResidual com uma métrica auditável (ver
     *       docs/auditoria_solvers_riccati.md) — o critério interno ‖ΔH‖/‖H‖ usado
     *       para decidir a parada de cada solver fica em lastStepDelta.
     */
    float computeDareResidualNorm() const;


    /**
     * @brief Decomposição de Schur do pencil simplético (direto, não iterativo).
     * Usa Eigen::ComplexSchur **sem reordenação** — a seleção de colunas do
     * subespaço estável por índice diagonal não garante base do subespaço
     * invariante correto (defeito conhecido, não corrigido nesta auditoria —
     * ver docs/auditoria_solvers_riccati.md). Órfão: nenhum chamador no
     * repositório.
     * Referência: Laub, A.J. "A Schur Method for Solving Algebraic Riccati
     * Equations." IEEE Trans. Automat. Control 24(6):913-921, 1979.
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixSchur();

    /**
     * @brief Método direto via pencil estendido (2n+m)×(2n+m) com deflação QR
     * e autoproblema generalizado — evita inverter R mesmo se singular.
     * Referência: Van Dooren, P. "A Generalized Eigenvalue Approach for
     * Solving Riccati Equations." SIAM J. Sci. Stat. Comput. 2(2):121-135,
     * 1981 (Problema II, DARE discreta).
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixVanDooren();

    /**
     * @brief Structure-preserving Doubling Algorithm (SDA) — referência exata
     * em float, convergência quadrática (~8-10 iterações em regime).
     * Referência: Chu, E.K.-W., Fan, H.-Y., Lin, W.-W., Wang, C.-S.
     * "Structure-preserving algorithms for periodic discrete-time algebraic
     * Riccati equations." Int. J. Control 77(8):767-788, 2004 (box "SDA
     * algorithm", p.770). Origem: Anderson, B.D.O. "Second-order convergent
     * algorithms for the steady-state Riccati equation." IEEE CDC, 1978
     * (eqs. 4a-4c).
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixSDA();

    /**
     * @brief Caminho rápido do SDA em fixed-point Q13.18 (ESP32-S2 sem FPU)
     * Resolve a DARE em int32 (~2.7× mais rápido que float, erro do K < 1%).
     * Validado só para o caso 6 estados / 3 controles. Retorna false em
     * overflow/saturação ou matriz singular → o chamador faz fallback p/ o SDA float.
     * Mesma recorrência de computeGainMatrixSDA() (mesma referência), só a
     * aritmética (fixed-point Q13.18) é escolha de engenharia deste projeto.
     * @return true se sucesso, false se deve cair no fallback float
     */
    bool computeGainMatrixSDA_Fixed();

    /**
     * @brief Iteração de ponto-fixo direta na DARE (value iteration clássica),
     * P_{k+1}=Q+A'P_kA-A'P_kB(R+B'P_kB)^-1B'P_kA, com warm-start dedicado
     * (P_warm) e regularização diagonal 1e-8 em (R+B'PB). Convergência
     * **linear** (não quadrática como o SDA) — não está em nenhum dos 8
     * papers de SOLVERS/, é a iteração elementar descrita em, e.g.,
     * Lancaster, P., Rodman, L. "The Algebraic Riccati Equation." Oxford
     * University Press, 1995 (cap. 2).
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixIterative();
    
    /**
     * @brief SDA com shift real único γ, deslocando o pencil simplético inteiro
     * (não só A) para preservar P — ver comentário de implementação em AutoLQR.cpp
     * para a derivação e a correção do bug da versão anterior (shift em A
     * isolado, que resolvia uma DARE diferente da original).
     * Referência: Chu, Fan, Lin & Wang, "Structure-preserving algorithms for
     * periodic discrete-time algebraic Riccati equations", Int. J. Control
     * 77(8):767-788, 2004 (técnica de shift generalizada em Chu, Fan & Lin,
     * Linear Algebra Appl. 396:55-80, 2005).
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixSDA_SS();

    /**
     * @brief SDA com escalonamento adaptativo (G,H)->(sG,H/s) a cada iteração,
     * revertido no final via P = H_k·∏s_i — ver comentário de implementação em
     * AutoLQR.cpp para a correção do bug da versão anterior (não revertia o
     * produto acumulado).
     * Referência: Chu, E.K.-W., Fan, H.-Y., Lin, W.-W. "A structure-preserving
     * doubling algorithm for continuous-time algebraic Riccati equations."
     * Linear Algebra Appl. 396:55-80, 2005 (mesmo paper de computeGainMatrixSDA_SS()
     * — cobre tanto o shift γ quanto a técnica de escalonamento adaptativo para
     * robustez numérica do SDA). Conteúdo integral do paper não verificado
     * diretamente por mim (acesso ao texto completo bloqueado nas fontes
     * consultadas); citação confirmada pelo usuário.
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixASDA();

    /**
     * @brief SDA com balanceamento diagonal D do pencil simplético (Â=DAD⁻¹,
     * Ĝ=DGD, Ĥ=D⁻¹QD⁻¹, P=D·P̂·D) — ver comentário de implementação em
     * AutoLQR.cpp para a correção dos expoentes de D da versão anterior.
     * Referência: Ward, "Balancing the Generalized Eigenvalue Problem", SIAM
     * J. Sci. Stat. Comput. 2(2):141-152, 1981.
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixSDA_Scaled();

    /**
     * @brief SDA em forma alternada (calcula V=(I+GH)^-1 e usa a identidade
     * push-through (I+HG)^-1 H ≡ H(I+GH)^-1 para a atualização de H). Mantido
     * pelo nome "ADDA" por compatibilidade com o dispatcher, mas é
     * algebricamente idêntico ao SDA — ver comentário de implementação em
     * AutoLQR.cpp. O ADDA de fato (dois parâmetros de shift α≠β sobre a MARE
     * não-simétrica de M-matriz) não está implementado.
     * Referência correta: Wang, Wang & Li, "Alternating-Directional Doubling
     * Algorithm for M-Matrix Algebraic Riccati Equations", SIAM J. Matrix
     * Anal. Appl. 33(1):170-194, 2012.
     * @return true if successful, false otherwise
     */
    bool computeGainMatrixADDA();

    // ------------------------------------------------------------------
    // Variantes em fixed-point Q13.18 (ESP32-S2 sem FPU), via o kernel
    // compartilhado em FixedPointQ.{h,cpp}. Mesmo gate n==6,m==3 e mesma
    // convenção de fallback do SDA_FIXED (retornam false em overflow/
    // saturação ou matriz singular; sem fallback automático para float).
    // Referências bibliográficas: as mesmas do par float correspondente
    // (ver docs/auditoria_solvers_riccati.md) — a aritmética Q13.18 em si
    // é engenharia própria deste projeto, sem paper específico.
    // ------------------------------------------------------------------

    /**
     * @brief SDA_SS em fixed-point. Setup do pencil 12×12 no mesmo shift
     * Q13.18 do resto — a faixa limitante é Φ=N1⁻¹, não N1, e seu tamanho
     * varia demais entre casos (mediana ~738, até ~46000 em cenários
     * adversariais) para um shift único cobrir tudo sem perder resolução
     * onde importa; Q13.18 cobre a esmagadora maioria dos casos reais (ver
     * comentário de implementação e docs/auditoria_solvers_riccati.md).
     * @return true se sucesso, false se deve cair no fallback float
     */
    bool computeGainMatrixSDA_SS_Fixed();

    /**
     * @brief ASDA em fixed-point. Reescalonamento (G,H)->(sG,H/s) por
     * iteração; a norma e o fator de escala são calculados em float (poucas
     * chamadas de sqrtf, desprezível frente às matmuls), os dados de
     * A/G/H permanecem inteiros o tempo todo.
     * @return true se sucesso, false se deve cair no fallback float
     */
    bool computeGainMatrixASDA_Fixed();

    /**
     * @brief SDA_SCALED em fixed-point. Balanceamento diagonal D por normas
     * de linha de A, calculado em Q13.18 (setup barato: D é diagonal).
     * @return true se sucesso, false se deve cair no fallback float
     */
    bool computeGainMatrixSDA_Scaled_Fixed();

    /**
     * @brief ADDA em fixed-point na forma V/W com duas inversões separadas
     * (mantida deliberadamente, ao contrário do par float que já usa só V
     * via push-through) — mede se a ordem de multiplicação W·H vs. H·V
     * quantiza diferente em Q13.18.
     * @return true se sucesso, false se deve cair no fallback float
     */
    bool computeGainMatrixADDA_Fixed();

    /**
     * @brief Iterativo (value iteration) em fixed-point. Convergência linear
     * — já não converge em 100 iterações float para boa parte dos casos da
     * bateria. Compartilha o warm-start (P_warm, float) com
     * computeGainMatrixIterative() — mesmo P físico, aritmética diferente;
     * chamadas sucessivas (float ou fixed) partem do resultado da anterior.
     * @return true se sucesso, false se deve cair no fallback float
     */
    bool computeGainMatrixIterative_Fixed();
};

#endif
