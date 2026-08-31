#include "AutoLQR.h"
#include "FixedPointQ.h"
#include <math.h>
#include <ArduinoEigen.h>
#include <string.h>  // Para strcmp
#include <algorithm> // Para std::sort
#include <stdint.h>

AutoLQR::AutoLQR(int stateSize, int controlSize)
    : stateSize(stateSize)
    , controlSize(controlSize)
    , A(nullptr)
    , B(nullptr)
    , Q(nullptr)
    , R(nullptr)
    , K(nullptr)
    , state(nullptr)
    , P(nullptr)
    , P_warm(nullptr)
    , Kr(nullptr)
    , reference(nullptr)
    , relTolerance(1e-3f)
    , maxIterations(100)
    , invRelTolerance(1000) // round(1/1e-3) — mantém em sincronia com relTolerance acima
    , lastOutcome(SolveOutcome::Converged)
    , ssGamma(0.7f) // medido, nao suposto — ver comentario em computeGainMatrixSDA_SS()
    , lastIterations(-1)
    , lastResidual(-1.0f)
    , residualDirty(false)
    , lastStepDelta(-1.0f)
    , lastStepIsBitExactZero(false)
    , lastFixedPointMaxAbsSeen(0.0f)
    , residualHistoryCount(0)
{
    // Inicializar histórico de resíduos
    for (int i = 0; i < 10; i++) {
        residualHistory[i] = 0.0f;
    }
    
    if (stateSize > 0 && controlSize > 0) {
        A = new float[stateSize * stateSize]();
        B = new float[stateSize * controlSize]();
        Q = new float[stateSize * stateSize]();
        R = new float[controlSize * controlSize]();
        K = new float[controlSize * stateSize]();
        state = new float[stateSize]();
        P = new float[stateSize * stateSize]();
        P_warm = new float[stateSize * stateSize]();
        Kr = new float[controlSize * controlSize]();
        reference = new float[controlSize]();
    }
}

AutoLQR::~AutoLQR()
{
    delete[] A;
    delete[] B;
    delete[] Q;
    delete[] R;
    delete[] K;
    delete[] state;
    delete[] P;
    delete[] P_warm;
    delete[] Kr;
    delete[] reference;
}

bool AutoLQR::setStateMatrix(const float* inputA)
{
    if (!inputA || !A)
        return false;
    matrixCopy(inputA, A, stateSize * stateSize);
    return true;
}

bool AutoLQR::setInputMatrix(const float* inputB)
{
    if (!inputB || !B)
        return false;
    matrixCopy(inputB, B, stateSize * controlSize);
    return true;
}

bool AutoLQR::setCostMatrices(const float* inputQ, const float* inputR)
{
    if (!inputQ || !inputR || !Q || !R)
        return false;
    matrixCopy(inputQ, Q, stateSize * stateSize);
    matrixCopy(inputR, R, controlSize * controlSize);
    return true;
}

void AutoLQR::setGains(const float* inputK)
{
    if (!inputK || !K)
        return;
    matrixCopy(inputK, K, controlSize * stateSize);
}

void AutoLQR::setStoppingCriterion(float relTol, int maxIters)
{
    if (relTol > 0.0f) {
        relTolerance = relTol;
        long r = lroundf(1.0f / relTol);
        invRelTolerance = (r < 1) ? 1 : (int)r; // ver FixedPointQ.cpp:doubling_loop_q
    }
    if (maxIters > 0) maxIterations = maxIters;
}

AutoLQR::SolveOutcome AutoLQR::getLastOutcome() const
{
    return lastOutcome;
}

void AutoLQR::setSDASSGamma(float gamma)
{
    if (gamma > 0.0f && gamma < 1.0f) ssGamma = gamma;
}

bool AutoLQR::computeGains(const char* method)
{
    bool K_flag = false;

    // Reset p/ sentinelas ANTES do despacho — Seção 15 da auditoria, "Achado 3":
    // lastIterations/lastResidual/lastFixedPointMaxAbsSeen só eram escritos no
    // caminho de sucesso de cada método _FIXED; todo `return false` antecipado
    // os deixava com o valor da chamada anterior (mesma instância persistente).
    // lastOutcome tinha o mesmo problema em SCHUR/VAN_DOOREN, que nunca o
    // tocam (nem sequer foram retrofitados com a taxonomia SolveOutcome) —
    // getLastOutcome() após um deles reportava o desfecho de uma chamada
    // anterior não relacionada. lastOutcome é reforçado para Converged logo
    // após o despacho quando K_flag==true (linha abaixo), o que é redundante
    // para os métodos que já se auto-reportam corretamente e CORRIGE
    // SCHUR/VAN_DOOREN de graça.
    lastIterations = -1;
    lastResidual = -1.0f;
    residualDirty = false;
    lastFixedPointMaxAbsSeen = 0.0f;
    lastOutcome = SolveOutcome::Breakdown; // sentinela: "método não se reportou"

    if (strcmp(method, "SDA") == 0) {
        K_flag = computeGainMatrixSDA(); 
    } else if (strcmp(method, "SDA_FIXED") == 0) {
        K_flag = computeGainMatrixSDA_Fixed(); // caminho rápido fixed-point Q13.18
    } else if (strcmp(method, "SDA_SS") == 0) {
        K_flag = computeGainMatrixSDA_SS();
    } else if (strcmp(method, "ASDA") == 0) {
        K_flag = computeGainMatrixASDA();
    } else if (strcmp(method, "SDA_SCALED") == 0) {
        K_flag = computeGainMatrixSDA_Scaled();
    } else if (strcmp(method, "SCHUR") == 0) {
        K_flag = computeGainMatrixSchur();
    } else if (strcmp(method, "VAN_DOOREN") == 0) {
        K_flag = computeGainMatrixVanDooren();
    } else if (strcmp(method, "ITERATIVE") == 0) {
        K_flag = computeGainMatrixIterative();
    } else if (strcmp(method, "ADDA") == 0) {
        K_flag = computeGainMatrixADDA();
    } else if (strcmp(method, "SDA_SS_FIXED") == 0) {
        K_flag = computeGainMatrixSDA_SS_Fixed();
    } else if (strcmp(method, "ASDA_FIXED") == 0) {
        K_flag = computeGainMatrixASDA_Fixed();
    } else if (strcmp(method, "SDA_SCALED_FIXED") == 0) {
        K_flag = computeGainMatrixSDA_Scaled_Fixed();
    } else if (strcmp(method, "ADDA_FIXED") == 0) {
        K_flag = computeGainMatrixADDA_Fixed();
    } else if (strcmp(method, "ITERATIVE_FIXED") == 0) {
        K_flag = computeGainMatrixIterative_Fixed();
    } else {
        // Método desconhecido: cai no default de computeGains(), SDA_FIXED
        Serial.print(F("Método desconhecido: "));
        Serial.print(method);
        Serial.println(F(". Usando SDA_FIXED."));
        K_flag = computeGainMatrixSDA_Fixed();
    }

    if (!K_flag)
        return false;

    // Sucesso: reforça Converged mesmo para métodos que não usam a taxonomia
    // (SCHUR, VAN_DOOREN — solvers diretos, sem noção de "orçamento
    // esgotado"). No-op para os métodos que já se auto-reportam.
    lastOutcome = SolveOutcome::Converged;

    // computeGainMatrixKr() só formata Kr a partir de K já calculado — seu
    // retorno não deve mascarar o sucesso/falha do cálculo do ganho principal.
    computeGainMatrixKr();
    return K_flag;
}

void AutoLQR::updateState(const float* currentState)
{
    if (!currentState || !state)
        return;
    matrixCopy(currentState, state, stateSize);
}

void AutoLQR::updateReference(const float* newReference)
{
    if (!newReference || !reference)
        return;
    matrixCopy(newReference, reference, controlSize);
}

void AutoLQR::calculateControl(float* controlOutput)
{
    if (!controlOutput || !K || !state || !Kr || !reference)
        return;

    // Initialize control outputs to zero
    matrixClear(controlOutput, controlSize);

    // u = -K·x + Kr·r
    for (int i = 0; i < controlSize; i++) {
        for (int j = 0; j < stateSize; j++) {
            controlOutput[i] -= K[i * stateSize + j] * state[j];
            if(j < controlSize) {
                controlOutput[i] += Kr[i * controlSize + j] * reference[j];
            }
        }
    }
}

bool AutoLQR::isSystemControllable()
{
    // Basic controllability check for 2x2 systems
    if (stateSize == 2 && controlSize == 1) {
        float det = B[0] * A[1] - B[1] * A[0];
        return fabs(det) > 1e-6;
    }

    // For larger systems, implement a more sophisticated controllability check
    // or return true and let the DARE solver determine feasibility
    return true;
}

const float* AutoLQR::getRicattiSolution() const
{
    return P;
}

bool AutoLQR::computeGainMatrixSDA_Fixed()
{
    // Caminho rápido do SDA em fixed-point Q13.18 (ESP32-S2 sem FPU).
    // Retorna false → chamador faz fallback p/ o SDA float quando há
    // overflow/saturação ou matriz singular no domínio fixed-point.
    // Setup + laço + extração equivalentes ao computeGainMatrixSDA() float,
    // usando o kernel compartilhado em FixedPointQ (ver docs/auditoria_solvers_riccati.md).
    using namespace fxq;
    const int n = stateSize, m = controlSize;
    const int sh = Q_SHIFT_DEFAULT;

    if (!A || !B || !Q || !R || !K)
        return false;
    if (n != 6 || m != 3)          // buffers e validação dimensionados p/ o caso 6x3
        return false;
    if (!isSystemControllable())
        return false;

    Status st;
    q_t Aq[36], Bq[18], Qq[36], Rq[9];   // stack (one-shot por chamada)

    for (int i = 0; i < n * n; i++) { Aq[i] = f2q(A[i], sh, &st); Qq[i] = f2q(Q[i], sh, &st); }
    for (int i = 0; i < n * m; i++)  Bq[i] = f2q(B[i], sh, &st);
    for (int i = 0; i < m * m; i++)  Rq[i] = f2q(R[i], sh, &st);
    // BUG CORRIGIDO 2026-08-18: esta era a ÚNICA saída por overflow em todo o
    // arquivo sem marcar lastOutcome (as equivalentes estão em :241 e nos quatro
    // outros métodos _FIXED). Como lastOutcome é membro persistente e
    // computeGains() não o reseta, este return devolvia o desfecho da chamada
    // ANTERIOR — junto com lastIterations/lastResidual/lastFixedPointMaxAbsSeen,
    // igualmente não tocados. Efeito medido na varredura Q/R: em R_scale>=1e3,
    // onde Rd satura o teto ±8192 já aqui na conversão da entrada, as 6000
    // execuções do SDA_FIXED reportaram telemetria congelada e idêntica
    // (iters=15, resid=8.520439e-03, outcome=0) em ~170 µs, produzindo a
    // conclusão FALSA de que o SDA_FIXED seria imune ao modo de falha em R
    // grande. Todos os cinco métodos _FIXED abortam aqui, identicamente.
    // Ver docs/auditoria_solvers_riccati.md, Seção 15.
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; } // entrada fora do range ±8192

    q_t BT[18], Rinv[9], BRi[18], Gk[36], Hk[36], Ak[36];
    transpose_q(Bq, BT, n, m);
    if (!invert_q(Rq, Rinv, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
    matmul_q(Bq, Rinv, BRi, n, m, m, sh, &st);
    matmul_q(BRi, BT, Gk, n, m, n, sh, &st);
    memcpy(Hk, Qq, sizeof(Hk));
    memcpy(Ak, Aq, sizeof(Ak));

    if (!doubling_loop_q(Ak, Gk, Hk, n, sh, Variant::Standard, maxIterations, invRelTolerance, nullptr, &st)) {
        lastOutcome = SolveOutcome::Breakdown; return false; // singular no domínio fixed-point → fallback
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; } // saturou durante o laço → fallback

    // K = (R + B'PB)^-1 B'PA, com A ORIGINAL (Aq), não Ak (mutado pelo laço)
    q_t BTP[18], BTPB[9], Rp[9], BTPA[18], Kq[18];
    matmul_q(BT, Hk, BTP, m, n, n, sh, &st);
    matmul_q(BTP, Bq, BTPB, m, n, m, sh, &st);
    add_q(Rq, BTPB, Rp, m * m);
    if (!invert_q(Rp, Rp, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
    matmul_q(BTP, Aq, BTPA, m, n, n, sh, &st);
    matmul_q(Rp, BTPA, Kq, m, m, n, sh, &st);
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    for (int i = 0; i < m * n; i++) K[i] = q2f(Kq[i], sh);
    if (P) for (int i = 0; i < n * n; i++) P[i] = q2f(Hk[i], sh);

    lastIterations = st.iterations;
    lastStepDelta = st.rel_step;
    lastStepIsBitExactZero = st.bit_exact_zero;
    // doubling_loop_q não retorna false por esgotar o orçamento (só por
    // singularidade — ver FixedPointQ.cpp), então st.iterations==maxIterations
    // sem overflow é censura por orçamento (Budget), não convergência. Antes
    // desta mudança, computeGains() dos métodos _FIXED retornava sempre true
    // aqui — inflava artificialmente "0 falhas" em execuções censuradas por
    // orçamento (ver docs/auditoria_solvers_riccati.md, Seção 13). Agora o
    // retorno booleano tem o MESMO significado nos dois caminhos: true só
    // quando SolveOutcome::Converged.
    lastOutcome = (st.iterations < maxIterations) ? SolveOutcome::Converged : SolveOutcome::Budget;
    residualHistoryCount = 0; // kernel fixed-point não expõe resíduo por iteração
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;
    residualDirty = true; // resíduo real calculado sob demanda em getLastResidual()
    lastFixedPointMaxAbsSeen = fxq::q2f(st.max_abs_seen, sh);
    return lastOutcome == SolveOutcome::Converged;
}

// Resíduo REAL da DARE para o (A,B,Q,R,P) atuais — independente do critério
// interno de parada de cada solver (ver AutoLQR.h: lastResidual vs lastStepDelta).
float AutoLQR::computeDareResidualNorm() const
{
    if (!A || !B || !Q || !R || !P) return -1.0f;

    const int n = stateSize;
    const int m = controlSize;

    float* AT    = new float[n * n];
    float* BT    = new float[m * n];
    float* PA    = new float[n * n];
    float* ATPA  = new float[n * n];
    float* PB    = new float[n * m];
    float* BTPB  = new float[m * m];
    float* BTPA  = new float[m * n];
    float* S     = new float[m * m];
    float* SinvBTPA = new float[m * n];
    float* ATPB  = new float[n * m];
    float* corr  = new float[n * n];

    transposeMatrix(A, AT, n, n);
    transposeMatrix(B, BT, n, m);

    matrixMultiply(P, A, PA, n, n, n);
    matrixMultiply(AT, PA, ATPA, n, n, n);
    matrixMultiply(P, B, PB, n, n, m);
    matrixMultiply(BT, PB, BTPB, m, n, m);
    matrixMultiply(BT, PA, BTPA, m, n, n);

    matrixAdd(R, BTPB, S, m, m);

    float result;
    if (!invertMatrix(S, S, m)) {
        result = -2.0f; // R + B'PB singular
    } else {
        matrixMultiply(S, BTPA, SinvBTPA, m, m, n);
        matrixMultiply(AT, PB, ATPB, n, n, m);
        matrixMultiply(ATPB, SinvBTPA, corr, n, m, n);

        float rNorm = 0.0f, qNorm = 0.0f;
        for (int i = 0; i < n * n; i++) {
            float v = ATPA[i] - P[i] - corr[i] + Q[i];
            rNorm += v * v;
            qNorm += Q[i] * Q[i];
        }
        rNorm = sqrtf(rNorm);
        qNorm = sqrtf(qNorm);
        result = (qNorm > 1e-20f) ? (rNorm / qNorm) : rNorm;
    }

    delete[] AT; delete[] BT; delete[] PA; delete[] ATPA; delete[] PB; delete[] BTPB;
    delete[] BTPA; delete[] S; delete[] SinvBTPA; delete[] ATPB; delete[] corr;

    return result;
}

bool AutoLQR::computeGainMatrixSDA()
{
    // Implementação do Structure-preserving Doubling Algorithm (SDA)
    // Para DARE: A'·P·A - P - A'·P·B·(R + B'·P·B)^(-1)·B'·P·A + Q = 0
    
    if (!A || !B || !Q || !R || !K || !P)
        return false;

    if (!isSystemControllable()) {
        return false;
    }

    // Alocação de memória
    float* Ak = new float[stateSize * stateSize]();
    float* Gk = new float[stateSize * stateSize]();
    float* Hk = new float[stateSize * stateSize]();
    
    float* Ak_next = new float[stateSize * stateSize]();
    float* Gk_next = new float[stateSize * stateSize]();
    float* Hk_next = new float[stateSize * stateSize]();
    
    float* R_inv = new float[controlSize * controlSize]();
    float* BT = new float[controlSize * stateSize]();
    float* AT = new float[stateSize * stateSize]();
    float* W = new float[stateSize * stateSize]();
    float* Temp1 = new float[stateSize * stateSize]();
    float* Temp2 = new float[stateSize * stateSize]();
    float* Temp3 = new float[stateSize * stateSize]();
    
    // ========================================================================
    // INICIALIZAÇÃO CORRETA DO SDA PARA DARE
    // ========================================================================
    
    // 1. Ak = A (correto)
    matrixCopy(A, Ak, stateSize * stateSize);
    
    // 2. Calcular transpostas
    transposeMatrix(A, AT, stateSize, stateSize);
    transposeMatrix(B, BT, stateSize, controlSize);
    
    // 3. Calcular R_inv
    matrixCopy(R, R_inv, controlSize * controlSize);
    if (!invertMatrix(R_inv, R_inv, controlSize)) {
        delete[] Ak; delete[] Gk; delete[] Hk;
        delete[] Ak_next; delete[] Gk_next; delete[] Hk_next;
        delete[] R_inv; delete[] BT; delete[] AT;
        delete[] W; delete[] Temp1; delete[] Temp2; delete[] Temp3;
        return false;
    }
    
    // 4. Gk = B * R^(-1) * B' (correto)
    float* B_Rinv = new float[stateSize * controlSize];
    matrixMultiply(B, R_inv, B_Rinv, stateSize, controlSize, controlSize);
    matrixMultiply(B_Rinv, BT, Gk, stateSize, controlSize, stateSize);
    delete[] B_Rinv;

    // 5. CORREÇÃO CRÍTICA: Hk = Q (não A'·Q·A + Q)
    matrixCopy(Q, Hk, stateSize * stateSize);

    // ========================================================================
    // ESCALONAMENTO INICIAL (removido para DARE pois altera a solução)
    // ========================================================================
    // O escalonamento beta é aplicável em CARE (Equação de Riccati Contínua),
    // mas em DARE (Discreta) ele modifica a resposta final do horizonte infinito.
    // Portanto, nenhuma alteração em Gk (ou Hk) deve ser feita.

    // ========================================================================
    // LOOP SDA
    // ========================================================================
    bool converged = false;
    bool breakdown = false;
    float rel_diff = 1.0f; // sobrevive ao loop p/ lastStepDelta mesmo sem convergência
    bool bit_exact = false;

    // Inicializar histórico de resíduos
    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;

    for (int iter = 0; iter < maxIterations; iter++) {
        // W = (I + Gk·Hk)^(-1)
        matrixMultiply(Gk, Hk, Temp1, stateSize, stateSize, stateSize);

        for (int i = 0; i < stateSize; i++) {
            Temp1[i * stateSize + i] += 1.0f;
        }

        matrixCopy(Temp1, W, stateSize * stateSize);
        if (!invertMatrix(W, W, stateSize)) {
            breakdown = true;
            break;
        }

        // Temp1 = Ak·W
        matrixMultiply(Ak, W, Temp1, stateSize, stateSize, stateSize);

        // Ak_next = (Ak·W)·Ak
        matrixMultiply(Temp1, Ak, Ak_next, stateSize, stateSize, stateSize);

        // Gk_next = Gk + (Ak·W)·Gk·Ak'  (output do produto duplo é simétrico)
        transposeMatrix(Ak, AT, stateSize, stateSize);
        matrixMultiply(Gk, AT, Temp2, stateSize, stateSize, stateSize);
        matrixMultiplySymOutput(Temp1, Temp2, Temp3, stateSize);
        matrixAdd(Gk, Temp3, Gk_next, stateSize, stateSize);

        // Hk_next = Hk + Ak'·Hk·W·Ak  (output do produto duplo é simétrico)
        matrixMultiply(W, Ak, Temp2, stateSize, stateSize, stateSize);
        matrixMultiply(Hk, Temp2, Temp3, stateSize, stateSize, stateSize);
        matrixMultiplySymOutput(AT, Temp3, Temp2, stateSize);
        matrixAdd(Hk, Temp2, Hk_next, stateSize, stateSize);

        // Critério de parada: norma de Frobenius relativa (unificado com o
        // caminho fixed-point via cálculo em float — ver setStoppingCriterion()
        // e FixedPointQ.cpp — por pedido explícito do usuário; ver
        // docs/auditoria_solvers_riccati.md, Seção 13).
        float diff = 0.0f;
        float norm_Hk = 0.0f;
        bit_exact = true;
        for (int i = 0; i < stateSize * stateSize; i++) {
            float d = Hk_next[i] - Hk[i];
            if (Hk_next[i] != Hk[i]) bit_exact = false;
            diff += d * d;
            norm_Hk += Hk[i] * Hk[i];
        }
        diff = sqrtf(diff);
        norm_Hk = sqrtf(norm_Hk);

        rel_diff = (norm_Hk > 1e-10f) ? (diff / norm_Hk) : diff;

        // Armazenar resíduo no histórico (primeiras 10 iterações)
        if (iter < 10) {
            residualHistory[iter] = rel_diff;
            residualHistoryCount = iter + 1;
        }

        // Atualizar
        matrixCopy(Ak_next, Ak, stateSize * stateSize);
        matrixCopy(Gk_next, Gk, stateSize * stateSize);
        matrixCopy(Hk_next, Hk, stateSize * stateSize);

        if (rel_diff < relTolerance) {
            converged = true;
            lastIterations = iter + 1;
            lastStepDelta = rel_diff;
            lastStepIsBitExactZero = bit_exact;
            lastOutcome = SolveOutcome::Converged;
            break;
        }
    }

    if (!converged) {
        lastIterations = maxIterations;
        lastStepDelta = rel_diff; // último valor calculado dentro do loop (Hk já foi sobrescrito)
        lastStepIsBitExactZero = bit_exact;
        lastOutcome = breakdown ? SolveOutcome::Breakdown : SolveOutcome::Budget;
    }

    // P = Hk (solução final)
    matrixCopy(Hk, P, stateSize * stateSize);

    // ========================================================================
    // CÁLCULO DO GANHO K
    // ========================================================================
    // K = (R + B'·P·B)^(-1) · B'·P·A
    
    float* BT_P = new float[controlSize * stateSize];
    float* BT_P_B = new float[controlSize * controlSize];
    float* BT_P_A = new float[controlSize * stateSize];
    float* R_plus_BTPB = new float[controlSize * controlSize];

    // BT_P = B'·P
    matrixMultiply(BT, P, BT_P, controlSize, stateSize, stateSize);
    
    // BT_P_B = (B'·P)·B
    matrixMultiply(BT_P, B, BT_P_B, controlSize, stateSize, controlSize);
    
    // R_plus_BTPB = R + B'·P·B
    matrixAdd(R, BT_P_B, R_plus_BTPB, controlSize, controlSize);
    
    // Inverter
    if (!invertMatrix(R_plus_BTPB, R_plus_BTPB, controlSize)) {
        converged = false;
        lastOutcome = SolveOutcome::Breakdown; // (R+B'PB) singular no cálculo final de K
    } else {
        // BT_P_A = (B'·P)·A
        matrixMultiply(BT_P, A, BT_P_A, controlSize, stateSize, stateSize);
        
        // K = (R + B'·P·B)^(-1) · (B'·P·A)
        matrixMultiply(R_plus_BTPB, BT_P_A, K, controlSize, controlSize, stateSize);
    }

    residualDirty = true;

    // Limpeza
    delete[] Ak; delete[] Gk; delete[] Hk;
    delete[] Ak_next; delete[] Gk_next; delete[] Hk_next;
    delete[] R_inv; delete[] BT; delete[] AT;
    delete[] W; delete[] Temp1; delete[] Temp2; delete[] Temp3;
    delete[] BT_P; delete[] BT_P_B; delete[] BT_P_A; delete[] R_plus_BTPB;

    return converged;
}

bool AutoLQR::computeGainMatrixSchur()
{
    // Método de Schur para DARE usando formulação do pencil simplético
    // Resolve: A'PA - P - A'PB(R+B'PB)^{-1}B'PA + Q = 0
    
    if (!A || !B || !Q || !R || !K || !P)
        return false;

    if (!isSystemControllable()) {
        return false;
    }

    const int n = stateSize;
    const int m = controlSize;
    
    // ========================================================================
    // PASSO 1: Alocar memória
    // ========================================================================
    float* R_inv = new float[m * m];
    float* BT = new float[m * n];
    float* AT = new float[n * n];
    float* A_inv = new float[n * n];
    float* G = new float[n * n];
    float* temp_nm = new float[n * m];
    float* temp_nn = new float[n * n];
    float* temp_nn2 = new float[n * n];
    
    // ========================================================================
    // PASSO 2: Calcular matrizes auxiliares
    // ========================================================================
    // R_inv = inv(R)
    matrixCopy(R, R_inv, m * m);
    if (!invertMatrix(R_inv, R_inv, m)) {
        delete[] R_inv; delete[] BT; delete[] AT; delete[] A_inv;
        delete[] G; delete[] temp_nm; delete[] temp_nn; delete[] temp_nn2;
        return false;
    }
    
    // BT = B'
    transposeMatrix(B, BT, n, m);
    
    // AT = A'
    transposeMatrix(A, AT, n, n);
    
    // G = B * R^{-1} * B'
    matrixMultiply(B, R_inv, temp_nm, n, m, m);
    matrixMultiply(temp_nm, BT, G, n, m, n);
    
    // A_inv = inv(A)
    matrixCopy(A, A_inv, n * n);
    if (!invertMatrix(A_inv, A_inv, n)) {
        delete[] R_inv; delete[] BT; delete[] AT; delete[] A_inv;
        delete[] G; delete[] temp_nm; delete[] temp_nn; delete[] temp_nn2;
        return false;
    }
    
    // ========================================================================
    // PASSO 3: Construir matriz Hamiltoniana simplética Z (2n x 2n)
    // ========================================================================
    // Para DARE, usamos a matriz simplética:
    // Z = [A + G*A'^{-1}*Q,  -G*A'^{-1}    ]
    //     [-A'^{-1}*Q,        A'^{-1}      ]
    // Os autovalores estáveis de Z dão a solução
    
    Eigen::MatrixXf Z(2*n, 2*n);
    
    // Calcular A'^{-1}
    float* AT_inv = new float[n * n];
    matrixCopy(AT, AT_inv, n * n);
    if (!invertMatrix(AT_inv, AT_inv, n)) {
        delete[] R_inv; delete[] BT; delete[] AT; delete[] A_inv;
        delete[] G; delete[] temp_nm; delete[] temp_nn; delete[] temp_nn2;
        delete[] AT_inv;
        return false;
    }
    
    // Bloco (0,0): A + G*A'^{-1}*Q
    // temp_nn = A'^{-1} * Q
    matrixMultiply(AT_inv, Q, temp_nn, n, n, n);
    // temp_nn2 = G * (A'^{-1} * Q)
    matrixMultiply(G, temp_nn, temp_nn2, n, n, n);
    // Z(0:n, 0:n) = A + G*A'^{-1}*Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Z(i, j) = A[i * n + j] + temp_nn2[i * n + j];
        }
    }
    
    // Bloco (0,n): -G*A'^{-1}
    matrixMultiply(G, AT_inv, temp_nn, n, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Z(i, n + j) = -temp_nn[i * n + j];
        }
    }
    
    // Bloco (n,0): -A'^{-1}*Q
    matrixMultiply(AT_inv, Q, temp_nn, n, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Z(n + i, j) = -temp_nn[i * n + j];
        }
    }
    
    // Bloco (n,n): A'^{-1}
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Z(n + i, n + j) = AT_inv[i * n + j];
        }
    }
    
    // ========================================================================
    // PASSO 4: Decomposição de Schur complexa
    // ========================================================================
    Eigen::ComplexSchur<Eigen::MatrixXf> schur(Z);
    
    if (schur.info() != Eigen::Success) {
        delete[] R_inv; delete[] BT; delete[] AT; delete[] A_inv;
        delete[] G; delete[] temp_nm; delete[] temp_nn; delete[] temp_nn2;
        delete[] AT_inv;
        return false;
    }
    
    Eigen::MatrixXcf T = schur.matrixT();
    Eigen::MatrixXcf U = schur.matrixU();
    
    // ========================================================================
    // PASSO 5: Identificar autovalores estáveis (|λ| < 1)
    // ========================================================================
    std::vector<int> stable_indices;
    stable_indices.reserve(n);
    
    for (int i = 0; i < 2*n; i++) {
        std::complex<float> eigenvalue = T(i, i);
        float magnitude = std::abs(eigenvalue);
        
        if (magnitude < 1.0f && magnitude > 1e-10f) {
            stable_indices.push_back(i);
        }
    }
    
    // Se não encontrou n autovalores estáveis, tentar com threshold mais relaxado
    if (stable_indices.size() < static_cast<size_t>(n)) {
        stable_indices.clear();
        for (int i = 0; i < 2*n; i++) {
            std::complex<float> eigenvalue = T(i, i);
            float magnitude = std::abs(eigenvalue);
            
            if (magnitude < 1.0f + 1e-6f) {
                stable_indices.push_back(i);
                if (stable_indices.size() == static_cast<size_t>(n)) break;
            }
        }
    }
    
    if (stable_indices.size() != static_cast<size_t>(n)) {
        delete[] R_inv; delete[] BT; delete[] AT; delete[] A_inv;
        delete[] G; delete[] temp_nm; delete[] temp_nn; delete[] temp_nn2;
        delete[] AT_inv;
        return false;
    }
    
    // ========================================================================
    // PASSO 6: Extrair subespaço invariante estável
    // ========================================================================
    Eigen::MatrixXcf U11(n, n);
    Eigen::MatrixXcf U21(n, n);
    
    for (int j = 0; j < n; j++) {
        int idx = stable_indices[j];
        for (int i = 0; i < n; i++) {
            U11(i, j) = U(i, idx);
            U21(i, j) = U(n + i, idx);
        }
    }
    
    // ========================================================================
    // PASSO 7: Calcular P = U21 * inv(U11)
    // ========================================================================
    Eigen::MatrixXcf P_complex = U21 * U11.inverse();
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[i * n + j] = P_complex(i, j).real();
        }
    }
    
    // Forçar simetria de P
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float avg = (P[i * n + j] + P[j * n + i]) * 0.5f;
            P[i * n + j] = avg;
            P[j * n + i] = avg;
        }
    }
    
    // ========================================================================
    // PASSO 8: Calcular K = (R + B'*P*B)^{-1} * B'*P*A
    // ========================================================================
    float* BT_P = new float[m * n];
    float* BT_P_B = new float[m * m];
    float* BT_P_A = new float[m * n];
    float* term = new float[m * m];
    
    matrixMultiply(BT, P, BT_P, m, n, n);
    matrixMultiply(BT_P, B, BT_P_B, m, n, m);
    matrixAdd(R, BT_P_B, term, m, m);
    
    if (!invertMatrix(term, term, m)) {
        delete[] R_inv; delete[] BT; delete[] AT; delete[] A_inv;
        delete[] G; delete[] temp_nm; delete[] temp_nn; delete[] temp_nn2;
        delete[] AT_inv;
        delete[] BT_P; delete[] BT_P_B; delete[] BT_P_A; delete[] term;
        return false;
    }
    
    matrixMultiply(BT_P, A, BT_P_A, m, n, n);
    matrixMultiply(term, BT_P_A, K, m, m, n);

    // ========================================================================
    // Limpeza de memória
    // ========================================================================
    delete[] R_inv;
    delete[] BT;
    delete[] AT;
    delete[] A_inv;
    delete[] G;
    delete[] temp_nm;
    delete[] temp_nn;
    delete[] temp_nn2;
    delete[] AT_inv;
    delete[] BT_P;
    delete[] BT_P_B;
    delete[] BT_P_A;
    delete[] term;
    
    return true;
}

bool AutoLQR::computeGainMatrixVanDooren()
{
    // ========================================================================
    // Método de Van Dooren para DARE
    // Baseado em: P. van Dooren, "A Generalized Eigenvalue Approach For Solving
    // Riccati Equations", SIAM J. Sci. Stat. Comput., Vol.2(2), 1981.
    //
    // Usa o pencil simplétictico estendido (2n+m)×(2n+m) com deflação QR
    // para obter um pencil (2n)×(2n) antes da decomposição QZ.
    // ========================================================================
    
    // Van Dooren é método direto - inicializar histórico de resíduos com zeros
    residualHistoryCount = 1;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;
    
    if (!A || !B || !Q || !R || !K || !P)
        return false;

    if (!isSystemControllable()) {
        return false;
    }

    const int n = stateSize;
    const int m = controlSize;
    const int pencil_size = 2*n + m;

    // ========================================================================
    // Pré-alocar memória
    // ========================================================================
    float* BT = new float[m * n];
    float* AT = new float[n * n];
    float* BT_P = new float[m * n];
    float* BT_P_B = new float[m * m];
    float* term = new float[m * m];
    float* BT_P_A = new float[m * n];

    transposeMatrix(A, AT, n, n);
    transposeMatrix(B, BT, n, m);

    // ========================================================================
    // PASSO 1: Construir pencil estendido H - λJ  [(2n+m) × (2n+m)]
    // ========================================================================
    // H = [ A    0    B ]       J = [ I   0   0 ]
    //     [-Q    I    0 ]           [ 0  A^T  0 ]
    //     [ 0    0    R ]           [ 0 -B^T  0 ]
    //
    // Nota: Esta formulação é equivalente mas mais estável numericamente
    // ========================================================================
    
    Eigen::MatrixXf H(pencil_size, pencil_size);
    Eigen::MatrixXf J(pencil_size, pencil_size);
    
    H.setZero();
    J.setZero();

    // H[:n, :n] = A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H(i, j) = A[i * n + j];
        }
    }
    
    // H[:n, 2n:] = B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            H(i, 2*n + j) = B[i * m + j];
        }
    }
    
    // H[n:2n, :n] = -Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H(n + i, j) = -Q[i * n + j];
        }
    }
    
    // H[n:2n, n:2n] = I
    for (int i = 0; i < n; i++) {
        H(n + i, n + i) = 1.0f;
    }
    
    // H[2n:, 2n:] = R
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            H(2*n + i, 2*n + j) = R[i * m + j];
        }
    }
    
    // J[:n, :n] = I
    for (int i = 0; i < n; i++) {
        J(i, i) = 1.0f;
    }
    
    // J[n:2n, n:2n] = A^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            J(n + i, n + j) = AT[i * n + j];
        }
    }
    
    // J[2n:, n:2n] = -B^T
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            J(2*n + i, n + j) = -BT[i * n + j];
        }
    }

    // ========================================================================
    // PASSO 2: Deflação QR para reduzir de (2n+m)×(2n+m) para (2n)×(2n)
    // ========================================================================
    // Extrai as últimas m colunas de H e faz QR
    // Depois projeta no espaço ortogonal
    
    Eigen::MatrixXf H_last_cols = H.rightCols(m);
    
    // Decomposição QR: H_last_cols = Q_qr * R_qr
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(H_last_cols);
    Eigen::MatrixXf Q_qr = qr.householderQ();
    
    // Q_deflate = últimas (2n) colunas de Q_qr (descarta as primeiras m)
    Eigen::MatrixXf Q_deflate = Q_qr.rightCols(2*n);
    
    // Aplica deflação: projeta H e J nas primeiras 2n colunas
    Eigen::MatrixXf H_deflated = Q_deflate.transpose() * H.leftCols(2*n);
    Eigen::MatrixXf J_deflated = Q_deflate.transpose() * J.leftCols(2*n);

    // ========================================================================
    // PASSO 3: Decomposição QZ Generalizada no pencil deflacionado (2n × 2n)
    // ========================================================================
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> ges;
    ges.compute(H_deflated, J_deflated, true);
    
    if (ges.info() != Eigen::Success) {
        delete[] BT; delete[] AT;
        delete[] BT_P; delete[] BT_P_B; delete[] term; delete[] BT_P_A;
        return false;
    }
    
    Eigen::VectorXcf alpha = ges.alphas();
    Eigen::VectorXcf beta = ges.betas();
    const Eigen::MatrixXcf& Z = ges.eigenvectors();

    // ========================================================================
    // PASSO 4: Seleção de autovalores estáveis (|λ| < 1 para DARE)
    // ========================================================================
    // Armazena pares (magnitude, índice) para ordenar
    std::vector<std::pair<float, int>> eig_pairs;
    eig_pairs.reserve(2*n);
    
    const float beta_min = 1e-10f;
    
    for (int i = 0; i < 2*n; i++) {
        float beta_abs = std::abs(beta(i));
        float magnitude;
        
        if (beta_abs > beta_min) {
            std::complex<float> eigenvalue = alpha(i) / beta(i);
            magnitude = std::abs(eigenvalue);
        } else {
            // Autovalor no infinito - não é estável
            magnitude = 1e10f;
        }
        
        eig_pairs.push_back({magnitude, i});
    }
    
    // Ordena por magnitude (menor primeiro)
    std::sort(eig_pairs.begin(), eig_pairs.end());
    
    // Seleciona os n menores (dentro do círculo unitário)
    std::vector<int> stable_indices;
    stable_indices.reserve(n);
    
    for (int i = 0; i < n && i < static_cast<int>(eig_pairs.size()); i++) {
        if (eig_pairs[i].first < 1.0f) {
            stable_indices.push_back(eig_pairs[i].second);
        }
    }
    
    // Se não encontrou n autovalores estáveis, relaxa threshold
    if (stable_indices.size() < static_cast<size_t>(n)) {
        stable_indices.clear();
        for (int i = 0; i < n && i < static_cast<int>(eig_pairs.size()); i++) {
            if (eig_pairs[i].first < 1.1f) {
                stable_indices.push_back(eig_pairs[i].second);
            }
        }
    }
    
    if (stable_indices.size() != static_cast<size_t>(n)) {
        delete[] BT; delete[] AT;
        delete[] BT_P; delete[] BT_P_B; delete[] term; delete[] BT_P_A;
        return false;
    }

    // ========================================================================
    // PASSO 5: Extrair subespaço invariante estável
    // ========================================================================
    // Z está particionado como [U1; U2] onde cada bloco é n × n
    
    Eigen::MatrixXcf U1(n, n);
    Eigen::MatrixXcf U2(n, n);
    
    for (int j = 0; j < n; j++) {
        int idx = stable_indices[j];
        U1.col(j) = Z.col(idx).head(n);
        U2.col(j) = Z.col(idx).segment(n, n);
    }

    // ========================================================================
    // PASSO 6: Calcular P = U2 * inv(U1)
    // ========================================================================
    // Usa decomposição LU para maior estabilidade
    Eigen::FullPivLU<Eigen::MatrixXcf> lu(U1);
    
    if (!lu.isInvertible()) {
        delete[] BT; delete[] AT;
        delete[] BT_P; delete[] BT_P_B; delete[] term; delete[] BT_P_A;
        return false;
    }
    
    Eigen::MatrixXcf P_complex = U2 * lu.inverse();
    
    // Extrai parte real e copia para P
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[i * n + j] = P_complex(i, j).real();
        }
    }
    
    // Forçar simetria: P = (P + P') / 2
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float avg = (P[i * n + j] + P[j * n + i]) * 0.5f;
            P[i * n + j] = avg;
            P[j * n + i] = avg;
        }
    }

    // ========================================================================
    // PASSO 7: Calcular K = (R + B'PB)^(-1) * B'PA
    // ========================================================================
    matrixMultiply(BT, P, BT_P, m, n, n);
    matrixMultiply(BT_P, B, BT_P_B, m, n, m);
    matrixAdd(R, BT_P_B, term, m, m);
    
    if (!invertMatrix(term, term, m)) {
        delete[] BT; delete[] AT;
        delete[] BT_P; delete[] BT_P_B; delete[] term; delete[] BT_P_A;
        return false;
    }
    
    matrixMultiply(BT_P, A, BT_P_A, m, n, n);
    matrixMultiply(term, BT_P_A, K, m, m, n);

    // ========================================================================
    // Van Dooren é método direto (não iterativo)
    // ========================================================================
    lastIterations = 1;  // Método direto = 1 "iteração"
    lastResidual = 0.0f; // Solução direta, sem resíduo iterativo
    residualDirty = false; // valor final, não recalcular sob demanda

    // ========================================================================
    // Limpeza
    // ========================================================================
    delete[] BT; delete[] AT;
    delete[] BT_P; delete[] BT_P_B; delete[] term; delete[] BT_P_A;
    
    return true;
}

bool AutoLQR::computeGainMatrixKr()
{
    // Primeiro, certifique-se de que os ganhos K foram calculados
    if (!K) {
        return false;
    }

    for (int i = 0; i < controlSize; ++i) { 
        for (int j = 0; j < controlSize; ++j) { 
            if (j < stateSize) {
                Kr[i * controlSize + j] = -K[i * stateSize + j];
            } else {
                Kr[i * controlSize + j] = 0.0f;
            }
        }
    }
    return true; 
}

bool AutoLQR::exportKr(float* exportedKr)
{
    if (!Kr || !exportedKr)
        return false;
    
    matrixCopy(Kr, exportedKr, controlSize * controlSize);
    return true;
}

void AutoLQR::estimateFeedforwardGain(float* ffGain, const float* desiredState)
{
    if (!ffGain || !desiredState || !A || !B || !K)
        return;

    // For steady-state tracking: u_ff = -(inv(B'B) * B' * (A-I)) * x_desired
    // This is simplified for common cases

    if (stateSize == 2 && controlSize == 1) {
        // Special case for position-velocity systems
        float Bsq = B[0] * B[0] + B[1] * B[1];
        if (Bsq < 1e-6)
            return;

        float invBsq = 1.0f / Bsq;

        // Compute (A-I) * x_desired
        float dx[2];
        dx[0] = (A[0] - 1.0f) * desiredState[0] + A[1] * desiredState[1];
        dx[1] = A[2] * desiredState[0] + (A[3] - 1.0f) * desiredState[1];

        // Compute feedforward gain
        ffGain[0] = -invBsq * (B[0] * dx[0] + B[1] * dx[1]);
    } else {
        // For other systems, initialize to zero
        matrixClear(ffGain, controlSize);
    }
}

float AutoLQR::estimateConvergenceTime(float convergenceThreshold)
{
    // Estimate convergence time based on eigenvalues
    // This is a simplified estimate for 2x2 systems

    if (stateSize == 2) {
        // Compute closed-loop dynamics: A - B*K
        float CL[4];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                CL[i * 2 + j] = A[i * 2 + j];
                for (int k = 0; k < controlSize; k++) {
                    CL[i * 2 + j] -= B[i * controlSize + k] * K[k * stateSize + j];
                }
            }
        }

        // Approximate dominant eigenvalue using trace and determinant
        float trace = CL[0] + CL[3];
        float det = CL[0] * CL[3] - CL[1] * CL[2];

        // Characteristic equation: λ² - trace·λ + det = 0
        float discriminant = trace * trace - 4 * det;

        if (discriminant >= 0) {
            // Real eigenvalues
            float lambda1 = (trace + sqrt(discriminant)) / 2;
            float lambda2 = (trace - sqrt(discriminant)) / 2;

            // Dominant eigenvalue (larger magnitude)
            float domEigenvalue = (fabs(lambda1) > fabs(lambda2)) ? lambda1 : lambda2;

            if (fabs(domEigenvalue) < 1.0f && fabs(domEigenvalue) > 0.0f) {
                // Estimate time constant
                float timeConstant = -1.0f / log(fabs(domEigenvalue));

                // Time to reach convergenceThreshold
                return timeConstant * log(1.0f / convergenceThreshold);
            }
        }
    }

    // Default value if calculation fails
    return -1.0f;
}

bool AutoLQR::exportGains(float* exportedK)
{
    if (!K || !exportedK)
        return false;

    matrixCopy(K, exportedK, controlSize * stateSize);
    return true;
}

float AutoLQR::calculateExpectedCost()
{
    if (!P || !state)
        return -1.0f;

    // Cost = x'Px
    float cost = 0;
    for (int i = 0; i < stateSize; i++) {
        for (int j = 0; j < stateSize; j++) {
            cost += state[i] * P[i * stateSize + j] * state[j];
        }
    }

    return cost;
}

bool AutoLQR::computeGainMatrixIterative()
{
    // Método Iterativo de Riccati com Warm Start para DARE.
    // Warm-start usa P_warm (buffer dedicado), não o membro P genérico —
    // este último pode conter a solução deixada por outro método na última
    // chamada a computeGains(), o que faria o warm-start herdar um ponto de
    // partida de um problema diferente do resolvido por este método.

    if (!A || !B || !Q || !R || !K || !P || !P_warm)
        return false;

    const int n = stateSize;
    const int m = controlSize;
    const int nn = n * n;
    const int nm = n * m;
    const int mm = m * m;

    float Pw_norm = 0.0f;
    for (int i = 0; i < nn; i++) Pw_norm += fabsf(P_warm[i]);
    bool has_warm_start = (Pw_norm > 1e-6f);

    float* Pw = new float[nn]();
    if (has_warm_start) {
        matrixCopy(P_warm, Pw, nn);
    } else {
        matrixCopy(Q, Pw, nn);
    }

    float* P_new = new float[nn]();
    float* AT = new float[nn]();
    float* BT = new float[nm]();
    float* PA = new float[nn]();
    float* PB = new float[nm]();
    float* ATPA = new float[nn]();
    float* BTPB = new float[mm]();
    float* BTPA = new float[m * n]();
    float* S = new float[mm]();
    float* S_inv = new float[mm]();
    float* K_temp = new float[m * n]();
    float* ATPB = new float[nm]();
    float* correction = new float[nn]();

    // Calcular transpostas (constantes)
    transposeMatrix(A, AT, n, n);
    transposeMatrix(B, BT, n, m);

    bool converged = false;
    bool breakdown = false;
    float rel_diff = 1.0f; // sobrevive ao loop p/ lastStepDelta mesmo sem convergência
    bool bit_exact = false;

    // Inicializar histórico de resíduos
    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;

    for (int iter = 0; iter < maxIterations; iter++) {
        // ================================================================
        // ITERAÇÃO DARE
        // P_new = Q + A'PA - A'PB(R + B'PB)^{-1}B'PA
        // ================================================================

        matrixMultiply(Pw, A, PA, n, n, n);
        matrixMultiply(Pw, B, PB, n, n, m);
        matrixMultiply(AT, PA, ATPA, n, n, n);
        matrixMultiply(BT, PB, BTPB, m, n, m);
        matrixMultiply(BT, PA, BTPA, m, n, n);

        matrixAdd(R, BTPB, S, m, m);

        // Regularização
        for (int i = 0; i < m; i++) {
            S[i * m + i] += 1e-8f;
        }

        matrixCopy(S, S_inv, mm);
        if (!invertMatrix(S_inv, S_inv, m)) {
            breakdown = true;
            break;
        }

        matrixMultiply(S_inv, BTPA, K_temp, m, m, n);
        matrixMultiply(AT, PB, ATPB, n, n, m);
        matrixMultiply(ATPB, K_temp, correction, n, m, n);

        // Critério de parada: norma de Frobenius relativa — ver comentário
        // equivalente em computeGainMatrixSDA().
        float diff = 0.0f;
        float P_norm = 0.0f;
        bit_exact = true;

        for (int i = 0; i < nn; i++) {
            P_new[i] = Q[i] + ATPA[i] - correction[i];
            float d = P_new[i] - Pw[i];
            if (P_new[i] != Pw[i]) bit_exact = false;
            diff += d * d;
            P_norm += Pw[i] * Pw[i];
        }

        // Forçar simetria
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float avg = (P_new[i * n + j] + P_new[j * n + i]) * 0.5f;
                P_new[i * n + j] = avg;
                P_new[j * n + i] = avg;
            }
        }

        matrixCopy(P_new, Pw, nn);

        diff = sqrtf(diff);
        P_norm = sqrtf(P_norm);
        rel_diff = (P_norm > 1e-10f) ? (diff / P_norm) : diff;

        // Armazenar resíduo no histórico (primeiras 10 iterações)
        if (iter < 10) {
            residualHistory[iter] = rel_diff;
            residualHistoryCount = iter + 1;
        }

        if (rel_diff < relTolerance) {
            converged = true;
            lastIterations = iter + 1;
            lastStepDelta = rel_diff;
            lastStepIsBitExactZero = bit_exact;
            lastOutcome = SolveOutcome::Converged;
            break;
        }
    }

    if (!converged) {
        lastIterations = maxIterations;
        lastStepDelta = rel_diff; // último valor calculado dentro do loop
        lastStepIsBitExactZero = bit_exact;
        lastOutcome = breakdown ? SolveOutcome::Breakdown : SolveOutcome::Budget;
    }

    // ================================================================
    // CÁLCULO FINAL DO GANHO K
    // ================================================================
    matrixMultiply(Pw, A, PA, n, n, n);
    matrixMultiply(Pw, B, PB, n, n, m);
    matrixMultiply(BT, PB, BTPB, m, n, m);
    matrixMultiply(BT, PA, BTPA, m, n, n);

    matrixAdd(R, BTPB, S, m, m);
    for (int i = 0; i < m; i++) {
        S[i * m + i] += 1e-8f;
    }

    matrixCopy(S, S_inv, mm);
    if (invertMatrix(S_inv, S_inv, m)) {
        matrixMultiply(S_inv, BTPA, K, m, m, n);
    } else {
        converged = false;
        lastOutcome = SolveOutcome::Breakdown; // (R+B'PB) singular no cálculo final de K,
                                                // mesmo que o loop de convergência tenha OK'ado
    }

    // Publica o resultado no P genérico (getRicattiSolution) e no buffer de
    // warm-start dedicado (próxima chamada a este método, e só a este).
    matrixCopy(Pw, P, nn);
    matrixCopy(Pw, P_warm, nn);

    residualDirty = true;

    // Limpeza
    delete[] Pw;
    delete[] P_new;
    delete[] AT;
    delete[] BT;
    delete[] PA;
    delete[] PB;
    delete[] ATPA;
    delete[] BTPB;
    delete[] BTPA;
    delete[] S;
    delete[] S_inv;
    delete[] K_temp;
    delete[] ATPB;
    delete[] correction;

    return converged;
}

// ============================================================================
// SDA COM SINGLE SHIFT (SDA-ss)
// Versão melhorada do SDA com parâmetro de shift para convergência aprimorada
// quando autovalores estão próximos de 1
// ============================================================================
// ============================================================================
// SDA com single shift real, γ ∈ (0,1) — CORRIGIDO (ver plano, Fase 4.4).
//
// A versão anterior aplicava o shift só à matriz A, (A-γI)/(1-γ), sem shiftar
// H, o que resolve uma DARE DIFERENTE da original (erro relativo confirmado
// de ~10% a ~10^6× em outputs/verify_float64_mirror.csv). A construção abaixo
// desloca o PENCIL simplético inteiro (M,L) — M=[A,0;-H,I], L=[I,G;0,A^T] — via
// M'=M-γL, L'=L-γM, o que move os autovalores por λ↦(λ-γ)/(1-γλ) preservando
// os AUTOVETORES (e portanto o subespaço deflacionário e P). Reduzindo (M',L')
// de volta à forma SSF por eliminação em bloco (invertendo o pencil 2n×2n
// N1=[[I-γA,-γG],[γH,I-γA^T]]) obtém-se (Â,Ĝ,Ĥ) tal que o SDA padrão aplicado
// a eles converge para o MESMO P da DARE original — a correção algébrica em
// si foi verificada numericamente em outputs/verify_float64_mirror.csv
// (erro vs. scipy ~1e-14), mas ESSE arquivo não contém uma varredura de γ.
//
// CORRIGIDO 2026-08-18 (ver docs/auditoria_solvers_riccati.md, Seção 15): esta
// nota afirmava anteriormente uma varredura γ∈{0.1..0.99} com "~2x menos
// iterações" que este arquivo não continha — não existia medição nenhuma,
// γ=0.5 era só o ponto médio de (0,1).
//
// MEDIDO em test/gamma_sweep.cpp (Exp. 3), τ=1e-3/200 iters (critério
// casado), 4 trajetórias × 300 pts, γ∈{0.1,0.3,0.5,0.7,0.9}, 1216 pontos por
// γ, 100% convergência em todos:
//
//   γ    iters   SDA_SS(us)  resid_f      SDA_SS_FIXED(us)  resid_fx
//   0.1   9      11340.6     3.669e-06    4466.1             1.438e-02
//   0.3   9      11356.8     2.382e-06    4501.6             1.321e-02
//   0.5   8      10360.0     1.356e-06    4168.9              9.759e-03
//   0.7   7       9358.5     1.153e-06    3829.8              8.992e-03
//   0.9   5       7357.3     1.664e-06    3133.7              1.509e-02
//
// γ=0.7 DOMINA o antigo padrão γ=0.5 nas duas aritméticas — 12,5% mais
// rápido E resíduo melhor (float 15% menor, fixed 8% menor). γ=0.9 é ainda
// mais rápido, mas o resíduo fixed-point piora 68% vs. γ=0.7 (1.509e-2
// contra 8.992e-3) — mais iterações "economizadas" viram mais ruído de
// quantização acumulado por passo maior, não menos. Não há dominância aí:
// é troca de velocidade por acurácia, então γ=0.7 foi adotado como novo
// padrão em vez do extremo mais rápido. O paper que fundamenta a técnica de
// shift, Chu, Fan & Lin (2005), Linear Algebra Appl. 396:55-80, propõe uma
// busca de Fibonacci para o γ ótimo (minimiza o raio espectral dos
// autovalores transformados); essa busca fina NÃO foi implementada aqui —
// a grade grosseira de 5 pontos acima já basta para descartar o ponto médio
// não-justificado e substituí-lo por um valor medido. γ é exposto via
// setSDASSGamma() caso se queira refinar essa busca sem recompilar.
// ============================================================================
bool AutoLQR::computeGainMatrixSDA_SS()
{
    if (!A || !B || !Q || !R || !K || !P)
        return false;

    if (!isSystemControllable()) {
        return false;
    }

    const int n = stateSize;
    const int m = controlSize;
    const int nn = n * n;
    const int mm = m * m;
    const int n2 = 2 * n;

    const float gamma = ssGamma; // shift ajustável (setSDASSGamma), default 0.5 — ver Exp. 3
    bool converged = false;
    bool breakdown = false;
    float rel_diff = 1.0f;
    bool bit_exact = false;

    float* R_inv = new float[mm]();
    float* BT = new float[m * n]();
    float* G0 = new float[nn]();

    transposeMatrix(B, BT, n, m);
    matrixCopy(R, R_inv, mm);
    bool init_ok = invertMatrix(R_inv, R_inv, m);

    if (!init_ok) {
        delete[] R_inv; delete[] BT; delete[] G0;
        return false;
    }

    {
        float* B_Rinv = new float[n * m];
        matrixMultiply(B, R_inv, B_Rinv, n, m, m);
        matrixMultiply(B_Rinv, BT, G0, n, m, n);
        delete[] B_Rinv;
    }

    // ------------------------------------------------------------------
    // Monta N1 = [[I-γA, -γG0], [γH0, I-γA']] (2n×2n) e inverte para obter
    // Φ = N1^-1, cujos blocos dão a construção que preserva P.
    // ------------------------------------------------------------------
    float* N1  = new float[n2 * n2]();
    float* Phi = new float[n2 * n2]();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float aij = A[i * n + j];
            N1[i * n2 + j]             = (i == j ? 1.0f : 0.0f) - gamma * aij;       // I-γA
            N1[i * n2 + (n + j)]       = -gamma * G0[i * n + j];                     // -γG0
            N1[(n + i) * n2 + j]       = gamma * Q[i * n + j];                       // γH0 (H0=Q)
            N1[(n + i) * n2 + (n + j)] = (i == j ? 1.0f : 0.0f) - gamma * A[j * n + i]; // I-γA'
        }
    }

    if (!invertMatrix(N1, Phi, n2)) {
        delete[] R_inv; delete[] BT; delete[] G0; delete[] N1; delete[] Phi;
        return false;
    }

    float* Phi11 = new float[nn]();
    float* Phi12 = new float[nn]();
    float* Phi21 = new float[nn]();
    float* Phi22 = new float[nn]();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Phi11[i * n + j] = Phi[i * n2 + j];
            Phi12[i * n + j] = Phi[i * n2 + (n + j)];
            Phi21[i * n + j] = Phi[(n + i) * n2 + j];
            Phi22[i * n + j] = Phi[(n + i) * n2 + (n + j)];
        }
    }
    delete[] N1; delete[] Phi;

    // AmG = A-γI, ATmG = A'-γI
    float* AmG  = new float[nn]();
    float* ATmG = new float[nn]();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float aij = A[i * n + j];
            AmG[i * n + j]  = aij - (i == j ? gamma : 0.0f);
            ATmG[i * n + j] = A[j * n + i] - (i == j ? gamma : 0.0f);
        }
    }

    // Â = Φ11·(A-γI) - Φ12·H0 ;  Ĝ = Φ11·G0 + Φ12·(A'-γI) ;  Ĥ = -Φ21·(A-γI) + Φ22·H0
    float* Ak = new float[nn]();
    float* Gk = new float[nn]();
    float* Hk = new float[nn]();
    float* Temp1 = new float[nn]();
    float* Temp2 = new float[nn]();

    matrixMultiply(Phi11, AmG, Temp1, n, n, n);
    matrixMultiply(Phi12, Q, Temp2, n, n, n);
    matrixSubtract(Temp1, Temp2, Ak, n, n);

    matrixMultiply(Phi11, G0, Temp1, n, n, n);
    matrixMultiply(Phi12, ATmG, Temp2, n, n, n);
    matrixAdd(Temp1, Temp2, Gk, n, n);

    matrixMultiply(Phi21, AmG, Temp1, n, n, n);
    matrixMultiply(Phi22, Q, Temp2, n, n, n);
    matrixSubtract(Temp2, Temp1, Hk, n, n);

    delete[] Phi11; delete[] Phi12; delete[] Phi21; delete[] Phi22;
    delete[] AmG; delete[] ATmG; delete[] G0;

    // ------------------------------------------------------------------
    // Loop SDA padrão sobre (Ak,Gk,Hk) — o P resultante já é o da DARE
    // original (ver comentário acima); K usa o A original.
    // ------------------------------------------------------------------
    float* Ak_next = new float[nn]();
    float* Gk_next = new float[nn]();
    float* Hk_next = new float[nn]();
    float* AT = new float[nn]();
    float* W  = new float[nn]();
    float* Temp3 = new float[nn]();

    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;

    for (int iter = 0; iter < maxIterations; iter++) {
        matrixMultiply(Gk, Hk, Temp1, n, n, n);
        for (int i = 0; i < n; i++) Temp1[i * n + i] += 1.0f;

        matrixCopy(Temp1, W, nn);
        if (!invertMatrix(W, W, n)) { breakdown = true; break; }

        matrixMultiply(Ak, W, Temp1, n, n, n);
        matrixMultiply(Temp1, Ak, Ak_next, n, n, n);

        transposeMatrix(Ak, AT, n, n);
        matrixMultiply(Gk, AT, Temp2, n, n, n);
        matrixMultiply(Temp1, Temp2, Temp3, n, n, n);
        matrixAdd(Gk, Temp3, Gk_next, n, n);

        matrixMultiply(W, Ak, Temp2, n, n, n);
        matrixMultiply(Hk, Temp2, Temp3, n, n, n);
        matrixMultiply(AT, Temp3, Temp2, n, n, n);
        matrixAdd(Hk, Temp2, Hk_next, n, n);

        // Critério de parada: norma de Frobenius relativa — ver
        // comentário equivalente em computeGainMatrixSDA().
        float diff = 0.0f, norm_Hk = 0.0f;
        bit_exact = true;
        for (int i = 0; i < nn; i++) {
            float d = Hk_next[i] - Hk[i];
            if (Hk_next[i] != Hk[i]) bit_exact = false;
            diff += d * d;
            norm_Hk += Hk[i] * Hk[i];
        }
        diff = sqrtf(diff);
        norm_Hk = sqrtf(norm_Hk);
        rel_diff = (norm_Hk > 1e-10f) ? (diff / norm_Hk) : diff;

        if (iter < 10) {
            residualHistory[iter] = rel_diff;
            residualHistoryCount = iter + 1;
        }

        matrixCopy(Ak_next, Ak, nn);
        matrixCopy(Gk_next, Gk, nn);
        matrixCopy(Hk_next, Hk, nn);

        if (rel_diff < relTolerance) {
            converged = true;
            lastIterations = iter + 1;
            lastStepDelta = rel_diff;
            lastStepIsBitExactZero = bit_exact;
            lastOutcome = SolveOutcome::Converged;
            break;
        }
    }

    if (!converged) {
        lastIterations = maxIterations;
        lastStepDelta = rel_diff;
        lastStepIsBitExactZero = bit_exact;
        lastOutcome = breakdown ? SolveOutcome::Breakdown : SolveOutcome::Budget;
    }

    matrixCopy(Hk, P, nn);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float avg = (P[i * n + j] + P[j * n + i]) * 0.5f;
            P[i * n + j] = avg;
            P[j * n + i] = avg;
        }
    }

    // K = (R + B'PB)^-1 B'PA, com A original
    {
        float* BT_P = new float[m * n];
        float* BT_P_B = new float[mm];
        float* BT_P_A = new float[m * n];
        float* R_plus_BTPB = new float[mm];

        matrixMultiply(BT, P, BT_P, m, n, n);
        matrixMultiply(BT_P, B, BT_P_B, m, n, m);
        matrixAdd(R, BT_P_B, R_plus_BTPB, m, m);

        if (!invertMatrix(R_plus_BTPB, R_plus_BTPB, m)) {
            converged = false;
            lastOutcome = SolveOutcome::Breakdown; // (R+B'PB) singular no cálculo final de K
        } else {
            matrixMultiply(BT_P, A, BT_P_A, m, n, n);
            matrixMultiply(R_plus_BTPB, BT_P_A, K, m, m, n);
        }

        delete[] BT_P; delete[] BT_P_B; delete[] BT_P_A; delete[] R_plus_BTPB;
    }

    residualDirty = true;

    delete[] R_inv; delete[] BT;
    delete[] Ak; delete[] Gk; delete[] Hk;
    delete[] Ak_next; delete[] Gk_next; delete[] Hk_next;
    delete[] AT; delete[] W; delete[] Temp1; delete[] Temp2; delete[] Temp3;

    return converged;
}

// ============================================================================
// SDA ADAPTATIVO (ASDA)
// Usa escalonamento adaptativo durante iterações para melhor estabilidade
// ============================================================================
// ============================================================================
// ASDA (SDA com escalonamento adaptativo) — CORRIGIDO (ver plano, Fase 4.2).
//
// A recorrência do SDA é invariante sob (G,H) -> (s·G, H/s): o produto G·H
// (logo W) não muda, então H_k converge para P/∏s_i, não para P. A versão
// anterior fazia P = H_k direto, sem desfazer o produto acumulado dos s_i, e
// aplicava β² só em G na inicialização (sem dividir H) — dois bugs, cada um
// verificado numericamente (outputs/verify_float64_mirror.csv: resíduo de até
// 2e7 em float64, ou seja, bug de fórmula e não de precisão). A correção
// aplica o mesmo fator ao par (G,H) sempre, acumula cum_s = ∏s_i, e devolve
// P = H_k · cum_s.
// ============================================================================
bool AutoLQR::computeGainMatrixASDA()
{
    if (!A || !B || !Q || !R || !K || !P)
        return false;

    if (!isSystemControllable()) {
        return false;
    }

    const int n = stateSize;
    const int m = controlSize;
    const int nn = n * n;
    const int mm = m * m;
    bool converged = false;
    bool breakdown = false;
    float rel_diff = 1.0f;
    bool bit_exact = false;
    float cum_s = 1.0f; // produto acumulado dos fatores de escala (para desfazer em P)

    // Alocação de memória
    float* Ak = new float[nn]();
    float* Gk = new float[nn]();
    float* Hk = new float[nn]();

    float* Ak_next = new float[nn]();
    float* Gk_next = new float[nn]();
    float* Hk_next = new float[nn]();

    float* R_inv = new float[mm]();
    float* BT = new float[m * n]();
    float* AT = new float[nn]();
    float* W = new float[nn]();
    float* Temp1 = new float[nn]();
    float* Temp2 = new float[nn]();
    float* Temp3 = new float[nn]();

    // Inicialização ASDA
    matrixCopy(A, Ak, nn);
    transposeMatrix(A, AT, n, n);
    transposeMatrix(B, BT, n, m);

    matrixCopy(R, R_inv, mm);
    bool init_ok = invertMatrix(R_inv, R_inv, m);

    if (init_ok) {
        float* B_Rinv = new float[n * m];
        matrixMultiply(B, R_inv, B_Rinv, n, m, m);
        matrixMultiply(B_Rinv, BT, Gk, n, m, n);
        delete[] B_Rinv;

        matrixCopy(Q, Hk, nn);

        // Escalonamento ótimo inicial: G←s0·G, H←H/s0 (eq. correta — aplicado
        // aos DOIS, não só a G), com s0 = sqrt(‖H‖/‖G‖).
        float norm_G = 0.0f, norm_H = 0.0f;
        for (int i = 0; i < nn; i++) {
            norm_G += Gk[i] * Gk[i];
            norm_H += Hk[i] * Hk[i];
        }
        norm_G = sqrtf(norm_G);
        norm_H = sqrtf(norm_H);

        float s0 = 1.0f;
        if (norm_H > 1e-10f && norm_G > 1e-10f) {
            s0 = sqrtf(norm_H / norm_G);
            s0 = fminf(fmaxf(s0, 0.1f), 10.0f);
        }
        for (int i = 0; i < nn; i++) {
            Gk[i] *= s0;
            Hk[i] /= s0;
        }
        cum_s *= s0;

        // Inicializar histórico de resíduos
        residualHistoryCount = 0;
        for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;

        // Loop ASDA
        for (int iter = 0; iter < maxIterations; iter++) {
            // Passo 1: escalonamento adaptativo
            float norm_Gk = 0.0f, norm_Hk = 0.0f;
            for (int i = 0; i < nn; i++) {
                norm_Gk += Gk[i] * Gk[i];
                norm_Hk += Hk[i] * Hk[i];
            }
            norm_Gk = sqrtf(norm_Gk);
            norm_Hk = sqrtf(norm_Hk);

            float scale_factor = 1.0f;
            if (norm_Gk > 1e-10f && norm_Hk > 1e-10f) {
                scale_factor = sqrtf(norm_Hk / norm_Gk);
                scale_factor = fminf(fmaxf(scale_factor, 0.1f), 10.0f);
            }

            for (int i = 0; i < nn; i++) {
                Gk[i] *= scale_factor;
                Hk[i] /= scale_factor;
            }
            cum_s *= scale_factor;

            // Passo 2: Iteração SDA padrão
            matrixMultiply(Gk, Hk, Temp1, n, n, n);
            for (int i = 0; i < n; i++) {
                Temp1[i * n + i] += 1.0f;
            }

            matrixCopy(Temp1, W, nn);
            if (!invertMatrix(W, W, n)) {
                breakdown = true;
                break;
            }

            matrixMultiply(Ak, W, Temp1, n, n, n);
            matrixMultiply(Temp1, Ak, Ak_next, n, n, n);

            transposeMatrix(Ak, AT, n, n);
            matrixMultiply(Gk, AT, Temp2, n, n, n);
            matrixMultiply(Temp1, Temp2, Temp3, n, n, n);
            matrixAdd(Gk, Temp3, Gk_next, n, n);

            matrixMultiply(W, Ak, Temp2, n, n, n);
            matrixMultiply(Hk, Temp2, Temp3, n, n, n);
            matrixMultiply(AT, Temp3, Temp2, n, n, n);
            matrixAdd(Hk, Temp2, Hk_next, n, n);

            // Critério de parada: norma de Frobenius relativa — ver
            // comentário equivalente em computeGainMatrixSDA(). Nomes
            // diff_conv/norm_Hk_conv (não diff/norm_Hk) para não colidir
            // com norm_Gk/norm_Hk do reescalonamento adaptativo (ASDA),
            // mesmo escopo do laço.
            float diff_conv = 0.0f, norm_Hk_conv = 0.0f;
            bit_exact = true;
            for (int i = 0; i < nn; i++) {
                float d = Hk_next[i] - Hk[i];
                if (Hk_next[i] != Hk[i]) bit_exact = false;
                diff_conv += d * d;
                norm_Hk_conv += Hk[i] * Hk[i];
            }
            diff_conv = sqrtf(diff_conv);
            norm_Hk_conv = sqrtf(norm_Hk_conv);
            rel_diff = (norm_Hk_conv > 1e-10f) ? (diff_conv / norm_Hk_conv) : diff_conv;

            // Armazenar resíduo no histórico (primeiras 10 iterações)
            if (iter < 10) {
                residualHistory[iter] = rel_diff;
                residualHistoryCount = iter + 1;
            }

            matrixCopy(Ak_next, Ak, nn);
            matrixCopy(Gk_next, Gk, nn);
            matrixCopy(Hk_next, Hk, nn);

            if (rel_diff < relTolerance) {
                converged = true;
                lastIterations = iter + 1;
                lastStepDelta = rel_diff;
                lastStepIsBitExactZero = bit_exact;
                lastOutcome = SolveOutcome::Converged;
                break;
            }
        }

        if (!converged) {
            lastIterations = maxIterations;
            lastStepDelta = rel_diff;
            lastStepIsBitExactZero = bit_exact;
            lastOutcome = breakdown ? SolveOutcome::Breakdown : SolveOutcome::Budget;
        }

        // P = Hk · cum_s (desfaz o escalonamento acumulado — ver comentário acima)
        for (int i = 0; i < nn; i++) {
            P[i] = Hk[i] * cum_s;
        }

        // Forçar simetria
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float avg = (P[i * n + j] + P[j * n + i]) * 0.5f;
                P[i * n + j] = avg;
                P[j * n + i] = avg;
            }
        }

        // Cálculo do ganho K
        float* BT_P = new float[m * n];
        float* BT_P_B = new float[mm];
        float* BT_P_A = new float[m * n];
        float* R_plus_BTPB = new float[mm];

        matrixMultiply(BT, P, BT_P, m, n, n);
        matrixMultiply(BT_P, B, BT_P_B, m, n, m);
        matrixAdd(R, BT_P_B, R_plus_BTPB, m, m);

        if (!invertMatrix(R_plus_BTPB, R_plus_BTPB, m)) {
            converged = false;
            lastOutcome = SolveOutcome::Breakdown; // (R+B'PB) singular no cálculo final de K
        } else {
            matrixMultiply(BT_P, A, BT_P_A, m, n, n);
            matrixMultiply(R_plus_BTPB, BT_P_A, K, m, m, n);
        }

        delete[] BT_P;
        delete[] BT_P_B;
        delete[] BT_P_A;
        delete[] R_plus_BTPB;

        residualDirty = true;
    }

    delete[] Ak; delete[] Gk; delete[] Hk;
    delete[] Ak_next; delete[] Gk_next; delete[] Hk_next;
    delete[] R_inv; delete[] BT; delete[] AT;
    delete[] W; delete[] Temp1; delete[] Temp2; delete[] Temp3;

    return converged;
}

// ============================================================================
// SDA COM ESCALONAMENTO ÓTIMO (Scaled SDA)
// Usa escalonamento ótimo do pencil Hamiltoniano para melhor condicionamento
// ============================================================================
// ============================================================================
// SDA com balanceamento diagonal do pencil — CORRIGIDO (ver plano, Fase 4.3).
//
// Para a similaridade Â=DAD⁻¹, Ĝ=DGD (já corretas antes), a convenção
// consistente exige Ĥ=D⁻¹QD⁻¹ e, na volta, P=D·P̂·D. A versão anterior usava
// Ĥ=DQD e P=D⁻¹P̂D⁻¹ — os DOIS expoentes de D invertidos, e os erros não se
// cancelam (verificado: resíduo de até 1e6 em float64, outputs/
// verify_float64_mirror.csv). Fórmula pelo balanceamento de Ward (1981),
// SIAM JSSC 2(2):141-152.
// ============================================================================
bool AutoLQR::computeGainMatrixSDA_Scaled()
{
    if (!A || !B || !Q || !R || !K || !P)
        return false;

    if (!isSystemControllable()) {
        return false;
    }

    const int n = stateSize;
    const int m = controlSize;
    const int nn = n * n;
    const int mm = m * m;
    bool converged = false;
    bool breakdown = false;
    float rel_diff = 1.0f;
    bool bit_exact = false;

    // Alocação de memória
    float* Ak = new float[nn]();
    float* Gk = new float[nn]();
    float* Hk = new float[nn]();

    float* Ak_next = new float[nn]();
    float* Gk_next = new float[nn]();
    float* Hk_next = new float[nn]();

    float* R_inv = new float[mm]();
    float* BT = new float[m * n]();
    float* AT = new float[nn]();
    float* W = new float[nn]();
    float* Temp1 = new float[nn]();
    float* Temp2 = new float[nn]();
    float* Temp3 = new float[nn]();

    // Matrizes de escalonamento
    float* D = new float[n]();
    float* Dinv = new float[n]();
    
    // Calcular escalonamento diagonal baseado nas normas das linhas de A
    for (int i = 0; i < n; i++) {
        float row_norm = 0.0f;
        for (int j = 0; j < n; j++) {
            row_norm += A[i * n + j] * A[i * n + j];
        }
        row_norm = sqrtf(row_norm);
        D[i] = (row_norm > 1e-10f) ? (1.0f / sqrtf(row_norm)) : 1.0f;
        Dinv[i] = 1.0f / D[i];
    }
    
    // Aplicar escalonamento a A: A_scaled = D * A * D^(-1)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ak[i * n + j] = D[i] * A[i * n + j] * Dinv[j];
        }
    }
    
    transposeMatrix(Ak, AT, n, n);
    transposeMatrix(B, BT, n, m);
    
    // Calcular R_inv
    matrixCopy(R, R_inv, mm);
    bool init_ok = invertMatrix(R_inv, R_inv, m);
    
    if (init_ok) {
        // Calcular G com escalonamento
        float* B_scaled = new float[n * m];
        float* B_Rinv = new float[n * m];
        float* BT_scaled = new float[m * n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                B_scaled[i * m + j] = D[i] * B[i * m + j];
            }
        }
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                BT_scaled[i * n + j] = BT[i * n + j] * D[j];
            }
        }
        
        matrixMultiply(B_scaled, R_inv, B_Rinv, n, m, m);
        matrixMultiply(B_Rinv, BT_scaled, Gk, n, m, n);
        
        delete[] B_scaled;
        delete[] B_Rinv;
        delete[] BT_scaled;
        
        // H0 = D^-1 * Q * D^-1 (convenção correta: Â=DAD⁻¹,Ĝ=DGD,Ĥ=D⁻¹QD⁻¹)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Hk[i * n + j] = Dinv[i] * Q[i * n + j] * Dinv[j];
            }
        }
        
        // Inicializar histórico de resíduos
        residualHistoryCount = 0;
        for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;

        // Loop SDA escalonado
        for (int iter = 0; iter < maxIterations; iter++) {
            matrixMultiply(Gk, Hk, Temp1, n, n, n);
            for (int i = 0; i < n; i++) {
                Temp1[i * n + i] += 1.0f;
            }
            
            matrixCopy(Temp1, W, nn);
            if (!invertMatrix(W, W, n)) {
                breakdown = true;
                break;
            }

            matrixMultiply(Ak, W, Temp1, n, n, n);
            matrixMultiply(Temp1, Ak, Ak_next, n, n, n);

            transposeMatrix(Ak, AT, n, n);
            matrixMultiply(Gk, AT, Temp2, n, n, n);
            matrixMultiply(Temp1, Temp2, Temp3, n, n, n);
            matrixAdd(Gk, Temp3, Gk_next, n, n);

            matrixMultiply(W, Ak, Temp2, n, n, n);
            matrixMultiply(Hk, Temp2, Temp3, n, n, n);
            matrixMultiply(AT, Temp3, Temp2, n, n, n);
            matrixAdd(Hk, Temp2, Hk_next, n, n);

            // Critério de parada: norma de Frobenius relativa — ver
            // comentário equivalente em computeGainMatrixSDA(). Nomes
            // diff_conv/norm_Hk_conv (não diff/norm_Hk) para não colidir
            // com norm_Gk/norm_Hk do reescalonamento adaptativo (ASDA),
            // mesmo escopo do laço.
            float diff_conv = 0.0f, norm_Hk_conv = 0.0f;
            bit_exact = true;
            for (int i = 0; i < nn; i++) {
                float d = Hk_next[i] - Hk[i];
                if (Hk_next[i] != Hk[i]) bit_exact = false;
                diff_conv += d * d;
                norm_Hk_conv += Hk[i] * Hk[i];
            }
            diff_conv = sqrtf(diff_conv);
            norm_Hk_conv = sqrtf(norm_Hk_conv);
            rel_diff = (norm_Hk_conv > 1e-10f) ? (diff_conv / norm_Hk_conv) : diff_conv;

            // Armazenar resíduo no histórico (primeiras 10 iterações)
            if (iter < 10) {
                residualHistory[iter] = rel_diff;
                residualHistoryCount = iter + 1;
            }

            matrixCopy(Ak_next, Ak, nn);
            matrixCopy(Gk_next, Gk, nn);
            matrixCopy(Hk_next, Hk, nn);

            if (rel_diff < relTolerance) {
                converged = true;
                lastIterations = iter + 1;
                lastStepDelta = rel_diff;
                lastStepIsBitExactZero = bit_exact;
                lastOutcome = SolveOutcome::Converged;
                break;
            }
        }

        if (!converged) {
            lastIterations = maxIterations;
            lastStepDelta = rel_diff;
            lastStepIsBitExactZero = bit_exact;
            lastOutcome = breakdown ? SolveOutcome::Breakdown : SolveOutcome::Budget;
        }

        // Recuperar P original: P = D * P_scaled * D
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                P[i * n + j] = D[i] * Hk[i * n + j] * D[j];
            }
        }
        
        // Forçar simetria
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float avg = (P[i * n + j] + P[j * n + i]) * 0.5f;
                P[i * n + j] = avg;
                P[j * n + i] = avg;
            }
        }

        // Cálculo do ganho K
        float* BT_P = new float[m * n];
        float* BT_P_B = new float[mm];
        float* BT_P_A = new float[m * n];
        float* R_plus_BTPB = new float[mm];
        
        matrixMultiply(BT, P, BT_P, m, n, n);
        matrixMultiply(BT_P, B, BT_P_B, m, n, m);
        matrixAdd(R, BT_P_B, R_plus_BTPB, m, m);
        
        if (!invertMatrix(R_plus_BTPB, R_plus_BTPB, m)) {
            converged = false;
            lastOutcome = SolveOutcome::Breakdown; // (R+B'PB) singular no cálculo final de K
        } else {
            matrixMultiply(BT_P, A, BT_P_A, m, n, n);
            matrixMultiply(R_plus_BTPB, BT_P_A, K, m, m, n);
        }

        delete[] BT_P;
        delete[] BT_P_B;
        delete[] BT_P_A;
        delete[] R_plus_BTPB;

        residualDirty = true;
    }

    delete[] Ak; delete[] Gk; delete[] Hk;
    delete[] Ak_next; delete[] Gk_next; delete[] Hk_next;
    delete[] R_inv; delete[] BT; delete[] AT;
    delete[] W; delete[] Temp1; delete[] Temp2; delete[] Temp3;
    delete[] D; delete[] Dinv;

    return converged;
}

// ============================================================================
// ADDA — CORRIGIDO (ver plano, Fase 4.5).
//
// A forma anterior calculava DUAS inversas por iteração, V=(I+GH)^-1 e
// W=(I+HG)^-1, usando W só na atualização de H. Pela identidade push-through
// (I+HG)^-1·H ≡ H·(I+GH)^-1, ou seja W·H ≡ H·V, a atualização
// Hk+1 = Hk + Ak'·W·Hk·Ak é ALGEBRICAMENTE IDÊNTICA a Hk + Ak'·(Hk·V)·Ak — a
// mesma recorrência do SDA (Chu, Fan, Lin & Wang 2004), só com uma inversão
// n×n redundante por iteração. Verificado numericamente: ‖P_ADDA-P_SDA‖/‖P_SDA‖
// ~1e-16 em float64 para todos os casos de outputs/verify_float64_mirror.csv.
//
// A citação anterior (Lin & Xu, 2006/2007 — inconsistente entre .h e .cpp)
// não corresponde a este algoritmo: o ADDA de fato (Wang, Wang & Li, SIAM J.
// Matrix Anal. Appl. 33(1):170-194, 2012) resolve a MARE não-simétrica de
// M-matriz com DOIS parâmetros de shift α≠β — não implementado aqui; portar
// essa construção para a DARE simétrica exige derivar a Cayley generalizada
// sobre o pencil simplético, o que é trabalho de pesquisa e fica registrado
// como próximo passo em docs/auditoria_solvers_riccati.md.
//
// Esta versão remove a segunda inversão (usa a identidade push-through) e
// mantém o nome/dispatcher por compatibilidade, documentando que o resultado
// é o SDA em forma alternada.
// ============================================================================
bool AutoLQR::computeGainMatrixADDA()
{
    if (!A || !B || !Q || !R || !K || !P)
        return false;

    if (!isSystemControllable()) {
        return false;
    }

    const int n = stateSize;
    const int m = controlSize;
    const int nn = n * n;
    const int mm = m * m;

    // Alocação de memória
    float* Ak = new float[nn]();
    float* Gk = new float[nn]();
    float* Hk = new float[nn]();

    float* Ak_next = new float[nn]();
    float* Gk_next = new float[nn]();
    float* Hk_next = new float[nn]();

    float* R_inv = new float[mm]();
    float* BT = new float[m * n]();
    float* AT = new float[nn]();
    float* V = new float[nn]();      // (I + Gk·Hk)^(-1)
    float* Temp1 = new float[nn]();
    float* Temp2 = new float[nn]();
    float* Temp3 = new float[nn]();

    // 1. Ak = A
    matrixCopy(A, Ak, nn);

    // 2. Calcular transpostas
    transposeMatrix(A, AT, n, n);
    transposeMatrix(B, BT, n, m);

    // 3. Calcular R_inv
    matrixCopy(R, R_inv, mm);
    if (!invertMatrix(R_inv, R_inv, m)) {
        delete[] Ak; delete[] Gk; delete[] Hk;
        delete[] Ak_next; delete[] Gk_next; delete[] Hk_next;
        delete[] R_inv; delete[] BT; delete[] AT;
        delete[] V; delete[] Temp1; delete[] Temp2; delete[] Temp3;
        return false;
    }

    // 4. Gk = B * R^(-1) * B'
    float* B_Rinv = new float[n * m];
    matrixMultiply(B, R_inv, B_Rinv, n, m, m);
    matrixMultiply(B_Rinv, BT, Gk, n, m, n);
    delete[] B_Rinv;

    // 5. Hk = Q
    matrixCopy(Q, Hk, nn);

    bool converged = false;
    bool breakdown = false;
    float rel_diff = 1.0f;
    bool bit_exact = false;

    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;

    for (int iter = 0; iter < maxIterations; iter++) {
        // V = (I + Gk·Hk)^(-1)
        matrixMultiply(Gk, Hk, Temp1, n, n, n);
        for (int i = 0; i < n; i++) {
            Temp1[i * n + i] += 1.0f;
        }
        matrixCopy(Temp1, V, nn);
        if (!invertMatrix(V, V, n)) {
            breakdown = true;
            break;
        }

        // Ak_next = Ak·V·Ak
        matrixMultiply(Ak, V, Temp1, n, n, n);
        matrixMultiply(Temp1, Ak, Ak_next, n, n, n);

        // Gk_next = Gk + Ak·V·Gk·Ak'
        transposeMatrix(Ak, AT, n, n);
        matrixMultiply(Gk, AT, Temp2, n, n, n);
        matrixMultiply(Temp1, Temp2, Temp3, n, n, n);  // Temp1 ainda é Ak·V
        matrixAdd(Gk, Temp3, Gk_next, n, n);

        // Hk_next = Hk + Ak'·(Hk·V)·Ak  (push-through: (I+HG)^-1 H ≡ H(I+GH)^-1,
        // evita computar W=(I+Hk·Gk)^-1 separadamente)
        matrixMultiply(Hk, V, Temp2, n, n, n);
        matrixMultiply(Temp2, Ak, Temp3, n, n, n);
        matrixMultiply(AT, Temp3, Temp2, n, n, n);
        matrixAdd(Hk, Temp2, Hk_next, n, n);

        // Critério de parada: norma de Frobenius relativa — ver
        // comentário equivalente em computeGainMatrixSDA().
        float diff = 0.0f, norm_Hk = 0.0f;
        bit_exact = true;
        for (int i = 0; i < nn; i++) {
            float d = Hk_next[i] - Hk[i];
            if (Hk_next[i] != Hk[i]) bit_exact = false;
            diff += d * d;
            norm_Hk += Hk[i] * Hk[i];
        }
        diff = sqrtf(diff);
        norm_Hk = sqrtf(norm_Hk);
        rel_diff = (norm_Hk > 1e-10f) ? (diff / norm_Hk) : diff;

        if (iter < 10) {
            residualHistory[iter] = rel_diff;
            residualHistoryCount = iter + 1;
        }

        matrixCopy(Ak_next, Ak, nn);
        matrixCopy(Gk_next, Gk, nn);
        matrixCopy(Hk_next, Hk, nn);

        if (rel_diff < relTolerance) {
            converged = true;
            lastIterations = iter + 1;
            lastStepDelta = rel_diff;
            lastStepIsBitExactZero = bit_exact;
            lastOutcome = SolveOutcome::Converged;
            break;
        }
    }

    if (!converged) {
        lastIterations = maxIterations;
        lastStepDelta = rel_diff;
        lastStepIsBitExactZero = bit_exact;
        lastOutcome = breakdown ? SolveOutcome::Breakdown : SolveOutcome::Budget;
    }

    // P = Hk (solução final)
    matrixCopy(Hk, P, nn);

    // Forçar simetria de P
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float avg = (P[i * n + j] + P[j * n + i]) * 0.5f;
            P[i * n + j] = avg;
            P[j * n + i] = avg;
        }
    }

    // K = (R + B'·P·B)^(-1) · B'·P·A
    float* BT_P = new float[m * n];
    float* BT_P_B = new float[mm];
    float* BT_P_A = new float[m * n];
    float* R_plus_BTPB = new float[mm];

    matrixMultiply(BT, P, BT_P, m, n, n);
    matrixMultiply(BT_P, B, BT_P_B, m, n, m);
    matrixAdd(R, BT_P_B, R_plus_BTPB, m, m);

    if (!invertMatrix(R_plus_BTPB, R_plus_BTPB, m)) {
        converged = false;
        lastOutcome = SolveOutcome::Breakdown; // (R+B'PB) singular no cálculo final de K
    } else {
        matrixMultiply(BT_P, A, BT_P_A, m, n, n);
        matrixMultiply(R_plus_BTPB, BT_P_A, K, m, m, n);
    }

    residualDirty = true;

    // Limpeza
    delete[] Ak; delete[] Gk; delete[] Hk;
    delete[] Ak_next; delete[] Gk_next; delete[] Hk_next;
    delete[] R_inv; delete[] BT; delete[] AT;
    delete[] V; delete[] Temp1; delete[] Temp2; delete[] Temp3;
    delete[] BT_P; delete[] BT_P_B; delete[] BT_P_A; delete[] R_plus_BTPB;

    return converged;
}

// ============================================================================
// Variantes fixed-point Q13.18 dos métodos de doubling — ver plano da
// auditoria e docs/auditoria_solvers_riccati.md. Setup/extração de cada uma
// mimetiza o par float correspondente (comentários lá têm a derivação
// completa); aqui só a aritmética muda. Mesmo gate n==6,m==3 e mesma
// convenção de fallback do SDA_FIXED original.
// ============================================================================

bool AutoLQR::computeGainMatrixADDA_Fixed()
{
    // Forma V/W com duas inversões (ao contrário do par float, que já usa só
    // V via push-through) — mantida deliberadamente para medir se a ordem de
    // multiplicação W·H vs. H·V quantiza diferente em Q13.18.
    using namespace fxq;
    const int n = stateSize, m = controlSize;
    const int sh = Q_SHIFT_DEFAULT;

    if (!A || !B || !Q || !R || !K) return false;
    if (n != 6 || m != 3) return false;
    if (!isSystemControllable()) return false;

    Status st;
    q_t Aq[36], Bq[18], Qq[36], Rq[9];
    for (int i = 0; i < n * n; i++) { Aq[i] = f2q(A[i], sh, &st); Qq[i] = f2q(Q[i], sh, &st); }
    for (int i = 0; i < n * m; i++)  Bq[i] = f2q(B[i], sh, &st);
    for (int i = 0; i < m * m; i++)  Rq[i] = f2q(R[i], sh, &st);
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t BT[18], Gk[36], Hk[36], Ak[36];
    transpose_q(Bq, BT, n, m);
    {
        q_t Rinv[9], BRi[18];
        if (!invert_q(Rq, Rinv, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(Bq, Rinv, BRi, n, m, m, sh, &st);
        matmul_q(BRi, BT, Gk, n, m, n, sh, &st);
    }
    memcpy(Hk, Qq, sizeof(Hk));
    memcpy(Ak, Aq, sizeof(Ak));
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    if (!doubling_loop_q(Ak, Gk, Hk, n, sh, Variant::AlternatingVW, maxIterations, invRelTolerance, nullptr, &st)) {
        lastOutcome = SolveOutcome::Breakdown; return false;
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t Kq[18];
    {
        q_t BTP[18], BTPB[9], Rp[9], BTPA[18];
        matmul_q(BT, Hk, BTP, m, n, n, sh, &st);
        matmul_q(BTP, Bq, BTPB, m, n, m, sh, &st);
        add_q(Rq, BTPB, Rp, m * m);
        if (!invert_q(Rp, Rp, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(BTP, Aq, BTPA, m, n, n, sh, &st);
        matmul_q(Rp, BTPA, Kq, m, m, n, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    for (int i = 0; i < m * n; i++) K[i] = q2f(Kq[i], sh);
    if (P) for (int i = 0; i < n * n; i++) P[i] = q2f(Hk[i], sh);

    lastIterations = st.iterations;
    lastStepDelta = st.rel_step;
    lastStepIsBitExactZero = st.bit_exact_zero;
    lastOutcome = (st.iterations < maxIterations) ? SolveOutcome::Converged : SolveOutcome::Budget;
    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;
    residualDirty = true;
    lastFixedPointMaxAbsSeen = fxq::q2f(st.max_abs_seen, sh);
    return lastOutcome == SolveOutcome::Converged;
}

bool AutoLQR::computeGainMatrixASDA_Fixed()
{
    using namespace fxq;
    const int n = stateSize, m = controlSize;
    const int sh = Q_SHIFT_DEFAULT;

    if (!A || !B || !Q || !R || !K) return false;
    if (n != 6 || m != 3) return false;
    if (!isSystemControllable()) return false;

    Status st;
    q_t Aq[36], Bq[18], Qq[36], Rq[9];
    for (int i = 0; i < n * n; i++) { Aq[i] = f2q(A[i], sh, &st); Qq[i] = f2q(Q[i], sh, &st); }
    for (int i = 0; i < n * m; i++)  Bq[i] = f2q(B[i], sh, &st);
    for (int i = 0; i < m * m; i++)  Rq[i] = f2q(R[i], sh, &st);
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t BT[18], Gk[36], Hk[36], Ak[36];
    transpose_q(Bq, BT, n, m);
    {
        q_t Rinv[9], BRi[18];
        if (!invert_q(Rq, Rinv, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(Bq, Rinv, BRi, n, m, m, sh, &st);
        matmul_q(BRi, BT, Gk, n, m, n, sh, &st);
    }
    memcpy(Hk, Qq, sizeof(Hk));
    memcpy(Ak, Aq, sizeof(Ak));
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    // Rescale (G,H)->(sG,H/s) acontece a cada iteração dentro do laço
    // (inclusive a primeira — equivalente ao s0 inicial + iteração 0
    // redundante do par float, ver docs/auditoria_solvers_riccati.md).
    float cum_s = 1.0f;
    if (!doubling_loop_q(Ak, Gk, Hk, n, sh, Variant::AdaptiveScaling, maxIterations, invRelTolerance, &cum_s, &st)) {
        lastOutcome = SolveOutcome::Breakdown; return false;
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    // P = H_final * cum_s — desfeito fora do laço em float (cum_s é produto
    // de ~10 fatores em [0,1;10], não cabe garantido em Q13.18) — depois
    // requantizado para terminar o cálculo de K em fixed-point.
    q_t Pq[36];
    for (int i = 0; i < n * n; i++) {
        float pf = q2f(Hk[i], sh) * cum_s;
        Pq[i] = f2q(pf, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t Kq[18];
    {
        q_t BTP[18], BTPB[9], Rp[9], BTPA[18];
        matmul_q(BT, Pq, BTP, m, n, n, sh, &st);
        matmul_q(BTP, Bq, BTPB, m, n, m, sh, &st);
        add_q(Rq, BTPB, Rp, m * m);
        if (!invert_q(Rp, Rp, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(BTP, Aq, BTPA, m, n, n, sh, &st);
        matmul_q(Rp, BTPA, Kq, m, m, n, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    for (int i = 0; i < m * n; i++) K[i] = q2f(Kq[i], sh);
    if (P) for (int i = 0; i < n * n; i++) P[i] = q2f(Pq[i], sh);

    lastIterations = st.iterations;
    lastStepDelta = st.rel_step;
    lastStepIsBitExactZero = st.bit_exact_zero;
    lastOutcome = (st.iterations < maxIterations) ? SolveOutcome::Converged : SolveOutcome::Budget;
    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;
    residualDirty = true;
    lastFixedPointMaxAbsSeen = fxq::q2f(st.max_abs_seen, sh);
    return lastOutcome == SolveOutcome::Converged;
}

bool AutoLQR::computeGainMatrixSDA_Scaled_Fixed()
{
    using namespace fxq;
    const int n = stateSize, m = controlSize;
    const int sh = Q_SHIFT_DEFAULT;

    if (!A || !B || !Q || !R || !K) return false;
    if (n != 6 || m != 3) return false;
    if (!isSystemControllable()) return false;

    Status st;
    q_t Aq[36], Bq[18], Qq[36], Rq[9];
    for (int i = 0; i < n * n; i++) { Aq[i] = f2q(A[i], sh, &st); Qq[i] = f2q(Q[i], sh, &st); }
    for (int i = 0; i < n * m; i++)  Bq[i] = f2q(B[i], sh, &st);
    for (int i = 0; i < m * m; i++)  Rq[i] = f2q(R[i], sh, &st);
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t BT[18];
    transpose_q(Bq, BT, n, m);
    q_t G0[36];
    {
        q_t Rinv[9], BRi[18];
        if (!invert_q(Rq, Rinv, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(Bq, Rinv, BRi, n, m, m, sh, &st);
        matmul_q(BRi, BT, G0, n, m, n, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    // D diagonal por normas de linha de A (float — n=6 valores, sqrt barato;
    // mesma heurística do par float computeGainMatrixSDA_Scaled()).
    float Df[6], Dinvf[6];
    for (int i = 0; i < n; i++) {
        float row_norm = 0.0f;
        for (int j = 0; j < n; j++) row_norm += A[i * n + j] * A[i * n + j];
        row_norm = sqrtf(row_norm);
        Df[i] = (row_norm > 1e-10f) ? (1.0f / sqrtf(row_norm)) : 1.0f;
        Dinvf[i] = 1.0f / Df[i];
    }
    q_t Dq[6], Dinvq[6];
    for (int i = 0; i < n; i++) { Dq[i] = f2q(Df[i], sh, &st); Dinvq[i] = f2q(Dinvf[i], sh, &st); }

    // Â=DAD⁻¹, Ĝ=DGD, Ĥ=D⁻¹QD⁻¹ (D diagonal ⇒ 3n² mults escalares)
    q_t Ak[36], Gk[36], Hk[36];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ak[i * n + j] = qmul(qmul(Dq[i], Aq[i * n + j], sh, &st), Dinvq[j], sh, &st);
            Gk[i * n + j] = qmul(qmul(Dq[i], G0[i * n + j], sh, &st), Dq[j], sh, &st);
            Hk[i * n + j] = qmul(qmul(Dinvq[i], Qq[i * n + j], sh, &st), Dinvq[j], sh, &st);
        }
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    if (!doubling_loop_q(Ak, Gk, Hk, n, sh, Variant::Standard, maxIterations, invRelTolerance, nullptr, &st)) {
        lastOutcome = SolveOutcome::Breakdown; return false;
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    // P = D · Ĥ · D (recuperação, mesma convenção corrigida do par float)
    q_t Pq[36];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Pq[i * n + j] = qmul(qmul(Dq[i], Hk[i * n + j], sh, &st), Dq[j], sh, &st);
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t Kq[18];
    {
        q_t BTP[18], BTPB[9], Rp[9], BTPA[18];
        matmul_q(BT, Pq, BTP, m, n, n, sh, &st);
        matmul_q(BTP, Bq, BTPB, m, n, m, sh, &st);
        add_q(Rq, BTPB, Rp, m * m);
        if (!invert_q(Rp, Rp, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(BTP, Aq, BTPA, m, n, n, sh, &st);
        matmul_q(Rp, BTPA, Kq, m, m, n, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    for (int i = 0; i < m * n; i++) K[i] = q2f(Kq[i], sh);
    if (P) for (int i = 0; i < n * n; i++) P[i] = q2f(Pq[i], sh);

    lastIterations = st.iterations;
    lastStepDelta = st.rel_step;
    lastStepIsBitExactZero = st.bit_exact_zero;
    lastOutcome = (st.iterations < maxIterations) ? SolveOutcome::Converged : SolveOutcome::Budget;
    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;
    residualDirty = true;
    lastFixedPointMaxAbsSeen = fxq::q2f(st.max_abs_seen, sh);
    return lastOutcome == SolveOutcome::Converged;
}

bool AutoLQR::computeGainMatrixSDA_SS_Fixed()
{
    // Setup do pencil 12x12 no MESMO shift do resto (Q13.18) — não Q9.22
    // como uma primeira versão tentou. O que limita a faixa não é N1
    // (‖N1‖∞ ~200-700 nos casos testados), é Φ=N1⁻¹: medido em toda a
    // bateria de 225 casos, ‖Φ‖∞ tem mediana 738 mas passa de 46000 no caso
    // adversarial C5 (disparidade de escala 1000×) — nenhum shift fixo único
    // cobre as duas pontas sem perder resolução onde importa. Q13.18 cobre
    // 224/225 casos (só falha o C5, um stress test deliberado, não uma
    // condição de voo real) — ver docs/auditoria_solvers_riccati.md.
    // Blocos em escopos aninhados de propósito — o maior consumo de stack
    // (N1/Phi/Phi11-22, ~2,3kB) sai de escopo antes do laço de duplicação,
    // que por sua vez chama invert_q (~1,15kB) de novo.
    using namespace fxq;
    const int n = stateSize, m = controlSize;
    const int sh = Q_SHIFT_DEFAULT;
    const float gamma = ssGamma; // shift ajustável (setSDASSGamma), default 0.5 — ver Exp. 3
    const int n2 = 2 * n;

    if (!A || !B || !Q || !R || !K) return false;
    if (n != 6 || m != 3) return false;
    if (!isSystemControllable()) return false;

    Status st;
    q_t Aq[36], Bq[18], Qq[36], Rq[9];
    for (int i = 0; i < n * n; i++) { Aq[i] = f2q(A[i], sh, &st); Qq[i] = f2q(Q[i], sh, &st); }
    for (int i = 0; i < n * m; i++)  Bq[i] = f2q(B[i], sh, &st);
    for (int i = 0; i < m * m; i++)  Rq[i] = f2q(R[i], sh, &st);
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t BT[18];
    transpose_q(Bq, BT, n, m);

    q_t G0[36];
    {
        q_t Rinv[9], BRi[18];
        if (!invert_q(Rq, Rinv, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(Bq, Rinv, BRi, n, m, m, sh, &st);
        matmul_q(BRi, BT, G0, n, m, n, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t gammaQ = f2q(gamma, sh, &st);
    q_t oneQ   = f2q(1.0f, sh, &st);

    q_t Ak[36], Gk[36], Hk[36]; // (Â,Ĝ,Ĥ) — montadas no bloco abaixo
    {
        q_t Ak2[36], Gk2[36], Hk2[36];
        {
            // N1 = [[I-γA, -γG0], [γH0, I-γA']] (2n×2n); H0=Q
            q_t N1[144], Phi[144];
            memset(N1, 0, sizeof(N1));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    N1[i * n2 + j]       = ((i == j) ? oneQ : 0) - qmul(gammaQ, Aq[i * n + j], sh, &st);
                    N1[i * n2 + (n + j)] = -qmul(gammaQ, G0[i * n + j], sh, &st);
                    N1[(n + i) * n2 + j] =  qmul(gammaQ, Qq[i * n + j], sh, &st);
                    N1[(n + i) * n2 + (n + j)] = ((i == j) ? oneQ : 0) - qmul(gammaQ, Aq[j * n + i], sh, &st);
                }
            }
            if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

            if (!invert_q(N1, Phi, n2, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; } // pencil singular/fora de faixa → fallback
            if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

            q_t Phi11[36], Phi12[36], Phi21[36], Phi22[36];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Phi11[i * n + j] = Phi[i * n2 + j];
                    Phi12[i * n + j] = Phi[i * n2 + (n + j)];
                    Phi21[i * n + j] = Phi[(n + i) * n2 + j];
                    Phi22[i * n + j] = Phi[(n + i) * n2 + (n + j)];
                }
            }

            q_t AmG[36], ATmG[36];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    AmG[i * n + j]  = Aq[i * n + j] - ((i == j) ? gammaQ : 0);
                    ATmG[i * n + j] = Aq[j * n + i] - ((i == j) ? gammaQ : 0);
                }
            }

            // Â = Φ11·(A-γI) - Φ12·Q ; Ĝ = Φ11·G0 + Φ12·(A'-γI) ; Ĥ = -Φ21·(A-γI) + Φ22·Q
            q_t T1[36], T2[36];
            matmul_q(Phi11, AmG, T1, n, n, n, sh, &st);
            matmul_q(Phi12, Qq, T2, n, n, n, sh, &st);
            sub_q(T1, T2, Ak2, n * n);

            matmul_q(Phi11, G0, T1, n, n, n, sh, &st);
            matmul_q(Phi12, ATmG, T2, n, n, n, sh, &st);
            add_q(T1, T2, Gk2, n * n);

            matmul_q(Phi21, AmG, T1, n, n, n, sh, &st);
            matmul_q(Phi22, Qq, T2, n, n, n, sh, &st);
            sub_q(T2, T1, Hk2, n * n);
        }
        if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

        memcpy(Ak, Ak2, sizeof(Ak));
        memcpy(Gk, Gk2, sizeof(Gk));
        memcpy(Hk, Hk2, sizeof(Hk));
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    if (!doubling_loop_q(Ak, Gk, Hk, n, sh, Variant::Standard, maxIterations, invRelTolerance, nullptr, &st)) {
        lastOutcome = SolveOutcome::Breakdown; return false;
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    // K = (R+B'PB)^-1 B'PA, com A ORIGINAL (Aq); P=Hk já é o da DARE original
    // (mesma propriedade do par float — ver comentário de computeGainMatrixSDA_SS()).
    q_t Kq[18];
    {
        q_t BTP[18], BTPB[9], Rp[9], BTPA[18];
        matmul_q(BT, Hk, BTP, m, n, n, sh, &st);
        matmul_q(BTP, Bq, BTPB, m, n, m, sh, &st);
        add_q(Rq, BTPB, Rp, m * m);
        if (!invert_q(Rp, Rp, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(BTP, Aq, BTPA, m, n, n, sh, &st);
        matmul_q(Rp, BTPA, Kq, m, m, n, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    for (int i = 0; i < m * n; i++) K[i] = q2f(Kq[i], sh);
    if (P) for (int i = 0; i < n * n; i++) P[i] = q2f(Hk[i], sh);

    lastIterations = st.iterations;
    lastStepDelta = st.rel_step;
    lastStepIsBitExactZero = st.bit_exact_zero;
    lastOutcome = (st.iterations < maxIterations) ? SolveOutcome::Converged : SolveOutcome::Budget;
    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;
    residualDirty = true;
    lastFixedPointMaxAbsSeen = fxq::q2f(st.max_abs_seen, sh);
    return lastOutcome == SolveOutcome::Converged;
}

bool AutoLQR::computeGainMatrixIterative_Fixed()
{
    // Value iteration em fixed-point (recorrência própria, não usa
    // doubling_loop_q). Warm-start compartilha o mesmo P_warm (float) do
    // par float computeGainMatrixIterative() — é o mesmo P físico sendo
    // aproximado, não importa a aritmética que o produziu; a alternativa de
    // um buffer fixed-point dedicado só complicaria sem motivo real.
    using namespace fxq;
    const int n = stateSize, m = controlSize;
    const int sh = Q_SHIFT_DEFAULT;
    const int nn = n * n;
    // maxIterations/invRelTolerance agora vêm de setStoppingCriterion() — mesmo
    // orçamento do par float por padrão (100), configurável para a campanha de
    // varredura de tolerância (ver docs/auditoria_solvers_riccati.md, Seção 13).

    if (!A || !B || !Q || !R || !K || !P_warm) return false;
    if (n != 6 || m != 3) return false;
    if (!isSystemControllable()) return false;

    Status st;
    q_t Aq[36], Bq[18], Qq[36], Rq[9];
    for (int i = 0; i < n * n; i++) { Aq[i] = f2q(A[i], sh, &st); Qq[i] = f2q(Q[i], sh, &st); }
    for (int i = 0; i < n * m; i++)  Bq[i] = f2q(B[i], sh, &st);
    for (int i = 0; i < m * m; i++)  Rq[i] = f2q(R[i], sh, &st);
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t AT[36], BT[18];
    transpose_q(Aq, AT, n, n);
    transpose_q(Bq, BT, n, m);

    // Regularização: 1e-8 do par float vira zero exato em Q13.18 (0,0026 LSB)
    // — sobe conscientemente para 1e-5 (2,6 LSB), acima do piso de quantização.
    q_t eps = f2q(1e-5f, sh, &st);

    float Pw_norm = 0.0f;
    for (int i = 0; i < nn; i++) Pw_norm += fabsf(P_warm[i]);
    bool has_warm_start = (Pw_norm > 1e-6f);

    q_t Pk[36];
    if (has_warm_start) {
        for (int i = 0; i < nn; i++) Pk[i] = f2q(P_warm[i], sh, &st);
    } else {
        memcpy(Pk, Qq, sizeof(Pk));
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    int iters = maxIterations;
    float relF = 1.0f; // sobrevive ao loop p/ lastStepDelta mesmo sem convergência
    bool bitExact = false;
    for (int it = 0; it < maxIterations; it++) {
        q_t PA[36], PB[18], ATPA[36], BTPB[9], BTPA[18];
        matmul_q(Pk, Aq, PA, n, n, n, sh, &st);
        matmul_q(Pk, Bq, PB, n, n, m, sh, &st);
        matmul_q(AT, PA, ATPA, n, n, n, sh, &st);
        matmul_q(BT, PB, BTPB, m, n, m, sh, &st);
        matmul_q(BT, PA, BTPA, m, n, n, sh, &st);

        q_t S[9];
        add_q(Rq, BTPB, S, m * m);
        for (int i = 0; i < m; i++) S[i * m + i] += eps;

        q_t Sinv[9];
        if (!invert_q(S, Sinv, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }

        q_t Ktmp[18], ATPB[18], corr[36];
        matmul_q(Sinv, BTPA, Ktmp, m, m, n, sh, &st);
        matmul_q(AT, PB, ATPB, n, n, m, sh, &st);
        matmul_q(ATPB, Ktmp, corr, n, m, n, sh, &st);

        q_t Pnext[36];
        for (int i = 0; i < nn; i++) Pnext[i] = Qq[i] + ATPA[i] - corr[i];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                q_t avg = (Pnext[i * n + j] + Pnext[j * n + i]) / 2;
                Pnext[i * n + j] = avg;
                Pnext[j * n + i] = avg;
            }

        // Critério de parada: norma de Frobenius relativa, calculada em
        // float a partir dos valores Q13.18 convertidos (q2f) — mesma norma
        // do caminho float (computeGainMatrixIterative()) e do kernel
        // compartilhado (FixedPointQ.cpp:doubling_loop_q). Soma de quadrados
        // de até 36 termos Q13.18 (~2^31 cada) estouraria int64 perto do
        // teto ±8192; em float o custo é desprezível frente às matmuls.
        float diffSq = 0.0f, hSq = 0.0f;
        bitExact = true;
        for (int i = 0; i < nn; i++) {
            if (Pnext[i] != Pk[i]) bitExact = false;
            float d = q2f(Pnext[i], sh) - q2f(Pk[i], sh);
            float h = q2f(Pk[i], sh);
            diffSq += d * d;
            hSq += h * h;
        }
        memcpy(Pk, Pnext, sizeof(Pk));
        relF = (hSq > 1e-20f) ? sqrtf(diffSq / hSq) : sqrtf(diffSq);
        if (relF < (1.0f / (float)invRelTolerance)) { iters = it + 1; break; }
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    q_t Kq[18];
    {
        q_t PA[36], PB[18], BTPB[9], BTPA[18], S[9];
        matmul_q(Pk, Aq, PA, n, n, n, sh, &st);
        matmul_q(Pk, Bq, PB, n, n, m, sh, &st);
        matmul_q(BT, PB, BTPB, m, n, m, sh, &st);
        matmul_q(BT, PA, BTPA, m, n, n, sh, &st);
        add_q(Rq, BTPB, S, m * m);
        for (int i = 0; i < m; i++) S[i * m + i] += eps;
        if (!invert_q(S, S, m, sh, &st)) { lastOutcome = SolveOutcome::Breakdown; return false; }
        matmul_q(S, BTPA, Kq, m, m, n, sh, &st);
    }
    if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }

    for (int i = 0; i < m * n; i++) K[i] = q2f(Kq[i], sh);
    if (P) for (int i = 0; i < n * n; i++) P[i] = q2f(Pk[i], sh);
    for (int i = 0; i < nn; i++) P_warm[i] = q2f(Pk[i], sh); // atualiza o warm-start p/ a próxima chamada

    lastIterations = iters;
    lastStepDelta = relF;
    lastStepIsBitExactZero = bitExact;
    // breakdown já tratado acima (retorna antes de chegar aqui); o que resta
    // distinguir é convergência dentro do orçamento (Converged) contra
    // esgotamento sem breakdown (Budget) — mesma taxonomia dos demais
    // métodos, ver SolveOutcome e docs/auditoria_solvers_riccati.md Seção 13.
    lastOutcome = (iters < maxIterations) ? SolveOutcome::Converged : SolveOutcome::Budget;
    residualHistoryCount = 0;
    for (int i = 0; i < 10; i++) residualHistory[i] = 0.0f;
    residualDirty = true;
    lastFixedPointMaxAbsSeen = fxq::q2f(st.max_abs_seen, sh);
    return lastOutcome == SolveOutcome::Converged;
}

int AutoLQR::getLastIterations() const {
    return lastIterations;
}

float AutoLQR::getLastResidual() const {
    if (residualDirty) {
        lastResidual = computeDareResidualNorm();
        residualDirty = false;
    }
    return lastResidual;
}

float AutoLQR::getLastStepDelta() const {
    return lastStepDelta;
}

bool AutoLQR::getLastStepIsBitExactZero() const {
    return lastStepIsBitExactZero;
}

float AutoLQR::getLastFixedPointMaxAbsSeen() const {
    return lastFixedPointMaxAbsSeen;
}

int AutoLQR::getResidualHistory(float* residuals) const {
    for (int i = 0; i < 10; i++) {
        residuals[i] = residualHistory[i];
    }
    return residualHistoryCount;
}
