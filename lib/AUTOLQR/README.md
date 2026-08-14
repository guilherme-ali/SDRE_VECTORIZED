# AutoLQR - Biblioteca de Controle LQR Adaptativo

Biblioteca otimizada para cálculo de ganhos LQR em tempo real, projetada para sistemas embarcados como ESP32. Implementa múltiplos algoritmos para resolver a Equação Algébrica de Riccati Discreta (DARE).

## 📋 Características

- **Caminho de produção `SDA_FIXED`** (fixed-point Q13.18) — default de `computeGains()`, derivado do SDA base
- **8 métodos de solução DARE em `float`** (SDA, SDA-ss, ASDA, SDA Scaled, ADDA, Schur, Van Dooren, Iterativo) — Schur não tem chamador no repositório (órfão)
- **6 variantes fixed-point Q13.18** (SDA, SDA-ss, ASDA, SDA Scaled, ADDA, Iterativo — Schur e Van Dooren ficam só em `float`), kernel compartilhado em `FixedPointQ.{h,cpp}`, todas 1,5-1,85× mais rápidas que o par `float` no ESP32-S2 real
- **Operações matriciais otimizadas** para sistemas de pequeno/médio porte (`MatrixOperations`)
- **Warm-start** automático para o método iterativo (buffer dedicado, não compartilhado com os demais métodos)
- Cada solver aloca seus buffers de trabalho por chamada (`new[]`/`delete[]`); os métodos fixed-point usam só stack

## 🔧 Métodos Disponíveis

### Comparação de Performance (ESP32-S2 @ 240MHz)

Sistema de teste: **6 estados × 3 controles**, **800 000 execuções** sob dinâmica real de quadricóptero (CBA 2026).

> **Tabela pré-correção.** Medida antes do fix de SDA-SS/ASDA/SDA_SCALED/ADDA (ver aviso no topo).
> As colunas "Falhas" e "Erro RMS" desta tabela refletiam os bugs algébricos, não o desempenho real
> dos algoritmos — o resíduo aqui não era o resíduo da DARE (ver `getLastResidual()`, corrigido).

#### Tempo de execução

| Método | Identificador | Média (μs) | σ (μs) | Pior caso (μs) | Falhas / 800k | Iter. (méd ± σ) | Iter. (pior) |
|--------|---------------|------------|--------|----------------|---------------|------------------|--------------|
| **SDA (base)** | `"SDA"` | **8663,59** | **146,49** | **8750** | **0** | 7,99 ± 0,13 | 8 |
| SDA Single-Shift | `"SDA_SS"` | 8413,26 | 541,55 | 10902 | 55 349 | 7,79 ± 0,52 | 10 |
| ASDA (Adaptativo) | `"ASDA"` | 9114,64 | **24,64** | 9180 | 0 | 8,00 ± 0,00 | 8 |
| SDA Scaled | `"SDA_SCALED"` | 8854,91 | 327,95 | 11024 | 48 302 | 8,08 ± 0,31 | 10 |
| SDA-ADDA | `"ADDA"` | 10754,63 | 556,55 | 13654 | 40 314 | 7,96 ± 0,42 | 10 |
| Van Dooren | `"VAN_DOOREN"` | 39281,13 | 3637,45 | 126877 | 0 | 1,00 ± 0,00 | 1 |
| Iterativo | `"ITERATIVE"` | 11912,00 | 2868,74 | 16884 | 0 | 22,49 ± 5,64 | 32 |

#### Precisão (erro RMS dos ganhos K vs. método iterativo)

| Método | Erro RMS |
|--------|----------|
| **SDA (base)** | **9,36 × 10⁻⁷** |
| ASDA | 1,92 × 10⁻⁵ |
| Van Dooren | 5,53 × 10⁻⁵ |
| SDA-ADDA | 1,85 × 10⁻⁴ |
| SDA-SS | 3,22 × 10⁻⁴ |
| SDA-Scaled | 3,43 × 10⁻⁴ |
| Iterativo | — (referência) |

### Descrição dos Métodos

#### 0. SDA_FIXED (SDA em ponto fixo Q13.18) ⭐ **PADRÃO DO FIRMWARE**
```cpp
lqr.computeGains();            // default = "SDA_FIXED"
lqr.computeGains("SDA_FIXED"); // equivalente, explícito
```
- **Melhor para**: Execução em tempo real no ESP32-S2 (sem FPU) — **default de `computeGains()`**
- **Características**: Mesma recorrência do SDA base, resolvida inteira em `int32` Q13.18 (~2,7× mais rápida)
- **Precisão**: erro do `K` < 1 % no ganho dominante (pura quantização)
- **Sem fallback automático**: em overflow/saturação ou matriz singular retorna `false`; quem chama
  mantém o `K` do ciclo anterior. Para forçar o caminho `float` exato, selecionar `"SDA"`.
- Ver [seção dedicada](#-caminho-rápido-sda-em-ponto-fixo-q1318).
- **Referência**: mesma recorrência de `"SDA"` (ver #1) — a aritmética Q13.18 é engenharia própria
  deste projeto, sem paper específico.

#### 1. SDA (Structure-preserving Doubling Algorithm) — referência `float`
```cpp
lqr.computeGains("SDA");
```
- **Melhor para**: Referência exata e *fallback* manual do `SDA_FIXED`
- **Características**: Convergência quadrática, preserva estrutura simpléctica, aritmética `float` pura
- **Complexidade**: O(n³) por iteração, ~8–10 iterações em regime
- **Robustez comprovada**: 0 falhas em 800 000 execuções; menor erro RMS (9,36×10⁻⁷)
- **Referência**: Chu, E.K.-W., Fan, H.-Y., Lin, W.-W., Wang, C.-S. "Structure-preserving
  algorithms for periodic discrete-time algebraic Riccati equations." *Int. J. Control*
  77(8):767-788, 2004 (box "SDA algorithm", p.770). Origem: Anderson, B.D.O. "Second-order
  convergent algorithms for the steady-state Riccati equation." IEEE CDC, 1978 (eqs. 4a-4c).

#### 2. SDA-ss (SDA com shift real único)
```cpp
lqr.computeGains("SDA_SS");
```
- **Melhor para**: Sistemas com autovalor dominante próximo de 1 (caso crítico do SDRE embarcado)
- **Características**: Desloca o *pencil* simplético inteiro `M-γL, L-γM` (γ=0,5 fixo), preservando
  autovetores e portanto `P` — ver derivação e prova numérica em `AutoLQR.cpp` e
  `docs/auditoria_solvers_riccati.md`. A versão anterior deslocava só `A`, o que resolvia uma DARE
  diferente (erro de até ~40 % confirmado em float64); corrigido nesta auditoria.
- **Trade-off**: Reduz iterações quando há autovalor perto do círculo unitário; sem busca de γ ótimo
  (Fibonacci, Chu-Fan-Lin 2005) — registrado como trabalho futuro.
- **Referência**: Chu, E.K.-W., Fan, H.-Y., Lin, W.-W. "A structure-preserving doubling algorithm
  for continuous-time algebraic Riccati equations." *Linear Algebra Appl.* 396:55-80, 2005
  (técnica de shift γ com busca de Fibonacci; adaptada aqui de CARE→DARE para DARE→DARE). **Não**
  é o SDA-ss de Guo-Iannazzo-Meini nem o *shrink-and-shift* de Bini-Meini-Poloni — aqueles são para
  NARE não-simétrica de M-matriz, problema diferente da DARE simétrica deste projeto.

#### 3. ASDA (SDA com escalonamento adaptativo)
```cpp
lqr.computeGains("ASDA");
```
- **Características**: A cada iteração aplica `(G,H)→(sG,H/s)`, `s=√(‖H‖/‖G‖)`; o produto acumulado
  `∏s_i` é revertido em `P = H_k·∏s_i` no final — a versão anterior não revertia (erro de até 7 ordens
  de grandeza confirmado em float64); corrigido nesta auditoria.
- **Referência**: Chu, E.K.-W., Fan, H.-Y., Lin, W.-W. "A structure-preserving doubling algorithm
  for continuous-time algebraic Riccati equations." *Linear Algebra Appl.* 396:55-80, 2005 — mesmo
  paper de `SDA_SS` (#2); cobre tanto a técnica de shift γ quanto o escalonamento adaptativo para
  robustez numérica do SDA. Conteúdo integral não verificado por leitura direta (acesso ao texto
  completo bloqueado nas fontes tentadas); citação confirmada pelo usuário.

#### 4. SDA Scaled
```cpp
lqr.computeGains("SDA_SCALED");
```
- **Melhor para**: Sistemas com normas de linha de `A` muito desiguais
- **Características**: Balanceamento diagonal `Â=DAD⁻¹, Ĝ=DGD, Ĥ=D⁻¹QD⁻¹`, `P=D·P̂·D` (Ward, 1981).
  A versão anterior tinha os expoentes de `D` invertidos em `Ĥ` e na extração de `P` (erro de até
  6 ordens de grandeza confirmado em float64); corrigido nesta auditoria.
- **Referência**: Ward, R.C. "Balancing the Generalized Eigenvalue Problem." *SIAM J. Sci. Stat.
  Comput.* 2(2):141-152, 1981 (técnica geral de balanceamento; a heurística de `D` por norma de
  linha usada aqui é uma simplificação própria do projeto, não o algoritmo iterativo completo do
  paper). Nenhum dos 8 papers de `SOLVERS/` cobre esta técnica diretamente.

#### 5. ADDA (SDA em forma alternada)
```cpp
lqr.computeGains("ADDA");
```
- **Características**: Calcula `V=(I+GH)⁻¹` e usa a identidade *push-through*
  `(I+HG)⁻¹H ≡ H(I+GH)⁻¹` na atualização de `H` — algebricamente **idêntico ao SDA**
  (verificado: `‖P_ADDA-P_SDA‖/‖P_SDA‖ ~ 1e-16` em float64). A versão anterior calculava as duas
  inversas `V` e `W` separadamente (mesmo resultado, custo extra). O ADDA de fato — dois parâmetros
  de shift α≠β sobre a MARE não-simétrica, Wang, Wang & Li 2012 — não está implementado; portar para
  a DARE simétrica é trabalho futuro (ver `docs/auditoria_solvers_riccati.md`).
- **Referência**: Wang, W., Wang, W., Li, R.-C. "Alternating-Directional Doubling Algorithm for
  M-Matrix Algebraic Riccati Equations." *SIAM J. Matrix Anal. Appl.* 33(1):170-194, 2012 — resolve
  a **MARE** não-simétrica `XDX-AX-XB+C=0` de M-matriz, problema diferente da DARE simétrica deste
  projeto. O que está implementado aqui sob este nome é o SDA (ver #1), não o algoritmo do paper.

#### 6. Van Dooren (Extended Symplectic Pencil)
```cpp
lqr.computeGains("VAN_DOOREN");
```
- **Melhor para**: Robustez numérica; único método que funciona mesmo com `R` singular
- **Características**: Usa pencil estendido (2n+m)×(2n+m) com deflação QR
- **Trade-off**: Significativamente mais lento; usa autovetores individuais em vez do subespaço
  deflacionário do QZ ordenado do paper original — mais frágil que o método publicado em casos
  extremos (não corrigido nesta auditoria, ver `docs/auditoria_solvers_riccati.md`)
- **Referência**: Van Dooren, P. "A Generalized Eigenvalue Approach for Solving Riccati Equations."
  *SIAM J. Sci. Stat. Comput.* 2(2):121-135, 1981 (Problema II, eq. 44/59, DARE discreta)

#### — Schur (órfão, sem chamador no repositório)
```cpp
lqr.computeGains("SCHUR");
```
- **Não recomendado**: `Eigen::ComplexSchur` sem reordenação — a seleção de colunas do subespaço
  estável por índice diagonal não garante base do subespaço invariante correto. Resíduo medido de
  ordens de grandeza $10^5$–$10^{10}$, `P` nem sempre positivo. Não corrigido nesta auditoria
  (nenhum chamador no repositório usa este método).
- **Referência**: Laub, A.J. "A Schur Method for Solving Algebraic Riccati Equations." *IEEE Trans.
  Automat. Control* 24(6):913-921, 1979 — o método publicado usa QZ **reordenado**; a implementação
  atual não reordena.

#### 7. Iterativo (Riccati Iteration)
```cpp
lqr.computeGains("ITERATIVE");
```
- **Melhor para**: Alta precisão, warm-start
- **Características**: Iteração direta da equação de Riccati (convergência **linear**, não
  quadrática como o SDA); warm-start usa buffer dedicado (`P_warm`), não herda a solução de outro
  método
- **Vantagem**: Excelente com warm-start (solução anterior como inicial)
- **Referência**: iteração de ponto-fixo clássica da DARE — não está em nenhum dos 8 papers de
  `SOLVERS/`; ver, e.g., Lancaster, P., Rodman, L. *The Algebraic Riccati Equation.* Oxford
  University Press, 1995 (cap. 2).

## 📊 Exemplo de Matriz K Resultante

Para o sistema de atitude de quadricóptero (6×3):

```
K [3 x 6]:
  [   0.012991,    0.000003,    0.000479,    0.001459,   -0.000000,   -0.000002]
  [   0.000034,    0.012779,   -0.001173,    0.000001,    0.001458,   -0.000002]
  [  -0.000775,    0.002094,    0.023238,   -0.000005,   -0.000000,    0.002624]
```

## 🚀 Uso Básico

### Inicialização

```cpp
#include <AutoLQR.h>

// Criar controlador: (número de estados, número de controles)
AutoLQR lqr(6, 3);

// Definir matrizes de custo Q e R
float Q[36] = {
    100, 0, 0, 0, 0, 0,
    0, 100, 0, 0, 0, 0,
    0, 0, 100, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1
};

float R[9] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
};

lqr.setCostMatrices(Q, R);
```

### Loop de Controle SDRE

```cpp
void loop() {
    // 1. Atualizar matrizes do sistema baseado no estado atual
    updateSystemMatrix(roll, pitch, yaw, p, q, r);
    
    // 2. Discretizar e configurar
    lqr.setStateMatrix(Ad);   // Matriz de estados discretizada
    lqr.setInputMatrix(Bd);   // Matriz de entrada discretizada
    
    // 3. Calcular ganhos (default = "SDA_FIXED", fixed-point Q13.18)
    lqr.computeGains();
    
    // 4. Calcular ação de controle
    lqr.updateState(current_state);
    lqr.updateReference(reference);
    
    float control[3];
    lqr.calculateControl(control);
    
    // 5. Aplicar aos atuadores
    applyControl(control);
}
```

### Exportar Ganhos

```cpp
float K[18];  // 3 controles × 6 estados
lqr.exportGains(K);

// Obter solução de Riccati P
const float* P = lqr.getRicattiSolution();
```

## 📐 API Completa

### Classe AutoLQR

```cpp
class AutoLQR {
public:
    // Construtor/Destrutor
    AutoLQR(int stateSize, int controlSize);
    ~AutoLQR();
    
    // Configuração do sistema
    bool setStateMatrix(const float* A);      // Matriz de estados (n×n)
    bool setInputMatrix(const float* B);      // Matriz de entrada (n×m)
    bool setCostMatrices(const float* Q, const float* R);  // Matrizes de custo
    
    // Cálculo de ganhos (default = "SDA_FIXED", fixed-point Q13.18)
    bool computeGains(const char* method = "SDA_FIXED");  // "SDA", "ASDA", "ADDA", "ITERATIVE", ...
    void setGains(const float* K);            // Definir ganhos manualmente
    
    // Controle
    void updateState(const float* state);     // Atualizar estado atual
    void updateReference(const float* ref);   // Atualizar referência
    void calculateControl(float* u);          // Calcular u = -Kx + Kr*r
    
    // Exportação
    bool exportGains(float* K);               // Exportar matriz K
    bool exportKr(float* Kr);                 // Exportar ganho de referência
    const float* getRicattiSolution() const;  // Obter matriz P
    
    // Utilitários
    bool isSystemControllable();              // Verificar controlabilidade
    float calculateExpectedCost();            // Calcular custo x'Px
    float estimateConvergenceTime(float threshold);
};
```

## ⚙️ Recomendações de Uso

### Para controle em tempo real (escolha padrão do projeto)
```cpp
// SDA_FIXED (default): fixed-point Q13.18, ~3.2 ms, erro do K < 1%.
// Em overflow/saturação retorna false e o ganho do ciclo anterior é mantido.
lqr.computeGains();
```

### Referência exata em float (fallback manual do SDA_FIXED)
```cpp
// SDA base: 0 falhas em 800k execuções, pior caso 8750 us, erro RMS 9.36e-7
lqr.computeGains("SDA");
```

### Quando previsibilidade temporal é crítica (jitter mínimo)
```cpp
// ASDA: desvio padrão de apenas 24.64 us (vs 146.49 us do SDA base)
lqr.computeGains("ASDA");
```

### Para validação offline / referência de precisão
```cpp
// Iterativo com warm-start: alta precisão, lento e jitter alto.
// NÃO recomendado em malha de tempo real.
lqr.computeGains("ITERATIVE");
```

## ⚡ Caminho rápido: SDA em ponto fixo (Q13.18)

O `ESP32-S2` **não tem FPU** — cada operação `float` é soft-float (~56 ciclos), e o custo do SDA é
dominado pelas 8 multiplicações 6×6 + inversão por iteração. Duas otimizações reduzem esse custo:

### 1. SDA inteiro em ponto fixo Q13.18

`computeGainMatrixSDA_Fixed()` resolve a DARE inteira em **`int32` ponto fixo** (formato **Q13.18**:
13 bits inteiros = ±8192, 18 fracionários ≈ resolução 3,8×10⁻⁶). É o método selecionado por
`computeGains("SDA_FIXED")` — e o **default** de `computeGains()`.

- **~2,7× mais rápido** que o SDA `float` (matmuls inteiros ≈ 4,4×; inversão ≈ soft-float).
- **Erro do `K` < 1 %** no ganho dominante — e o erro é **pura quantização** (não amplificado pelo
  condicionamento da Riccati; cai pela metade a cada bit fracionário a mais).
- **Formato escolhido pelo range real** do problema (pico ≈ 2980 nas matrizes do SDA): Q13.18 dá
  margem 2,7× sobre esse pico, importante porque o SDRE varia as matrizes com o estado.
- **Sem fallback automático**: se houver overflow/saturação (flag interna) ou matriz singular no domínio
  fixed-point, `computeGainMatrixSDA_Fixed()` retorna `false` e `computeGains()` propaga `false` sem
  recalcular. O chamador (ex.: `main.cpp`) então **mantém o `K` do ciclo anterior**. Para forçar o
  caminho `float` exato, chamar `computeGains("SDA")`.

### 2. Kernel de multiplicação com saída simétrica

`MatrixOperations::matrixMultiplySymOutput(a, b, c, n)` calcula `c = a·b` **assumindo `c` simétrica**:
computa só o triângulo superior (21 de 36 elementos em 6×6) e espelha. Usado nas duas atualizações do
SDA cujo produto é provadamente simétrico (identidade *push-through*: $W G_k$ e $H_k W$ são simétricas),
economizando ~42 % nessas matmuls. **Válido apenas quando o chamador garante a simetria do resultado.**

> Os benchmarks publicados (tabela acima, ~8,6 ms) referem-se ao SDA **`float`** (`"SDA"`) — que serve
> de referência exata e *fallback* manual. O caminho de produção atual (`"SDA_FIXED"`, fixed-point
> Q13.18, default de `computeGains()`) roda em ~3,2 ms.

### 3. Fixed-point para os demais métodos de doubling

`"SDA_SS_FIXED"`, `"ASDA_FIXED"`, `"SDA_SCALED_FIXED"`, `"ADDA_FIXED"` e `"ITERATIVE_FIXED"` levam
a mesma ideia do `SDA_FIXED` aos outros cinco métodos (exceto Schur e Van Dooren, que ficam só em
`float`). O kernel Q13.18 foi extraído para `FixedPointQ.{h,cpp}` — um único laço de duplicação
compartilhado (`fxq::doubling_loop_q`), parametrizado por `Variant` (`Standard`, `AdaptiveScaling`
para o ASDA, `AlternatingVW` para o ADDA), já que os quatro métodos de doubling só diferem em como
montam `(A₀,G₀,H₀)` e extraem `P` — o laço em si é idêntico. `ITERATIVE_FIXED` tem recorrência
própria (não usa o laço de duplicação).

Medido no ESP32-S2 físico (caso de hover real, `test/verify_gains_onboard.cpp`):

| Método | tempo float | tempo fixed | *speedup* | resíduo DARE (fixed) |
|---|---|---|---|---|
| `SDA_FIXED` (referência) | 7,76 ms | **4,44 ms** | 1,75× | 6,34×10⁻³ |
| `SDA_SS_FIXED` | 7,99 ms | **4,45 ms** | 1,80× | **4,03×10⁻³** (melhor que o SDA_FIXED) |
| `ASDA_FIXED` | 9,00 ms | 4,98 ms | 1,81× | 4,81×10⁻³ |
| `SDA_SCALED_FIXED` | 8,50 ms | 4,59 ms | 1,85× | 9,14×10⁻³ |
| `ADDA_FIXED` | 8,39 ms | 5,57 ms | 1,51× | 7,61×10⁻³ |
| `ITERATIVE_FIXED` (100 iter.) | 32,6 ms | 19,3 ms | 1,69× | 5,80×10⁻² (≈ igual ao float, 5,82×10⁻²) |

Achados:
- **Todas as variantes ganham entre 1,5× e 1,85×** sobre o par `float` correspondente no S2 real —
  o ganho do fixed-point não é exclusividade do SDA base.
- **`SDA_SS_FIXED` é o mais preciso *e* praticamente tão rápido quanto o `SDA_FIXED`** (4,45 ms vs.
  4,44 ms): o custo do setup do pencil 12×12 é amortizado por precisar de menos iterações do laço
  (8 vs. 10). O shift γ parece ajudar tanto a convergência quanto o comportamento sob quantização.
- **`ASDA_FIXED` vem em segundo** em precisão — o escalonamento adaptativo `(G,H)→(sG,H/s)` limita
  o crescimento de `‖G‖`, que é o que mais aproxima a faixa Q13.18 do teto (pico histórico
  `maxGk≈2980` contra ±8192, margem de só 2,7×).
- **`ADDA_FIXED` mantém deliberadamente as duas inversões** (`V` e `W` separadas, ao contrário do
  par `float` que já usa só `V` via push-through) — é o mais lento dos cinco por isso, mas serve
  para medir se a ordem de multiplicação `W·H` vs. `H·V` quantiza diferente (resultado: sim, dá um
  resíduo um pouco diferente do SDA_FIXED, mas não dramaticamente).
- **`ITERATIVE_FIXED` prova que a quantização não é o gargalo ali**: com o mesmo orçamento de 100
  iterações do `float`, o resíduo final é essencialmente igual (5,80×10⁻² vs. 5,82×10⁻²) — é a
  **convergência linear** que não dá conta em 100 iterações para estes sistemas rígidos, tanto em
  `float` quanto em fixed-point, não um efeito de precisão numérica.
- **Setup do `SDA_SS_FIXED`**: o pencil 12×12 (`N1=[[I-γA,-γG₀],[γQ,I-γA']]`, invertido para obter
  `Φ`) usa o **mesmo shift Q13.18** do resto — uma primeira tentativa em Q9.22 (mais resolução para
  as entradas de `N1`, que somam só ~200-700 de magnitude) **falhou em quase todos os casos**: o que
  limita a faixa não é `N1`, é `Φ=N1⁻¹`, cuja magnitude varia muito mais entre casos (mediana ~738,
  até ~46000 em cenários adversariais de escala) — nenhum shift fixo cobre as duas pontas. Q13.18
  cobre 224 dos 225 casos testados (falha só o caso deliberadamente adversarial de disparidade de
  escala 1000×) — ver `docs/auditoria_solvers_riccati.md`.

## 🔬 Detalhes de Implementação

### Equação DARE Resolvida

$$A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q = 0$$

### Cálculo do Ganho K

$$K = (R + B^T P B)^{-1} B^T P A$$

### Controle Aplicado

$$u = -K x + K_r r$$

onde $K_r$ é o ganho de referência para tracking.

## 📁 Arquivos

```
AUTOLQR/
├── AutoLQR.cpp / .h            # Classe principal (estende MatrixOperations)
├── MatrixOperations.cpp / .h   # Operações lineares (inv, mul, transp, QR, etc.)
├── FixedPointQ.cpp / .h        # Kernel Q13.18 compartilhado (laço de duplicação + primitivas)
└── README.md                   # Esta documentação
```

## 📚 Referências

Uma linha por método (`computeGains("...")`), na ordem da seção [Descrição dos Métodos](#descrição-dos-métodos).
Gabarito completo — com o que cada método *realmente* implementa vs. a referência citada, contraexemplos
e prova em float64 — em `docs/auditoria_solvers_riccati.md`.

| # | Método | Referência |
|---|---|---|
| 0 | `SDA_FIXED` | Mesma de `SDA` (#1) — a aritmética Q13.18 é engenharia própria, sem paper específico |
| 1 | `SDA` | Chu, E.K.-W., Fan, H.-Y., Lin, W.-W., Wang, C.-S. "Structure-preserving algorithms for periodic discrete-time algebraic Riccati equations." *Int. J. Control* 77(8):767-788, 2004 (box "SDA algorithm", p.770). Origem: Anderson, B.D.O. "Second-order convergent algorithms for the steady-state Riccati equation." IEEE CDC, 1978 (eqs. 4a-4c) |
| 2 | `SDA_SS` | Chu, E.K.-W., Fan, H.-Y., Lin, W.-W. "A structure-preserving doubling algorithm for continuous-time algebraic Riccati equations." *Linear Algebra Appl.* 396:55-80, 2005 (shift γ com busca de Fibonacci; γ=0,5 fixo aqui, sem a busca) |
| 3 | `ASDA` | Chu, E.K.-W., Fan, H.-Y., Lin, W.-W. "A structure-preserving doubling algorithm for continuous-time algebraic Riccati equations." *Linear Algebra Appl.* 396:55-80, 2005 — mesmo paper de `SDA_SS` (#2), técnica de escalonamento adaptativo |
| 4 | `SDA_SCALED` | Ward, R.C. "Balancing the Generalized Eigenvalue Problem." *SIAM J. Sci. Stat. Comput.* 2(2):141-152, 1981 (a heurística de `D` por norma de linha usada aqui é simplificação própria, não o algoritmo iterativo completo do paper) |
| 5 | `ADDA` | Implementado = SDA (#1) em forma alternada. Paper do ADDA de fato: Wang, W., Wang, W., Li, R.-C. "Alternating-Directional Doubling Algorithm for M-Matrix Algebraic Riccati Equations." *SIAM J. Matrix Anal. Appl.* 33(1):170-194, 2012 — resolve uma MARE não-simétrica, problema diferente, **não implementado** |
| 6 | `VAN_DOOREN` | Van Dooren, P. "A Generalized Eigenvalue Approach for Solving Riccati Equations." *SIAM J. Sci. Stat. Comput.* 2(2):121-135, 1981 (Problema II, eq. 44/59) |
| 7 | `ITERATIVE` | Iteração de ponto-fixo clássica — Lancaster, P., Rodman, L. *The Algebraic Riccati Equation.* Oxford University Press, 1995 (cap. 2) |
| — | `SCHUR` (órfão) | Laub, A.J. "A Schur Method for Solving Algebraic Riccati Equations." *IEEE Trans. Automat. Control* 24(6):913-921, 1979 (o paper usa QZ reordenado; a implementação atual não reordena) |

Contexto geral do SDRE: Çimen, T. "State-Dependent Riccati Equation (SDRE) Control: A Survey." IFAC
Proceedings Volumes, 2008.

## 📄 Licença

MIT License - Veja o arquivo LICENSE no diretório raiz do projeto.
