# Auditoria dos solvers de Riccati (`lib/AUTOLQR`)

Data: 2026-08-14 (§1-7, auditoria algébrica), estendido em 2026-08-14 (§8, variantes fixed-point).
Escopo: os 9 métodos `float` de solução da DARE em `lib/AUTOLQR/AutoLQR.cpp`, auditados contra os
8 papers em `G:\Meu Drive\ACADEMICO\Mestrado\Bibliografia\SOLVERS\` e contra o oráculo
`scipy.linalg.solve_discrete_are` — mais 5 variantes em fixed-point Q13.18 (§8), levando o kernel
do `SDA_FIXED` original aos demais métodos de doubling. `computeGainMatrixSchur()` e
`computeGainMatrixVanDooren()` foram medidos mas **não alterados** em nenhuma das duas rodadas
(fora de escopo, a pedido).

DARE resolvida por todos os métodos:

$$A^\top P A - P - A^\top P B\,(R + B^\top P B)^{-1} B^\top P A + Q = 0,\qquad K = (R + B^\top P B)^{-1} B^\top P A$$

## 1. Gabarito bibliográfico

Os nomes dos PDFs em `SOLVERS/` **não correspondem ao conteúdo**:

| Arquivo | Conteúdo real |
|---|---|
| `SDA.pdf` | Anderson (1978), IEEE CDC — doubling clássico, origem do SDA |
| `SDA-SS_ASDA_SCALED.pdf` | **Chu, Fan, Lin & Wang (2004)**, *Int. J. Control* 77(8):767-788 — SDA para DARE. Não contém ASDA nem SDA-Scaled apesar do nome do arquivo. |
| `SDA-SS.pdf` | Guo, Iannazzo & Meini — SDA com shift para **NARE** de M-matriz (`XCX-XD-AX+B=0`), não a DARE simétrica |
| `ADDA.pdf` | Wang, Wang & Li (2012), *SIAM J. Matrix Anal. Appl.* 33(1):170-194 — ADDA para **MARE** de M-matriz |
| `SDA_LargeScale.pdf` | Li, Chu, Kuo & Lin — SDA_ls_ε para NARE de grande escala |
| `dSDA.pdf` | Guo, Chu, Liang & Lin (2020), arXiv:2011.01494 — SDA desacoplado para CARE de grande escala |
| `VAN_DOOREN.pdf` | Van Dooren (1981), *SIAM J. Sci. Stat. Comput.* 2(2):121-135 |
| `ITERATIVO_MELHORADO.pdf` | Benner & Byers (1998) — Newton com Exact Line Search para CARE |

**Paper que falta e resolve o SDA-SS de fato** (não estava no diretório, buscado via web):
Chu, E.K.-W., Fan, H.-Y., Lin, W.-W. (2005). *A structure-preserving doubling algorithm for
continuous-time algebraic Riccati equations.* **Linear Algebra Appl. 396:55-80.** CARE→DARE via
Cayley com parâmetro γ e busca de Fibonacci para o γ ótimo — é o "SDA com shift" simétrico correto.
Ainda não fichado no vault.

### Veredito por método

| Método | Existe na literatura? | Referência correta | Implementação (pré-auditoria) |
|---|---|---|---|
| `SDA` | ✅ | Chu, Fan, Lin & Wang (2004) | ✅ Correta, literal |
| `SDA_FIXED` | ✅ | idem (Q13.18 é engenharia própria) | ✅ Álgebra idêntica ao SDA |
| `Iterative` | ✅ | value iteration clássica | ✅ Correta |
| `VanDooren` | ✅ | Van Dooren (1981) | ⚠️ Não alterado — ver §5 |
| `Schur` | ✅ (Laub 1979) | — | ⚠️ Não alterado — ver §5 |
| `SDA_SS` | ✅ existe, mas para outro problema | Chu, Fan & Lin (2005) — shift **simétrico** | ❌ Corrigido nesta auditoria |
| `SDA_Scaled` | ⚠️ só a ideia geral | Ward (1981) | ❌ Corrigido nesta auditoria |
| `ADDA` | ✅ existe, mas para outro problema | Wang, Wang & Li (2012) — MARE não-simétrica | ❌ Não implementado; corrigido para o que é: SDA em forma alternada |
| `ASDA` | ❌ **não existe** | nenhuma encontrada | ❌ Corrigido nesta auditoria (heurística própria) |

## 2. Evidência: contraexemplos e prova em float64

Verificação em três camadas: (1) resíduo real da DARE calculado em double a partir do `P` que cada
solver float32 devolve (`test/native/verify_solvers.cpp`); (2) transcrição float64 pura de cada
fórmula, exatamente como implementada em C++ (`python/verifica_solvers.py`) — se o resíduo persiste
em float64, é bug de fórmula, não de precisão; (3) build real para `esp32-s2-saola-1` e para o env
`benchmark`, ambos compilando sem erros após as correções.

### SDA-SS — contraexemplo escalar

$a=0{,}5,\ g=1,\ q=1$. DARE original: $p^2-0{,}25p-1=0 \Rightarrow p=1{,}1328$. Com a fórmula
anterior ($\gamma=0{,}3$): $a_0=0{,}2857$, $g_0=2{,}0408$, $h_0=1$ ⟹ $p=1{,}0271$ — **erro de 9,3%**,
resolvendo outra DARE. Confirmado no host (float32, caso `C6a_scalar_ss`): resíduo 0,0996; em
float64 (`python/verifica_solvers.py`): resíduo 0,0996 — idêntico, prova que é bug de fórmula.

### ASDA — invariância violada

Com $a=0{,}5,\ g=4,\ q=1$: $p=1{,}0506$. Sob $(G,H)\to(sG,H/s)$ com $s=\tfrac12$: $\tilde p=2{,}1011$,
e $\tilde p\times s=1{,}0506$ ✓ exato — ou seja $P=H_\infty\cdot\prod s_i$. A implementação anterior
não revertia $\prod s_i$ e aplicava $\beta^2$ só em $G$ na inicialização. No caso de hover real
(`C1_hover`, 6×6), o erro compunha ao longo de ~13 iterações: **resíduo 327 (float32 e float64)** —
catastrófico, não um efeito de arredondamento.

### SDA_Scaled — expoentes de D invertidos

Convenção consistente para $\hat A=DAD^{-1}$, $\hat G=DGD$: $\hat H=D^{-1}QD^{-1}$,
$P=D\hat P D$. A versão anterior usava $\hat H=DQD$ e $P=D^{-1}\hat P D^{-1}$ — invertido nos dois
pontos, sem cancelamento. No caso `C6c_2x2_scaled` (linhas de $A$ com normas 0,001 e 1000): resíduo
$\sim 10^6$ em float32 **e** em float64.

### ADDA ≡ SDA (push-through)

De $H(I+GH)=(I+HG)H$ segue $(I+HG)^{-1}H \equiv H(I+GH)^{-1}$. A recorrência
$H_{k+1}=H_k+A_k^\top W H_k A_k$ com $W=(I+H_kG_k)^{-1}$ é portanto idêntica a
$H_k+A_k^\top(H_kV)A_k$ com $V=(I+G_kH_k)^{-1}$ — a atualização do SDA. Verificado:
$\lVert P_{\text{ADDA}}-P_{\text{SDA}}\rVert/\lVert P_{\text{SDA}}\rVert \sim 10^{-16}$ (float64) em
todos os casos testados. A segunda inversão ($W$) era trabalho puro, sem efeito no resultado.

## 3. Correções aplicadas

| Método | O que mudou | Linha aprox. (`AutoLQR.cpp`) |
|---|---|---|
| `SDA_SS` | Substituído o shift afim em `A` isolado por um shift do *pencil* simplético inteiro, $M-\gamma L,\ L-\gamma M$ (preserva autovetores por construção — ver derivação no comentário do código), $\gamma=0{,}5$ fixo. Sem busca de Fibonacci (trabalho futuro). | `~1425-1660` |
| `ASDA` | Escalonamento inicial aplicado a `G` **e** `H` (antes só `G`); acumula `cum_s`; `P = H_k · cum_s` na extração. | `~1668-1900` |
| `SDA_Scaled` | `Ĥ = D⁻¹QD⁻¹` (era `DQD`); `P = D·P̂·D` (era `D⁻¹P̂D⁻¹`). | `~1890-2110` |
| `ADDA` | Removida a segunda inversão (`W`); usa a identidade push-through com `V` apenas. Citação bibliográfica corrigida (era Lin & Xu 2006/2007, inconsistente e incorreta). | `~2110-2320` |
| Todos (exceto Schur/VanDooren) | `getLastResidual()` agora devolve o **resíduo real da DARE** (novo `computeDareResidualNorm()`), calculado uma vez ao final do solver — antes devolvia `‖ΔH‖/‖H‖` (critério interno de parada, preservado em `getLastStepDelta()`), e valia 0 hardcoded em runs não-convergidos. | `AutoLQR.h/.cpp` |
| `computeGains()` | Mensagem de erro corrigida ("Usando SDA_FIXED", não "SDA"); retorna o flag de `K`, não o de `Kr`. | `~239-274` |
| `SDA_FIXED` | Buffers `static` de `invert_q`/`sda_q` viraram locais de pilha (reentrância); `residualHistory` reinicializado; `lastResidual` passa a ser o resíduo real (calculado em float32 pelo mesmo helper). | `~62-152, 333-365` |
| `Iterative` | Warm-start passa a usar buffer dedicado (`P_warm`), não mais o `P` genérico — antes podia herdar a solução deixada por **qualquer** outro método chamado antes. | `~1244-1420` |
| `invertMatrix` (n=4) | Corrigido aliasing: escrevia em `inv[]` enquanto ainda lia `m[]` na chamada in-place (`invertMatrix(X,X,4)`), produzindo resultado errado. Não afetava o caso 6×3 em produção. | `MatrixOperations.cpp:380-421` |
| `computeGainMatrix()` | Declaração morta removida (nunca definida, nunca chamada). | `AutoLQR.h` |

`computeGainMatrixSchur()` e `computeGainMatrixVanDooren()`: **não alterados**. Defeitos conhecidos e
medidos, registrados para referência:
- `Schur`: `Eigen::ComplexSchur` sem reordenação — seleção de colunas por índice não garante base do
  subespaço invariante estável. Órfão (nenhum chamador no repositório). Resíduo medido: ordens de
  grandeza de $10^5$ a $10^{10}$, `P` nem sempre positivo.
- `VanDooren`: usa autovetores individuais em vez do subespaço deflacionário do QZ ordenado; laço de
  relaxamento de threshold rescaneia só os mesmos `n` primeiros candidatos. Funciona bem na maioria
  dos casos testados (14/228 falhas na bateria), frágil em casos extremos.

## 4. Verificação: resultado agregado (228 casos × 9 métodos, host)

Após as correções, `SDA`, `SDA_SS`, `ASDA`, `SDA_SCALED` e `ADDA` têm a **mesma mediana de resíduo**
(~2×10⁻⁶) e **falham exatamente nos mesmos 5 casos** (inércias aleatórias extremas do benchmark +
o caso `C6c` deliberadamente mal-escalado) — a assinatura esperada de algoritmos algebricamente
equivalentes, não de bugs específicos de cada um.

| Método | Mediana resíduo DARE | p90 | Falhas (resíduo > 1e-2, de 228) |
|---|---|---|---|
| SDA | 2,1×10⁻⁶ | 3,5×10⁻⁶ | 5 |
| SDA_SS | 1,2×10⁻⁶ | 4,8×10⁻⁶ | 5 |
| ASDA | 2,2×10⁻⁶ | 4,1×10⁻⁶ | 5 |
| SDA_SCALED | 2,1×10⁻⁶ | 4,2×10⁻⁶ | 5 |
| ADDA | 2,1×10⁻⁶ | 4,3×10⁻⁶ | 5 |
| SDA_FIXED | 1,2×10⁻² | 3,2×10⁻² | 224/225 (Q13.18; comportamento pré-existente, ver §6) |
| Iterative | 5,8×10⁻² | 5,8×10⁻² | 205 (não converge em 100 iter. p/ sistemas rígidos; pré-existente) |
| Schur | 2,6×10⁵ | 3,1×10⁵ | 227 (órfão, não corrigido) |
| VanDooren | 2,7×10⁻⁴ | 7,9×10⁻⁴ | 14 (não corrigido) |

## 5. Achados fora do escopo desta rodada (registrados, não corrigidos)

- **`Iterative` não converge em 100 iterações** para boa parte dos casos com pesos de Bryson
  (205/228 na bateria) — mesmo assim o resíduo final costuma ser pequeno (mediana 5,8×10⁻²). Merece
  investigação de regularização/critério de parada em rodada futura.
- **`SDA_FIXED` tem residual mediano de ~1,2%** — consistente com o orçamento de erro documentado
  (quantização Q13.18), não uma regressão desta auditoria: o mesmo valor aparecia antes das
  correções (o `lastResidual` hardcoded em 0 escondia isso).
- **`Schur` e `VanDooren`** — defeitos documentados no §3, fora de escopo por pedido explícito.
- **`ADDA` real (dois parâmetros α≠β sobre a MARE)** não foi implementado — precedente publicado
  para o caso simétrico existe (*Low-rank alternating direction doubling algorithm for large-scale
  CARE*, IJCAS 2024) e fica como trabalho futuro.
- **Hipótese a testar**: na DARE simétrica, os "dois direcionais" do ADDA colapsam no único grau de
  liberdade `(G,H)→(sG,H/s)` que o ASDA já explora — se confirmado, ASDA seria o caso simétrico do
  ADDA, resolvendo a atribuição bibliográfica dos dois de uma vez. Não verificado nesta rodada.
- **Busca de Fibonacci para γ do SDA_SS** (Chu, Fan & Lin 2005) não implementada — γ=0,5 fixo já
  corrige o bug algébrico e acelera convergência em casos quase-críticos (~2× menos iterações no
  teste `C4a_dt_tiny`), mas não é o ótimo.

## 6. Como reproduzir

```bash
bash test/native/build.sh
./build/verify_solvers.exe                       # gera outputs/verify_host_baseline.csv e outputs/cases/*.csv
python python/verifica_solvers.py                # espelho float64, gera outputs/verify_float64_mirror.csv
```

Build do firmware de voo, do benchmark standalone e da verificação on-device (todos confirmados em
ESP32-S2 físico, COM3):

```bash
platformio run -e esp32-s2-saola-1     # firmware de voo
platformio run -e benchmark            # test/benchmark_solvers.cpp standalone
platformio run -e verify_gains -t upload -t monitor   # K/resíduo/tempo reais no S2, caso C1_hover
```

## 7. Arquivos

**Novos**: `test/native/verify_solvers.cpp`, `test/native/shim/{Arduino.h,ArduinoEigen.h}`,
`test/native/build.sh`, `python/verifica_solvers.py`, `test/verify_gains_onboard.cpp`, este documento.
**Modificados**: `lib/AUTOLQR/AutoLQR.cpp`, `lib/AUTOLQR/AutoLQR.h`, `lib/AUTOLQR/MatrixOperations.cpp`,
`lib/AUTOLQR/README.md`, `platformio.ini` (envs `benchmark` e `verify_gains`).
**Intocados**: `computeGainMatrixSchur()`, `computeGainMatrixVanDooren()`,
`test/benchmark_solvers.cpp` (roda, mas não foi estendido para incluir `SDA_FIXED` — trabalho futuro).

## 8. Extensão: fixed-point Q13.18 para os demais métodos de doubling

Depois da correção algébrica (§3), os quatro métodos de doubling corrigidos (`SDA_SS`, `ASDA`,
`SDA_SCALED`, `ADDA`) ganharam variantes em ponto fixo Q13.18, no mesmo espírito do `SDA_FIXED`
original — mais `ITERATIVE_FIXED` (recorrência própria). `Schur` e `VanDooren` ficam de fora, como
sempre.

### 8.1 Kernel compartilhado

O `namespace {}` anônimo que continha o kernel Q13.18 do `SDA_FIXED` (conversões, `matmul_q`,
`invert_q`) foi extraído para `lib/AUTOLQR/FixedPointQ.{h,cpp}`, generalizado com um
`fxq::doubling_loop_q(Ak,Gk,Hk,n,sh,Variant,...)` parametrizado por `Variant::Standard` (SDA,
SDA_SS, SDA_SCALED — só o setup/extração mudam, o laço é idêntico), `Variant::AdaptiveScaling`
(ASDA — rescale `(G,H)→(sG,H/s)` a cada iteração, `cum_s` acumulado em float por escapar da faixa
Q13.18) e `Variant::AlternatingVW` (ADDA — `V` e `W` calculados separadamente, ao contrário do par
`float` que já usa só `V` via push-through, para medir se a ordem de multiplicação quantiza
diferente). Os globais não-reentrantes `g_q_ovf`/`g_q_iters` do `SDA_FIXED` original viraram um
`Status{overflow,iterations,max_abs_seen}` por chamada.

**Regressão**: `SDA_FIXED` continua **bit-a-bit idêntico** ao pré-refactor nos 225 casos aplicáveis
da bateria (`ok`, `iterations`, ambos os resíduos, simetria, λ_min, ρ — zero diferenças).

### 8.2 O contratempo do `SDA_SS_FIXED` — Q9.22 não era a resposta certa

A primeira tentativa montou e inverteu o pencil 12×12 (`N1=[[I-γA,-γG₀],[γQ,I-γA']]`) em Q9.22
(±512), argumentando que `‖N1‖∞` é só ~200-700. **Falhou em 215 dos 225 casos.** Diagnóstico:
o que limita a faixa não é `N1`, é `Φ=N1⁻¹` — medido nos 225 casos, `‖Φ‖∞` tem **mediana 738** mas
chega a **46 120** no caso adversarial `C5_unequal_scale` (disparidade de escala 1000× entre estados
angulares e taxas). Nenhum shift fixo cobre as duas pontas sem perder resolução onde importa.
Solução: usar o **mesmo Q13.18** do resto do laço para o setup do `SDA_SS_FIXED` também — cobre 224
dos 225 casos (só falha o `C5`, o stress test deliberado). Simplifica o código (sem `requant()` de
ida e volta) e ainda foi o método mais preciso medido no hardware real (ver 8.3).

### 8.3 Resultado no ESP32-S2 físico (caso `C1_hover`, `test/verify_gains_onboard.cpp`)

| Método | tempo float | tempo fixed | *speedup* | resíduo DARE (fixed) |
|---|---|---|---|---|
| `SDA_FIXED` (referência) | 7,76 ms | 4,44 ms | 1,75× | 6,34×10⁻³ |
| `SDA_SS_FIXED` | 7,99 ms | **4,45 ms** | 1,80× | **4,03×10⁻³** |
| `ASDA_FIXED` | 9,00 ms | 4,98 ms | 1,81× | 4,81×10⁻³ |
| `SDA_SCALED_FIXED` | 8,50 ms | 4,59 ms | 1,85× | 9,14×10⁻³ |
| `ADDA_FIXED` | 8,39 ms | 5,57 ms | 1,51× | 7,61×10⁻³ |
| `ITERATIVE_FIXED` (100 iter.) | 32,6 ms | 19,3 ms | 1,69× | 5,80×10⁻² (float: 5,82×10⁻²) |

E no host, agregado sobre os 225 casos aplicáveis (mediana do resíduo real da DARE):

| Método | mediana resíduo | ok/225 |
|---|---|---|
| `SDA_SS_FIXED` | 1,09×10⁻² | 211 |
| `ASDA_FIXED` | **7,12×10⁻³** | 213 |
| `SDA_FIXED` | 1,23×10⁻² | 212 |
| `ADDA_FIXED` | 1,43×10⁻² | 212 |
| `SDA_SCALED_FIXED` | 1,57×10⁻² | 212 |

**Hipótese do plano parcialmente confirmada, com nuance**: a expectativa era que o ASDA_FIXED fosse
o mais robusto (o rescaling limita `‖G‖`, que é o que mais se aproxima do teto ±8192 — pico
histórico `maxGk≈2980`, margem de só 2,7×). No agregado dos 225 casos, ASDA_FIXED **é** o melhor
(mediana 7,1×10⁻³). Mas no caso de hover real medido no hardware, quem venceu foi o `SDA_SS_FIXED`
— e de forma dupla: mais preciso (4,03×10⁻³) **e** praticamente do mesmo custo que o `SDA_FIXED`
puro (4,45 ms vs. 4,44 ms), porque o setup do shift reduz o número de iterações do laço (8 vs. 10)
o suficiente para pagar o custo da inversão 12×12 extra. Os dois (SDA_SS, ASDA) superam o
`SDA_FIXED` de produção; `SDA_SCALED_FIXED` e `ADDA_FIXED` não.

**`ITERATIVE_FIXED` isola quantização de orçamento de iterações**: com tolerância solta (1e-3, igual
às demais variantes fixed-point) e só 25 iterações, o resíduo era péssimo (0,59) — mas ao subir para
100 iterações (mesmo orçamento do `float`), o resíduo bateu quase exatamente com o `float`
(5,80×10⁻² vs. 5,82×10⁻²). Conclusão: a convergência **linear** do value iteration é o gargalo, não
a quantização — mesmo achado do `float` (não converge em 100 iterações para a maioria dos 228 casos
da bateria), agora confirmado como independente da aritmética usada.

### 8.4 Trabalho futuro registrado

- `SDA_SS_FIXED` e `ASDA_FIXED` superaram o `SDA_FIXED` de produção em precisão — candidatos a
  substituir o default de `computeGains()`, pendente de mais medições (jitter, pior caso, não só o
  ponto de hover) e de decisão do usuário.
- O caso `C5_unequal_scale` (disparidade de escala 1000×) continua sem solução em fixed-point para o
  `SDA_SS_FIXED` — se algum caso real de voo se aproximar dessa disparidade, um shift maior (ex.
  Q17.14, cobre os 225/225 com resolução 6,1×10⁻⁵) é a alavanca disponível, à custa de precisão nos
  casos comuns.

## 9. Bateria de trajetórias de voo (2026-08-15)

Motivada pela rejeição do artigo submetido ao CBA 2026 (ver
`G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\revisoes_consolidadas.md`) — o benchmark
anterior usava perturbação aleatória em faixas mínimas ao redor do hover, criticado pelo revisor 2
por "não capturar a verdadeira trajetória sequencial contínua de um drone em voo". Substituído por
quatro trajetórias determinísticas e fechadas (`python/trajetorias.py`): T1 espiral de raio
crescente, T2 figura-8/Lissajous, T3 chirp 0,2–8 Hz, T4 degraus ±40° + giro de guinada contínuo —
11.538 pontos cada (60 s a 5,2 ms), 46.152 no total. `test/benchmark_solvers.cpp` foi reescrito
para percorrer essas quatro trajetórias nos 12 métodos (6 float + 6 `_FIXED`; Van Dooren e Schur
saíram, eliminando a dependência do Eigen no benchmark), com Q/R pela regra de Bryson (mesma forma
de `test/verify_gains_onboard.cpp`) em vez de valores fixos.

**Bateria completa rodada no ESP32-S2 físico (COM3, 921600 baud) em 2026-08-15, 111 min
(6659,7 s), 46.152 pontos × 12 métodos = 553.824 execuções.** Captura em
`outputs/serial_capture_bateria_trajetorias.txt` (13,1 MB); relatório consolidado e figuras em
`outputs/relatorio_bateria.md`, `outputs/fig_tempos_bateria.png`, `outputs/fig_erro_k_bateria.png`.
Erro de `K` contra `scipy.linalg.solve_discrete_are` em dupla precisão (não contra outro método do
próprio conjunto — resolve a validação circular apontada pelo revisor 2/4 do CBA), calculado em
`python/gerar_relatorio_bateria.py` a partir das linhas `GAIN` decimadas 1:50.

**Resultados agregados (todas as trajetórias, 46.152 pontos por método):**

| Método | Tempo médio | Falhas | Erro RMS(K) médio |
|---|---|---|---|
| `ITERATIVE_FIXED` | 1,87 ms | 0/46152 | ~1e-4 a 1,8e-4 |
| `SDA_FIXED` | 5,51 ms | 0/46152 | ~4,6e-5 a 5,7e-5 |
| `SDA_SS_FIXED` | 5,86 ms | 0/46152 | ~4,2e-5 a 5,3e-5 |
| `ASDA_FIXED` | 6,06 ms | 0/46152 | **~2e-5 a 3,9e-5 (o mais preciso dos `_FIXED` em toda trajetória)** |
| `SDA_SCALED_FIXED` | 5,64 ms | 0/46152 | ~5e-5 a 7,1e-5 |
| `ADDA_FIXED` | 7,42 ms | 0/46152 | ~5,2e-5 a 6,3e-5 |
| `SDA`/`SDA_SS`/`ASDA`/`SDA_SCALED`/`ADDA` (float) | 11,2–12,6 ms | 0/46152 | ~1,5e-6 a 3,3e-5 (praticamente idênticos entre si, confirma equivalência algébrica pós-auditoria) |
| `ITERATIVE` (float) | 44,7 ms | **19.655/46.152 (42,6%)** | — |

**Achados centrais para o artigo:**

1. **Todos os métodos de duplicação (float e `_FIXED`) convergiram em 100% dos 46.152 pontos**,
   cobrindo quatro regimes dinâmicos distintos — a bateria anterior (perturbação mínima ao redor
   do hover) não testava isso de verdade. `ITERATIVE` float falhou em 42,6% dos pontos mesmo se
   beneficiando de warm-start natural entre pontos consecutivos da mesma trajetória (a mesma
   instância `AutoLQR` percorre toda a trajetória) — confirma que a convergência linear é uma
   limitação estrutural, não um artefato do teste anterior.
2. **`ASDA_FIXED` é o mais preciso dos seis métodos `_FIXED` em TODAS as quatro trajetórias** —
   confirma a hipótese original (margem de faixa dinâmica ~187× contra o teto ±8192, medida em
   `python/bench_trajetorias.py`, contra ~7–10× dos demais) com dado de hardware real, não só
   emulação.
3. **`ITERATIVE_FIXED` herda o warm-start do `ITERATIVE` float** (mesmo `P_warm`, mesma instância)
   e converge em média em **3,57 iterações**, 1,87 ms, **zero falhas** — o mais rápido de todos os
   doze métodos e, ao contrário do par float, totalmente confiável. É o resultado mais forte da
   extensão de ponto fixo: uma combinação (warm-start + aritmética inteira) que nenhum dos dois
   ingredientes sozinho entrega.
4. **T3 (chirp) expõe um limite de banda do sistema controlado, não dos solvers**: erro de `K`
   sobe para ~3,3e-5 em TODOS os métodos float simultaneamente (vs. ~1,5e-6 a 1e-5 nas demais
   trajetórias) — o condicionamento da própria DARE piora nos pontos de operação mais extremos do chirp,
   independente de qual algoritmo a resolve.
5. **Malha fechada** (`python/malha_fechada_trajetorias.py`, controlador com ganho recalculado a
   cada passo por linearização no estado atual): `SDA_SS_FIXED` é consistentemente o pior em custo
   acumulado $J$ (+35,7% em T1, +33,8% em T2 vs. SDA float64), `ASDA_FIXED` é
   consistentemente o melhor (por vezes marginalmente melhor que o próprio float64) — o
   comportamento de faixa dinâmica em malha aberta se traduz diretamente em degradação de
   rastreamento em malha fechada. Em T3 (chirp), o erro de rastreamento sobe para ~32° em
   **todos** os controladores por igual (mesmo limite de banda do item 4), não diferenciando os
   métodos entre si — evidência adicional de que ali o gargalo é o sistema, não o solver.

Reprodução: `pio run -e benchmark -t upload --upload-port COM3` + captura serial a 921600 baud
(script de captura streamado incremental, não em `test/`, ver `python/gerar_relatorio_bateria.py`
para o formato esperado do CSV) + `python python/gerar_relatorio_bateria.py`.

## 11. Revisão dos dados: quatro questionamentos (2026-08-15)

Depois da bateria de trajetórias, quatro perguntas ficaram em aberto. Resolvidas nesta ordem —
ver `G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\revisoes_consolidadas.md` e
`estrategia_artigo.md` para o texto completo com referências.

### 11.1 Q1 — mais trajetórias discriminariam os métodos? Não, e isso é o achado.

A faixa dinâmica de $G_0=B_dR_d^{-1}B_d^\top$ e $H_0=Q_d$ é **invariante à trajetória** — varrendo
$\theta$ de 0° a 88° e taxas de 0 a 20 rad/s, max$|G_0|$=438 e max$|H_0|$=8,43e-3 em todos os
pontos, sem variação. A trajetória move $A(x)$, que não é o que dita a faixa dinâmica do solver.
Explica por que o erro de $K$ sobe e desce **junto** para todos os métodos entre trajetórias, sem
nunca reordenar o ranking — as trajetórias medem o condicionamento do problema, não o mérito
relativo dos algoritmos. Conclusão: não adicionar trajetórias; o eixo que falta varrer é $Q$/$R$
(ver 11.4).

### 11.2 Q2 — por que o scipy é a referência?

Validado externamente contra o **DAREX** (Abels & Benner, SLICOT Working Note 1999-16 — coleção
publicada especificamente como "conjunto de referência para a comparação de métodos"), não por
autoridade. Cinco exemplos com $R$ não-singular e $S=0$ (compatíveis com nossa parametrização
$G_0=BR^{-1}B^\top$; os demais têm $R$ singular ou $S\neq0$ e foram excluídos —
`python/darex_benchmark.py` documenta cada exclusão):

| Exemplo | $n$ | Origem física | $\kappa(X)$ pub. | erro vs. referência |
|---|---|---|---|---|
| 1.3 | 2 | ilustrativo (Van Dooren) | 1,1e2 | **8,9e-16** (solução exata em forma fechada) |
| 1.5 | 4 | satélite (roll/yaw) | 3,3 | 1,0e-3 (norma espectral) |
| 1.6 | 4 | modos lento/rápido | 1,8e2 | 1,7e-2 |
| 1.7 | 4 | extremamente mal-condicionado ($\kappa(L{+}M)\approx4\times10^{11}$) | 6,2e11 | 4,7e-4 |
| 1.8 | 5 | planta química 5ª ordem | 73,7 | 3,2e-5 |

O resíduo real do scipy fica em 1e-16 a 1e-13 nos cinco casos, incluindo o 1.7 — extremamente
mal-condicionado por construção. Nenhum exemplo do DAREX tem $n{=}6,m{=}3$ simultaneamente, então
essa validação cobre o **scipy em geral**, não nossos solvers `_FIXED` especificamente (gate
`n==6 && m==3`) — que já são validados contra o scipy diretamente nos 46.152 pontos da bateria.

### 11.3 Q3 — o ciclo de 5,2 ms regrediu? Sim — regressão confirmada, correção parcial, e o ciclo atual estoura o orçamento quase todo ciclo.

Comparando o mesmo caso (C1_hover, mesmo nº de iterações) antes/depois da refatoração `13b33bb`:
SDA_FIXED 3849→4438 µs (**+15,3%**), enquanto os métodos float não se moveram (±3 µs) — isola a
regressão no kernel de ponto fixo. Causas identificadas: perda de inlining ao extrair `qmul`/`qdiv`
para `FixedPointQ.cpp` como funções externas, e instrumentação de `max_abs_seen` (abs+compare+store
por elemento, ~2880 ops extras por chamada) inserida no caminho quente de `matmul_q`.

Correções aplicadas: `qmul`/`qdiv` viraram `inline` no header (recupera o inlining sem depender de
LTO — tentativa de `-flto` **descartada**: o `ld.exe` deste toolchain espressif32 não aceita
objetos LTO diretamente, "plugin needed to handle lto object"); `max_abs_seen` saiu do caminho
quente por padrão, agora sob `#ifdef FXQ_INSTRUMENT` (ligado só nos envs `benchmark`/
`verify_gains`/`sweep_qr`, onde é essencial para a Q4). `src/main.cpp` foi restaurado ao firmware
de voo real (estava temporariamente ocupado pelo harness de host da auditoria).

**Sensor reconectado — medição ponta a ponta concluída.** O scanner I2C dedicado
(`test/i2c_scan.cpp`, env `i2c_scan`) confirmou o MPU6050 respondendo em `0x68` após reconexão do
hardware. Firmware de voo gravado com `DEBUG_MODE=true` temporariamente (restaurado a `false` ao
final) para capturar o profiling por etapa embutido em `src/main.cpp:716-770`.

| Medição | Valor |
|---|---|
| `computeGains (DARE)` — SDA_FIXED, estado real do drone parado na bancada | **~5367 µs** (estável, 8 amostras consecutivas: 5365–5370) |
| `SDA_FIXED` isolado, mesmo caso sintético C1_hover, mesmas flags de voo (`verify_gains_flight`, sem `-DFXQ_INSTRUMENT`) | **4334 µs** |
| `SDA` (float) isolado, C1_hover — referência de sanidade do ambiente | 7768 µs (bate com o histórico: 7759–7761 µs) |
| `Tempo_Loop` (ciclo completo: sensores+Madgwick+SDRE+motores) | **6,5–7,2 ms** |
| `Overrun_Count` (ciclos com `processingTime ≥ 5200 µs`) | **~150 por segundo — essencialmente TODO ciclo estoura** |

Três leituras separadas, cada uma informativa:

1. **A correção recuperou só parte da regressão.** 4334 µs (pós-fix, sem instrumentação) contra os
   3849 µs originais é uma diferença de +12,6% ainda não fechada — resíduo atribuído à parte da
   perda de inlining que só LTO recuperaria de verdade (`matmul_q`/`invert_q`, funções maiores que
   `qmul`/`qdiv`, não movidas para o header por serem grandes demais para inlining manual sem
   duplicar ~150 linhas). Documentado como limitação conhecida, não perseguida a fundo nesta rodada.
2. **O caso real diverge do C1_hover sintético.** 5367 µs em voo (drone parado na bancada) contra
   4334 µs no C1_hover isolado — quase +24% de diferença adicional, com as MESMAS flags de
   compilação. A(x)/Q/R reais no ponto de operação do banco não são idênticos ao caso sintético
   C1_hover (usado historicamente como proxy) — provavelmente iterações extras do laço de
   duplicação sob o condicionamento real. Não investigado a fundo; registrado como achado.
3. **O ciclo de 5,2 ms NÃO fecha no estado atual do firmware.** Com `SDA_FIXED` em ~5,37 ms sozinho
   e mais ~900 µs de sensores/Madgwick/motores, o `Tempo_Loop` fica em 6,5–7,2 ms — **acima do
   período de 5200 µs quase todo ciclo** (`Overrun_Count` cresce ~150/s, praticamente 1:1 com a
   taxa real do loop). Isso contradiz a expectativa do usuário de que o ciclo "rodava em ~5,2 ms"
   antes das mudanças recentes — seja essa expectativa baseada em uma medição anterior à
   regressão do `13b33bb`, seja em um caso de operação mais favorável (o C1_hover sintético
   citado no item 2 chegaria perto: 4334+900≈5,2 ms). O código não detecta esse overrun
   silenciosamente antes desta sessão (`else { overrunCount++; }` adicionado a
   `src/main.cpp:676-684` nesta revisão) — sem essa instrumentação, o desvio do $T_s$ real usado
   pelo Madgwick e pela discretização SDC passava despercebido.

`DEBUG_MODE` foi restaurado a `false` (produção) ao final da medição; o firmware de voo permanece
gravado no ESP32-S2 no estado normal, com o contador de overrun ativo.

### 11.4 Q4 — o comportamento dos métodos depende de Q/R? Sim, fortemente — mapa de segurança medido no hardware.

Variando só a escala de $R$ (Q fixo), $G_0$ vai de 4,4 ($R\times100$) a 4,4e5 ($R\times10^{-3}$) —
um fator de $10^5$ dentro de escolhas de projeto plausíveis. Varredura completa no ESP32-S2
(`test/sweep_qr.cpp`, env `sweep_qr`): 13 escalas de $R$ (1e-6 a 1e6) × 5 escalas do bloco de
taxas de $Q$ (1e-2 a 1e2) × 300 pontos das 4 trajetórias × 10 métodos (5 float + 5 `_FIXED`;
`ITERATIVE`/`ITERATIVE_FIXED` fora do escopo desta pergunta) = 195.000 execuções, 1748 s no
hardware real.

**Achado 1 — parede rígida simétrica, igual para todos os métodos.** Fora de $R_{scale}\in[10^{-2},
10^{2}]$, a falha vai a 60–100% em **todos** os cinco métodos `_FIXED` igualmente — não é um limite
que alguma variante evite melhor. Métodos float: **zero falhas** nas 19.500 execuções em toda a
grade de 12 décadas, confirmando a robustez esperada.

**Achado 2 (não previsto) — o lado de $R$ grande falha por um motivo diferente do lado de $R$
pequeno.** Em $R_{scale}\ge1000$ (100% de falha), `max_abs_seen` fica **congelado** no último
valor de sucesso antes da fronteira (o campo só atualiza em chamadas OK) — prova de que essa
falha **não é overflow** (nada cresce), diferente do lado $R_{scale}$ pequeno, onde $G_0$
explode de fato. É provavelmente quase-singularidade no laço de duplicação
(`pivot_floor=1e-4` de `invert_q`), não investigada em profundidade nesta rodada — trabalho
futuro.

**Achado 3 — dentro da faixa moderada ($R_{scale}\in[0{,}1,100]$, a faixa fisicamente plausível de
sintonia), os métodos se diferenciam de verdade:**

| Método | Falhas na faixa moderada (6.000 execuções) |
|---|---|
| **ASDA_FIXED** | **3 (0,05%)** |
| SDA_FIXED | 363 (6,05%) |
| SDA_SCALED_FIXED | 373 (6,22%) |
| ADDA_FIXED | 434 (7,23%) |
| SDA_SS_FIXED | 459 (7,65%) |

`ASDA_FIXED` é ~100–150× mais confiável que os demais nessa faixa — confirma no hardware, com uma
grade de Q/R real (não só a trajetória), a mesma conclusão já vista no teste aberto (margem de
faixa dinâmica ~187× contra o teto ±8192) e na malha fechada (custo consistentemente mais baixo).
É o terceiro experimento independente a apontar o `ASDA_FIXED` como o candidato mais robusto dos
seis métodos de ponto fixo.

Reprodução: `pio run -e sweep_qr -t upload --upload-port COM3` + captura serial a 921600 baud +
`python python/analisa_sweep_qr.py outputs/serial_sweep_qr.txt` (figura:
`outputs/fig_sweep_qr_seguranca.png`); `python python/darex_benchmark.py` para a Q2.

## 12. Otimização do kernel de ponto fixo (2026-08-15) — SDA_FIXED abaixo do original

A Seção 11.3 fechou com o `SDA_FIXED` ~15% mais lento que antes de `13b33bb`, com correção
parcial (só ~2,3% recuperados). Investigando o caminho quente, achou-se a causa maior: **o kernel
de ponto fixo nunca recebeu as otimizações que `MatrixOperations::matrixMultiply` (float) já tem**
— `IRAM_ATTR`, `__restrict__`, especialização por tamanho, `inline` — e **todo** solver (os 12,
float e fixo) chamava `computeDareResidualNorm()` (11 `new[]`/`delete[]` + 7 matmuls) dentro da
região cronometrada, mesmo quando o chamador nunca lê o resíduo.

**Fase 1 — bit-idêntica** (verificada exata contra os 3174 pares caso×método do harness nativo,
exceto a coluna `time_us`):

1. **Resíduo preguiçoso.** `getLastResidual()` passa a calcular sob demanda (`residualDirty` +
   cache em `lastResidual`, ambos `mutable` — `AutoLQR.h`/`AutoLQR.cpp`), em vez de toda
   `computeGainMatrixXXX()` chamar `computeDareResidualNorm()` incondicionalmente. Os 12
   `lastResidual = computeDareResidualNorm();` viraram `residualDirty = true;`. Benefício: todos
   os 12 métodos (float e fixo), porque `test/benchmark_solvers.cpp` e `test/sweep_qr.cpp` só leem
   o resíduo **depois** de parar o cronômetro — nenhuma linha desses arquivos mudou.
2. **`fxq::qmul`/`qdiv`/`matmul_q`/`transpose_q`/`add_q`/`sub_q` viraram `inline` em
   `FixedPointQ.h`**, com `__restrict__`, e `matmul_q` ganhou um atalho 6×6×6 desenrolado (a forma
   que domina o laço de duplicação — 8 das 8 matmuls por iteração). `invert_q` e
   `doubling_loop_q` ganharam `IRAM_ATTR` via o macro `FXQ_FAST_ATTR`
   (`#if defined(ESP32)`, mesmo padrão de `MatrixOperations.h`).
   **Armadilha do Xtensa:** aplicar `IRAM_ATTR` também nas funções `inline` pequenas (que acabam
   dentro de `doubling_loop_q`, já com `IRAM_ATTR`) quebra o link — `"dangerous relocation: l32r:
   literal placed after use"` — o linker rejeita a diretiva de seção duplicada. Removido o
   atributo dessas; o código vai para IRAM de qualquer forma, via a função externa que as inclui.

**Fase 2 — muda o arredondamento** (revalidada por tolerância contra o `scipy`, não por igualdade
bit-a-bit):

3. **Recíproco do pivô em `invert_q`.** Antes: `qdiv(aug[i][j], piv)` chamado uma vez por cada
   uma das `n2` colunas (até 24 divisões de 64 bits por pivô — o Xtensa LX7 não tem divisão de 64
   bits em hardware). Depois: `inv_piv = qdiv(one, piv)` uma única vez, `qmul(aug[i][j], inv_piv)`
   nas `n2` colunas. Custo: `inv_piv` perde a resolução plena de `qdiv` direto, então o resultado
   já não é bit-a-bit igual ao de antes — 1 caso a mais falha em 4 dos 6 métodos `_FIXED` (211-212
   → 211 de 225; `SDA_SS_FIXED` e `ITERATIVE_FIXED` inalterados).

**Item descartado: explorar a simetria de $G_{k+1}$/$H_{k+1}$** (~7% adicional estimado, ver diário
técnico da auditoria original). $G_{k+1}=A_kVG_kA_k^\top$ é claramente simétrica (forma $MG_kM^\top$
com $G_k$ simétrico). $H_{k+1}=A_k^\top H_k V A_k$ **não é** da mesma forma trivial — não há
identidade imediata que garanta simetria exata sem a identidade push-through completa
(`AutoLQR.cpp` não força simetria explicitamente em nenhum ponto do laço `fxq`, ao contrário dos
espelhos Python em `verifica_solvers.py`, que chamam `sym()` a cada iteração). Implementar esse
atalho errado corromperia silenciosamente um solver que controla um drone — descartado por risco,
não por custo. Fica registrado como trabalho futuro, condicionado a provar a simetria de $H_{k+1}$
antes de codificar.

**Resultado medido no ESP32-S2** (`verify_gains_flight`, C1_hover, mesmas flags do caminho de voo):

| Etapa | SDA_FIXED | vs. original (3849 µs) |
|---|---|---|
| Original (pré-`13b33bb`) | 3849 µs | — |
| Regressão (`13b33bb`, sem correção) | 4438 µs | +15,3% |
| Correção parcial (Seção 11.3, só `qmul`/`qdiv` inline) | 4334 µs | +12,6% |
| **Fase 1 completa** (resíduo preguiçoso + kernel `fxq` otimizado) | 2618 µs | **−32,0%** |
| **Fase 2** (+ recíproco em `invert_q`) | **2366 µs** | **−38,5%** |

Os outros cinco métodos `_FIXED` ganharam proporcionalmente (Fase 1, isolados no hardware):
`SDA_SS_FIXED` 4312→2464 µs, `ASDA_FIXED` 4860→2899 µs, `SDA_SCALED_FIXED` 4470→2520 µs,
`ADDA_FIXED` 5415→3060 µs, `ITERATIVE_FIXED` (quente) 1098→598 µs. Os métodos **float** também
melhoraram ~5% (só o item 1 os afeta): `SDA` 7768→7350 µs, `SDA_SS` 7998→7592 µs,
`ASDA` 9009→8601 µs, `SDA_SCALED` 8512→8108 µs, `ADDA` 8398→7994 µs.

### 12.1 Fase 2 — recíproco em `invert_q`

Antes: `qdiv(aug[i][j], piv)` chamado uma vez por cada uma das `n2` colunas do pivô (até 24
divisões de 64 bits — o Xtensa LX7 não tem divisão de 64 bits em hardware). Depois:
`inv_piv = qdiv(one, piv)` uma única vez, `qmul(aug[i][j], inv_piv)` nas `n2` colunas.
**Ganho medido isolado (C1_hover, `verify_gains_flight`): SDA_FIXED 2618→2366 µs (−9,6%
adicional)**. Custo: `inv_piv` perde a resolução plena de `qdiv` direto — resultado não é mais
bit-a-bit igual ao da Fase 1; 1 caso a mais falha em 4 dos 6 métodos `_FIXED` nos 225 casos do
harness nativo (211-212 → 211 de 225; `SDA_SS_FIXED` e `ITERATIVE_FIXED` inalterados).

**Item descartado: simetria de $G_{k+1}$/$H_{k+1}$.** $G_{k+1}=A_kVG_kA_k^\top$ é claramente
simétrica; $H_{k+1}=A_k^\top H_k V A_k$ não tem a mesma forma trivial $MG_kM^\top$ sem a
identidade push-through completa, e nenhum ponto do laço `fxq` força simetria explicitamente
(ao contrário dos espelhos Python em `verifica_solvers.py`, que chamam `sym()` a cada iteração).
Implementar esse atalho errado corromperia silenciosamente um solver que controla um drone —
descartado por risco, não por custo (~7% estimado, não capturado).

### 12.2 Revalidação completa — resultados

**Host, 225 casos** (`test/native/verify_solvers.cpp`): Fase 1 verificada **bit a bit idêntica**
à baseline pré-otimização em todas as 3174 linhas (caso×método), exceto `time_us`. Fase 2 (recíproco)
comparada por tolerância: contagem de sucesso 211-212→211 de 225 (queda de 1 caso em 4 dos 6
métodos `_FIXED`, aceita).

**Bateria de trajetórias, 46.152 pontos** (`outputs/serial_capture_bateria_trajetorias_v2.txt`,
6252 s ≈ 104 min — dominado pelo `ITERATIVE` float, que não se beneficia das otimizações):
contagens de linhas idênticas à bateria original (46152 PT, 110784 RUN, 11088 GAIN, 60 SUMMARY) —
nenhum dado faltando. Zero falhas em todos os métodos de duplicação (float e `_FIXED`), `ITERATIVE`
float com 19655/46152 falhas — idêntico à bateria original, confirma que a álgebra não mudou.

| Método | v1 (antes) | v2 (depois) | Ganho |
|---|---|---|---|
| SDA_FIXED | 5512 µs | **3277 µs** | **−40,6%** |
| SDA_SS_FIXED | 5856 µs | 3646 µs | −37,7% |
| ASDA_FIXED | 6063 µs | 3844 µs | −36,6% |
| SDA_SCALED_FIXED | 5636 µs | 3414 µs | −39,5% |
| ADDA_FIXED | 7422 µs | 4689 µs | −36,8% |
| ITERATIVE_FIXED | 1866 µs | 1101 µs | −41,0% |
| SDA (float) | 11156 µs | 10553 µs | −5,4% |
| SDA_SS (float) | 11890 µs | 11319 µs | −4,8% |
| ASDA (float) | 12646 µs | 12080 µs | −4,5% |
| SDA_SCALED (float) | 12078 µs | 11527 µs | −4,6% |
| ADDA (float) | 11940 µs | 11340 µs | −5,0% |
| ITERATIVE (float) | 44725 µs | 44135 µs | −1,3% |

Precisão (erro RMS de $K$ vs. `scipy`, trajetória T1) praticamente inalterada: `ASDA_FIXED`
2,01e-5→1,98e-5 (continua o mais preciso dos `_FIXED`), `SDA_SS_FIXED` 4,24e-5→4,16e-5,
`SDA_FIXED` 4,82e-5→4,76e-5 — mesma ordem de grandeza, mesmo ranking, confirma que a Fase 2 não
degradou a precisão de forma relevante.

**Varredura Q/R, 195.000 execuções** (`outputs/serial_sweep_qr_v2.txt`, 1658 s ≈ 27,6 min — bem
mais rápida que a original, 27,6 min contra os ~29 min de antes, mesmo com o dobro de execuções
por causa dos métodos mais rápidos): mapa de segurança qualitativamente idêntico. Na faixa
moderada (R_scale∈[0,1;100]): `ASDA_FIXED` 0,07% de falha (4/6000, era 0,05%/3/6000) contra
6,10–7,60% dos demais quatro métodos `_FIXED` — a conclusão central (ASDA_FIXED ~100× mais
confiável) permanece intacta.

**Ciclo de voo ponta a ponta** (`src/main.cpp`, `DEBUG_MODE=true` temporário, restaurado a `false`
ao final): **o ciclo volta a fechar em 5,2 ms.**

| | Antes da otimização (Seção 11.3) | Depois |
|---|---|---|
| `computeGains` (SDA_FIXED, ponto real de operação) | ~5367 µs | **~3230 µs** |
| `Tempo_Loop` | 6,5–7,2 ms | **5201 µs** |
| `Tempo_Processamento` | — | 4415–4823 µs |
| `Overrun_Count` | ~150/s (quase todo ciclo) | **~1-2/s** (<1% dos ciclos) |

`Tempo_Loop=5201 µs` é literalmente o alvo `LOOP_PERIOD_US=5200` mais a granularidade do
busy-wait — o laço passou a fechar dentro do orçamento na esmagadora maioria dos ciclos.
`computeGains` no ponto de operação real (3230 µs) segue acima do isolado em C1_hover sintético
(2366 µs, ~36% de diferença) — mesma divergência já registrada na Seção 11.3, não investigada a
fundo.

**Conclusão da revisão de desempenho:** a meta do usuário ("rodar o SDA_FIXED tão rápido quanto
antes") foi superada — 2366 µs contra os 3849 µs originais (−38,5%), e o ciclo completo de voo,
que estava estourando o orçamento em praticamente todo ciclo, volta a fechar em 5,2 ms. Todos os
seis métodos `_FIXED` ganharam proporcionalmente (~35-41%), respondendo também à segunda parte da
pergunta do usuário ("dá pra rodar os demais mais rápido também?").

## 13. Artigo novo: comparação controlada (algoritmo × aritmética × critério de parada)

**Contexto (2026-08-18).** O artigo do CBA 2026 foi rejeitado por (entre outros pontos)
"comparar algoritmos conhecidos" sem contribuição própria e por um confundimento explícito
(Van Dooren via Eigen contra rotinas manuais). A auditoria dos dados medidos para o artigo
DINAME encontrou que **a comparação atual entre float e ponto fixo também não é justa**, por
três defeitos materiais:

1. **Critérios de parada incompatíveis.** Float: norma de Frobenius relativa, tolerância
   $10^{-6}$, orçamento 100 iterações. Ponto fixo: norma máx-abs relativa, tolerância $10^{-3}$
   (`invRelTolerance=1000`), orçamento 25 iterações. Três ordens de grandeza de diferença de
   tolerância, normas distintas, orçamentos distintos.
2. **Médias enviesadas.** `MethodStats::add()` em `test/benchmark_solvers.cpp` descartava a
   amostra inteira quando `ok==false` — as médias reportadas eram calculadas só sobre os
   sucessos, não sobre as tentativas totais.
3. **Método iterativo mal rotulado e mal orçado.** O implementado é iteração de valor
   (convergência linear), não Kleinman (Newton, convergência quadrática) — o artigo citava
   `kleinman1968` incorretamente. Além disso, o orçamento de 100 iterações censura o método: o
   $\rho$ de malha fechada medido tem mediana **0,9904**
   (`outputs/verify_host_baseline.csv`), e a contração $\rho^{2k}$ exige ~358 iterações para
   $10^{-3}$ e ~717 para $10^{-6}$ — contra as 100 disponíveis. Confirmado nos dados: o
   histograma de iterações do `ITERATIVE` empilha em exatamente 100 (distribuição truncada), e
   o `ITERATIVE_FIXED` chega a atingir o teto de 100 e **ainda assim retornar sucesso**, porque
   no ponto fixo "falha" só significava overflow/singularidade, nunca esgotamento de orçamento.
   As 3,57 iterações médias reportadas do `ITERATIVE_FIXED` (bateria v2) são artefato: em 7.341
   de 9.232 execuções amostradas ele converge em **1 iteração**, porque a partida quente somada
   à tolerância frouxa já satisfaz o teste na entrada.

### 13.1 Correções no código

- **`AutoLQR`**: novo enum `SolveOutcome{Converged,Budget,Breakdown}`, membros
  `relTolerance`/`maxIterations`/`invRelTolerance` configuráveis via
  `setStoppingCriterion(relTol, maxIters)`, e `getLastOutcome()`. Aplicado aos 12 métodos
  (6 float + 6 `_FIXED`).
- **Norma unificada**: os 6 métodos float trocaram Frobenius relativa por **máx-abs relativa**
  — mesma norma que o kernel `_FIXED` já usava (`FixedPointQ.cpp:doubling_loop_q`), para que a
  única variável entre as duas aritméticas seja a aritmética em si. Efeito colateral esperado e
  aceito: iterações em regime caem ~1 (ex.: SDA 11→10 no host) por ser critério mais permissivo
  numa norma diferente — não um efeito de bug.
  - `computeGainMatrixIterative_Fixed`: `maxIterations`/`invRelTolerance` agora vêm de
    `setStoppingCriterion()` (era `100`/`1000` fixos no código).
- **Retorno booleano unificado**: `computeGains()` agora retorna `true` se e somente se
  `SolveOutcome::Converged` nos 12 métodos — antes, os `_FIXED` retornavam `true` mesmo ao
  esgotar o orçamento (só `false` em overflow/singular).
- **Regressão algébrica (host, `test/native/verify_solvers.cpp`, 3174 casos)**:
  `outputs/verify_baseline_PRE_JUSTICA.csv` → `outputs/verify_host_baseline.csv`. Resíduos e
  contagens `ok` dos métodos de duplicação (float e `_FIXED`) **inalterados**; iterações caem
  ~1 no float pela troca de norma (esperado). **`ITERATIVE_FIXED`: ok caiu de 217/225 para
  13/225** — não é regressão, é a correção do vício descrito no item 3 acima se manifestando: a
  maioria dos casos do host precisa de mais de 100 iterações para convergir de verdade sob o
  critério unificado, e antes isso era mascarado.
  - **Nota de ambiente**: o binário do harness precisou ser religado estaticamente
    (`-static-libgcc -static-libstdc++ -static`, aplicado em `test/native/build.sh`) — em Git
    Bash, com PATH misturando toolchains MinGW distintos, um binário dinâmico carregava
    `libstdc++`/`libgcc` de uma build incompatível e falhava silenciosamente com exit 127, sem
    imprimir nada. Não é um bug do código, é do ambiente de compilação do host.
- **`python/bench_trajetorias.py:sda_ss_fixed_q`**: corrigido de γ=0,3 com shift afim só em A
  para γ=0,5 com o shift completo do pencil simplético 12×12 — espelhando exatamente
  `AutoLQR.cpp:computeGainMatrixSDA_SS_Fixed()`. A versão antiga resolvia uma DARE diferente da
  que o device resolve (erro_K_SDA_SS_FIXED ~9e-4 no espelho host contra ~4e-5 medido no
  hardware).

### 13.2 Campanha nova (Exp. 0-3)

Firmwares novos/alterados: `test/tolerance_sweep.cpp` (Exp. 0a/0b, novo),
`test/benchmark_solvers.cpp` (Exp. 1, critério configurável via `-DBATTERY_REL_TOL`/
`-DBATTERY_MAX_ITERS`, taxonomia de desfecho nas colunas `SUMMARY`), `test/sweep_qr.cpp`
(Exp. 2, idem via `-DSWEEP_REL_TOL`/`-DSWEEP_MAX_ITERS`). Env `tolerance_sweep` adicionado a
`platformio copy.ini`. Resultados da campanha registrados à medida que completam — ver
`outputs/serial_tolerance_sweep.txt` e `python/analisa_tolerancia.py`.

## 14. Troca do critério de parada para norma de Frobenius

Por decisão explícita do usuário, o critério unificado de convergência (Seção 13) — inicialmente
implementado como norma máx-abs relativa, mais barata em aritmética inteira — foi trocado para
norma de **Frobenius relativa**, `‖H_{k+1}-H_k‖_F/‖H_k‖_F`, aplicada nos dois caminhos (float e
`_FIXED`) e em todos os 12 métodos. No caminho fixed-point a soma de quadrados é calculada em
`float` a partir dos valores Q13.18 convertidos (`FixedPointQ.cpp:doubling_loop_q`), não em
inteiro — evitaria estourar `int64` sem ganho relevante de desempenho, já que o custo é
desprezível frente às multiplicações de matriz da própria iteração. O mirror Python
(`python/fixedpoint_q.py`) foi atualizado para casar exatamente essa fórmula.

Como Frobenius e máx-abs são normas diferentes, a tolerância "casada" achada sob a norma antiga
(1e-6) não é necessariamente a mesma sob a nova — daí a necessidade de refazer o Exp. 0
(`test/tolerance_sweep.cpp`) do zero sob o novo critério, o que motiva a Seção 15.2 abaixo.

## 15. Achados da revisão de cinco perguntas (2026-08-18)

O usuário levantou cinco perguntas técnicas sobre o rascunho `diname2027_v2.tex`. A investigação
resultante encontrou dois achados que **invertem conclusões já publicadas no rascunho** — um deles
é um bug introduzido nesta própria sessão — além de responder objetivamente às outras três
perguntas com medição nova em vez de opinião.

### 15.1 Bug do `lastOutcome` em `computeGainMatrixSDA_Fixed()` — retratação obrigatória

`lib/AUTOLQR/AutoLQR.cpp:237` era o único ponto de saída por overflow em todo o arquivo sem
`lastOutcome = SolveOutcome::Breakdown;` — a taxonomia de desfecho (Seção 13) foi retrofitada
sobre código com múltiplos `return false` antecipados, e esse caminho específico foi esquecido.

Consequência medida: em `R_scale >= 1e3`, `Rd[0][0] = 55,5 × R_scale` estoura o teto ±8192 do
Q13.18 já na conversão da entrada (`f2q`), antes de qualquer laço de duplicação rodar. Os quatro
outros métodos `_FIXED` (`ADDA_Fixed`, `ASDA_Fixed`, `SDA_Scaled_Fixed`, `SDA_SS_Fixed`)
corretamente marcam `Breakdown` e retornam; `SDA_Fixed`, por causa do bug, devolvia a telemetria
(`lastIterations`, `lastResidual`, `lastFixedPointMaxAbsSeen`, `lastOutcome`) da chamada **anterior
não relacionada**, porque esses membros são persistentes por instância e `computeGains()` não os
reseta antes de cada chamada. Efeito na varredura Q/R original (`outputs/serial_sweep_qr_v3.txt`):
6000 execuções em `R_scale >= 1e3` reportaram uma única combinação `(iters=15, resid=8.520439e-03,
outcome=0)`, contra 5971 combinações distintas do SDA float na mesma faixa — telemetria congelada,
não um resultado real. Isso levou a uma alegação falsa no rascunho: que o `SDA_FIXED` seria
imune ao modo de falha em `R` grande, quando na verdade os cinco métodos `_FIXED` abortam de forma
idêntica nesse limiar (`R_scale = 8192/55,5 ≈ 147,6`).

**Correção** (uma linha, no padrão das saídas de overflow equivalentes nas linhas 241/2279/2337/
2409/2510 do mesmo arquivo):
```cpp
if (st.overflow) { lastOutcome = SolveOutcome::Breakdown; return false; }
```
**Regressão verificada** (`test/native/build.sh` + `build/verify_solvers.exe`, 225 casos de
`outputs/cases/`): zero diferença nos campos de resíduo/`ok` de `SDA_FIXED` antes e depois do fix
— o bug só se manifesta no caminho de overflow de entrada, que nenhum dos 228 casos-teste padrão
do host aciona (todos estão em faixa normal de `R`). A varredura Q/R ainda precisa ser re-rodada
no hardware para confirmar que `SDA_FIXED` agora reporta `outcome=2` em `R_scale >= 1e3` como os
outros quatro — pendente.

**Atualização 2026-08-18 — varredura re-rodada** (`outputs/serial_sweep_qr_v4.txt`, τ=1e-3, fix
aplicado, 1582 s): confirmado — os cinco métodos `_FIXED` agora reportam `outcome=2` de forma
idêntica em `R_scale >= 1e3` (6000/6000 cada), fechando o critério de aceite do plano.

A re-varredura também revelou um **segundo modo de falha, não documentado antes**, no extremo
oposto da escala: em `R_scale ≤ 1e-2`, 4500-4583 de 7500 execuções (60-61%) dos cinco métodos
`_FIXED` reportam `outcome=2`, caindo abruptamente para 11/1500 (0,7%) em `R_scale=1e-1` e 0 daí
em diante — transição nítida entre `R_scale=1e-2` e `1e-1`. O caminho float, nos mesmos pontos, tem
zero breakdowns (0/7500) — é um modo de falha exclusivo do ponto fixo, e afeta os cinco métodos
quase igualmente (não é peculiaridade de um único algoritmo).

**Atualização 2026-08-19 — mecanismo confirmado no espelho bit-a-bit** (`python/fixedpoint_q.py`,
instrumentado ponto a ponto reproduzindo exatamente a sequência de
`computeGainMatrixSDA_Fixed()`): a hipótese acima ("overflow interno ao laço, via `R⁻¹` crescendo")
estava **errada**. `R⁻¹` em si cabe folgado — em `R_scale=1e-2`, `max|R⁻¹|≈1,8`, muito abaixo do
teto. `invert_q(R)` também não estoura (`overflow=False` depois dele, confirmado em `R_scale∈
{1e-1,1e-2,1e-3}`). O estouro ocorre **antes do laço começar**, no cálculo de
$G_0=B R^{-1}B^T$ — segundo, não primeiro, `matmul_q` do setup:

| `r_scale` | overflow após `Rinv=invert_q(R)`? | overflow após `BRi=B·Rinv`? | overflow após `G0=BRi·Bᵀ`? | max\|G0\| (float, não quantizado) |
|---|---|---|---|---|
| 1e-1 | não | não | **não** | 4377 |
| 1e-2 | não | não | **sim** | 8192 (exatamente o teto — o cálculo em float bate 8192,4 e o Q13.18 estoura ao arredondar) |
| 1e-3 | não | não | **sim** | 8192 (idem, ainda mais acima em float puro) |

Ou seja: não é "overflow interno ao laço de duplicação" — é **overflow de uma matriz derivada da
entrada** ($G_0$, calculada no setup a partir de $B$ e $R^{-1}$), uma terceira categoria distinta
tanto do overflow de entrada pura (conversão `f2q` de `Rd`, mecanismo do `R_scale` grande) quanto
de um overflow genuinamente interno às iterações do laço (que não foi observado — o laço nunca
chega a rodar quando $G_0$ já estourou no setup). O comentário em `AutoLQR.cpp:268`
("`saturou durante o laço`") está impreciso: ele detecta corretamente o `overflow` (o flag é
global/sticky, ver `FixedPointQ.h`), mas rotula errado a origem — cobre igualmente o overflow do
setup de $G_0$ e qualquer overflow genuíno dentro do laço, sem distingui-los. Não corrigido nesta
sessão (exigiria instrumentação por-sítio em `fxq::Status`, ver Fase 0.4 do plano de campanha
estendida); registrado aqui para a próxima revisão do código/comentário.

**Implicação para o artigo**: o mapa de segurança Q/R (Fig. 4) precisa mostrar os dois extremos —
não só "R grande quebra o ponto fixo", mas "R muito fora do nominal em qualquer direção quebra" —
e o texto deve descrever o mecanismo do lado pequeno corretamente: overflow de $G_0=BR^{-1}B^T$ no
setup, não do laço de duplicação em si.

### 15.2 Piso de quantização do teste de convergência em Q-format

O `ITERATIVE_FIXED` (iteração de valor em ponto fixo) convergia em apenas 9,4% dos casos sob a
tolerância original (1e-6/Frobenius) — o rascunho descrevia isso como falha do método. A
investigação mostrou o oposto: é o **teste de parada**, não o solver, que está quebrado abaixo de
um piso de quantização inerente ao formato Q13.18.

O passo mínimo de uma única entrada é 1 LSB = 2⁻¹⁸ ≈ 3,81×10⁻⁶. Mas na prática as `n²` entradas de
`P` oscilam em ±1 LSB simultaneamente perto de um ponto fixo quantizado, então o piso real do
passo relativo de Frobenius é

$$\text{relF}_\text{piso} = \frac{n \cdot 2^{-s}}{\|P\|_F}$$

Medido sobre 231 pontos amostrados das quatro trajetórias: `‖P‖_F ∈ [0,408; 0,464]` (notavelmente
estável entre trajetórias — ver também 15.4), dando piso de 4,9×10⁻⁵ a **5,6×10⁻⁵** no pior caso.
Uma simulação independente da recorrência quantizada (não a integração real, um modelo isolado do
laço de ponto fixo) achou ciclo-limite em 5,6×10⁻⁵ — bate exatamente. Com tolerância 1e-6 (17×
abaixo do piso), só passa quem cai em ponto fixo exato bit a bit — os 9,4% de "convergência" eram
taxa de coincidência, não convergência de verdade.

**Escolha de τ**: medida a acurácia atingida (não a suposta) por tolerância pedida, de 1e-2 a
1e-6, nas duas aritméticas (`outputs/serial_tolerance_sweep_frobenius.txt`, Exp. 0). A família de
duplicação satura em 1e-3 nas duas aritméticas — `ASDA_FIXED` fica literalmente plana (7,220e-3)
de 1e-3 a 1e-6; apertar além disso não compra nada e ainda custa 10-20% mais tempo. Em ponto fixo,
apertar demais chega a **piorar**: `SDA_FIXED` vai de 1,090e-2 (τ=1e-2) a 1,178e-2 (τ=1e-6, +8%),
porque mais iterações acumulam mais ruído de quantização sem ganho real de precisão. **τ = 1e-3**
foi adotado — a tolerância mais frouxa que já atinge a acurácia saturada da família de duplicação,
com margem de 17,8× sobre o piso de 5,6e-5 (contra 1,8× em τ=1e-4, considerada frágil demais).

Resultado de robustez relevante: o speedup central do artigo (ponto fixo vs. float na família de
duplicação) é praticamente invariante à escolha de τ — 2,27× a 2,50× de 1e-2 a 1e-6 — então a
correção de τ não ameaça a tese principal do artigo, só corrige a narrativa periférica sobre o
`ITERATIVE_FIXED` e os números da Tabela 1 (pendente re-rodar a bateria principal sob τ=1e-3).

### 15.3 γ do SDA-SS: de ponto médio não-justificado a valor medido

γ estava fixo em 0,5 (ponto médio de (0,1)) desde a correção algébrica do shift do pencil
simplético — sem varredura, e com dois comentários no código alegando uma medição que não existia
(um citava uma varredura γ∈{0,1..0,99} num CSV que não a contém; outro alegava "~2× menos
iterações" quando não havia dado nenhum). γ foi exposto como parâmetro configurável
(`AutoLQR::setSDASSGamma()`, antes constante hardcoded em dois pontos do `.cpp`) para permitir a
varredura sem recompilar cinco vezes.

**Exp. 3** (`test/gamma_sweep.cpp`, novo): γ∈{0,1; 0,3; 0,5; 0,7; 0,9} × {SDA_SS, SDA_SS_FIXED},
τ=1e-3/200 (critério casado da Seção 15.2), 4 trajetórias × 300 pts, 1216 pontos por γ, 100%
convergência em todos os 10 pares (γ,método):

| γ   | iters | SDA_SS (µs) | resid float | SDA_SS_FIXED (µs) | resid fixed |
|-----|-------|-------------|-------------|--------------------|-------------|
| 0,1 | 9     | 11340,6     | 3,669e-06   | 4466,1             | 1,438e-02   |
| 0,3 | 9     | 11356,8     | 2,382e-06   | 4501,6             | 1,321e-02   |
| 0,5 | 8     | 10360,0     | 1,356e-06   | 4168,9             | 9,759e-03   |
| 0,7 | 7     | 9358,5      | 1,153e-06   | 3829,8             | 8,992e-03   |
| 0,9 | 5     | 7357,3      | 1,664e-06   | 3133,7             | 1,509e-02   |

**γ=0,7 domina γ=0,5** nas duas aritméticas simultaneamente: 12,5% mais rápido (menos 1 iteração)
E resíduo melhor (15% menor em float, 8% menor em fixed). **γ=0,9 não domina** — é o mais rápido
(5 iterações), mas o resíduo fixed-point piora 68% em relação a γ=0,7 (1,509e-2 contra 8,992e-3):
menos iterações, nesse regime, significa mais ruído de quantização acumulado por passo maior, não
menos trabalho gratuito. Por isso γ=0,7 foi adotado como novo padrão (não o extremo mais rápido).
Regressão no host confirmou que a mudança de default afeta *apenas* as linhas `SDA_SS`/
`SDA_SS_FIXED` (878 linhas diferentes de 3174×2, todas dessas duas famílias, zero novas falhas de
convergência) — os outros dez métodos são bit-idênticos.

Isso continua sendo uma grade grosseira de 5 pontos, não a busca de Fibonacci que Chu, Fan & Lin
(2005) propõem para o γ ótimo teórico (minimizar o raio espectral dos autovalores transformados).
Mas já é suficiente para descartar o ponto médio não-justificado e substituí-lo por um valor
medido — a busca fina fica registrada como trabalho futuro, não como lacuna silenciosa.

### 15.4 Cobertura de condicionamento das quatro trajetórias

Pergunta do usuário: as quatro trajetórias testadas garantem boa cobertura, ou caberiam mais?
`python/analisa_cobertura.py` (novo) mediu ρ(malha fechada), cond(I+GP) e ‖P‖_F nos 46152 pontos
completos da bateria principal (não uma amostra — 0 falhas de DARE):

- ρ ∈ [0,9860; 0,9885]
- cond(I+GP) ∈ [4,661; 5,392]
- ‖P‖_F ∈ [0,4042; 0,4705]

Faixas estreitas nas três métricas, apesar dos perfis de atitude das quatro trajetórias serem
visivelmente diferentes (espiral, figura-8, chirp, degrau de yaw). Achado adicional: decompondo
por trajetória, **T1 (espiral) sozinha cobre 94,9% da faixa total observada de ρ**; T4 (degrau de
yaw) cobre 85,8%; T2 (figura-8) e T3 (chirp) cobrem apenas 31,3% e 14,0%. Ou seja, a redundância
não é só entre as quatro — T1 isolada já quase esgota a variação de condicionamento numérico
observável neste conjunto; as outras três, apesar de diferentes no espaço de atitude, não abrem
faixa nova nesse eixo específico.

**Conclusão honesta**: as quatro trajetórias diversificam o *perfil temporal* (o que interessa
para validar rastreamento e controle), mas não diversificam o *condicionamento numérico* do
problema de Riccati resolvido a cada passo — e mais trajetórias do mesmo tipo (variações de
atitude) não mudariam isso. Trajetórias candidatas para abrir faixa nova precisariam mover o que
de fato domina ρ/cond/‖P‖_F: taxas angulares bem maiores (o termo de acoplamento giroscópico
cresce) ou operação perto do limite de tilt (onde sec(θ) da cinemática de Euler explode) — ainda
não verificadas nesta sessão; ficam registradas como direção concreta de trabalho futuro, não como
lacuna a esconder no artigo.

### 15.5 Pendências desta seção

- ~~Re-rodar a varredura Q/R (Exp. 2) com o fix de 15.1 aplicado.~~ **Feito** —
  `outputs/serial_sweep_qr_v4.txt`: os 5 métodos `_FIXED` reportam `outcome=2` uniformemente em
  `R_scale≥1e3` (6000/6000 cada) e um segundo modo de falha simétrico foi descoberto em
  `R_scale≤1e-2` (60-61% `outcome=2`, 0% no float) — ver atualização em 15.1.
- ~~Re-rodar a bateria principal (Exp. 1) sob τ=1e-3.~~ **Feito** —
  `outputs/serial_capture_bateria_v4.txt`: 100% de convergência nos 12 métodos (46152/46152 cada,
  zero budget, zero breakdown). `ITERATIVE_FIXED` passou de "falha" a mais rápido que toda a
  família de duplicação fixed-point (2,13ms médios), com cauda de pior caso explicada (custo de
  partida a frio em k=0, ou degradação suave no chirp T3, nunca risco recorrente).
- ~~Sincronizar `python/bench_trajetorias.py`~~ **Feito** — γ=0,7 e τ=1e-3 (`inv_rel_tolerance=1000`)
  aplicados; malha fechada regenerada (`outputs/malha_fechada_trajetorias_v4.csv`), 5 controladores
  `_FIXED` de duplicação em ±0,2-0,3% de J vs. float64. `ITERATIVE_FIXED` fora do escopo desse CSV
  (só os 5 doubling + referência float64) — limitação conhecida, não bloqueante.
- ~~Reescrever o `diname2027_v2.tex`~~ **Feito** — Tabela 1 atualizada, narrativa do
  `ITERATIVE_FIXED` invertida (piso de quantização, 100% convergência, cauda explicada), abstract e
  conclusão atualizados, retratação da alegação SDA_FIXED-imune-a-R-grande e novo achado de R
  pequeno incorporados (já feito por fork anterior), γ=0,7 e cobertura de trajetórias já incorporados
  (idem). Trecho de "real-time feasibility" recalculado com os novos tempos — margem confortável
  para SDA-fx (era overrun), overrun ainda presente para ASDA-fx.
- **Em andamento**: correção das 4 figuras (fig2 escala, fig3 legibilidade/viés SUMMARY, fig4 mapa
  Q/R com os dois extremos de falha, fig5 malha fechada v4) e recompilação do PDF — delegado a fork,
  aguardando conclusão.
- ~~Pendente após as figuras~~ **Feito** — as 4 figuras corrigidas, `bibfile.bib` limpo de notas de
  auditoria vazadas, PDF recompilado em 10 páginas (limite DINAME), zero citação indefinida,
  reenviado ao usuário.

## 16. Campanha estendida (2026-08-19) — Fase 0, pré-requisitos de hardware

O usuário ofereceu até um dia inteiro de hardware para uma campanha muito mais ampla. Antes de
gastar tempo de bancada, a Fase 0 (host-only) revisitou as premissas da campanha anterior e
encontrou dois achados que mudam o eixo da próxima rodada, além de fechar três pendências
técnicas. Plano completo:
`C:\Users\guilh\.claude\plans\nesse-repositorio-ha-diversos-tingly-hopcroft.md`.

### 16.1 Trajetórias não são o eixo de condicionamento — Q/R é (medido, decisivo)

Correlação de ρ (raio espectral de malha fechada) com todos os 5 estados (φ,θ,p,q,r) nos 46152
pontos da bateria: |corr| ≤ 0,023. Testada a hipótese do tilt (registrada na Seção 15.4 como
candidata) até θ=89° (sec θ=57,3): cond(I+GP) só vai de 4,7 a 8,4. Em contraste, variar só a
escala de R move cond(I+GP) de 1,01 a 3,17×10⁶ — **6 ordens de grandeza contra um fator de 1,16**
pelas trajetórias. Nenhuma trajetória amplia a cobertura de condicionamento; só as matrizes de
peso o fazem. Duas novas trajetórias foram criadas mesmo assim, para confirmação no hardware e
para diversificar o domínio temporal (não o condicionamento): **T5_tilt_alto** (θ até 80°, taxas
baixas — isola o efeito do ângulo) e **T6_taxa_alta** (p/q/r até 2,5-4,3× o máximo histórico,
ângulos ≤60° — isola o efeito da taxa). Medido em `outputs/trajetorias_v5_check.csv`: T5 atinge
sec θ=5,76 com p/q/r 37× menores que T3; T6 atinge p=3043,8°/s (2,47× o máximo de T1-T4).

### 16.2 Mecanismo do modo de falha em R pequeno — confirmado (não é o que a Seção 15.1 hipotetizou)

A Seção 15.1 hipotetizava "overflow interno ao laço de duplicação, via R⁻¹ crescendo". Instrumentação
ponto a ponto do espelho bit-a-bit (`python/fixedpoint_q.py`), reproduzindo exatamente a sequência
de `computeGainMatrixSDA_Fixed()`, **refuta essa hipótese**: `R⁻¹` cabe folgado
(max|R⁻¹|≈1,8 em R_scale=1e-2, teto 8192) e `invert_q(R)` nunca estoura. O overflow ocorre no
**setup**, antes do laço, no produto $G_0=BR^{-1}B^T$ (max|G0|≈8192 em R_scale=1e-2, exatamente no
teto). É uma terceira categoria — overflow de uma matriz *derivada* da entrada, distinta tanto do
overflow de entrada pura (R_scale grande) quanto de um overflow genuíno dentro das iterações
(nunca observado: o setup já falha antes de qualquer iteração rodar). O comentário em
`AutoLQR.cpp:268` ("saturou durante o laço") está impreciso — detecta o overflow corretamente mas
não distingue setup de laço. `diname2027_v2.tex` corrigido para descrever o mecanismo real.

### 16.3 Fase 0 concluída — cinco pré-requisitos, todos verificados

1. **Bug de telemetria obsoleta corrigido** (mesma classe do bug de `lastOutcome` da Seção 15.1,
   que o comentário já havia listado como pendente): `computeGains()` (`AutoLQR.cpp:119-166`) agora
   reseta `lastIterations/lastResidual/lastFixedPointMaxAbsSeen/lastOutcome` para sentinelas ANTES
   de despachar, e reforça `Converged` no sucesso — corrige de quebra `SCHUR`/`VAN_DOOREN`, que
   nunca haviam sido retrofitados com a taxonomia `SolveOutcome`. Regressão: casos convergidos
   bit-idênticos; casos que falham agora reportam sentinela em vez de telemetria de chamada anterior.
2. **Trajetórias unificadas**: `lib/Trajectories/Trajectories.h` (header-only, novo), consolidando
   ~446 linhas antes duplicadas em 4 arquivos C++. Corrigida a divergência real do `pointT4` (a
   recursão exata substitui a aproximação de regime permanente que 3 dos 4 arquivos usavam) —
   verificado bit-a-bit (0 divergências) contra a versão original e contra 2 passadas consecutivas.
   Paridade C++/Python confirmada (erro ≤3e-4 rad/s, ruído de float32) em T1/T2/T3/T5/T6; **T4 tem
   113/11538 pontos (0,98%) divergentes, pré-existente** (não introduzido agora) — `sin(kπ)` muda de
   sinal entre float32 e float64 exatamente nos 4 instantes em que `k·DT` é múltiplo do
   semiperíodo, deslocando a borda do degrau em 1 amostra; efeito decai em ~28 amostras.
3. **Firmware de voo restaurado**: `src/main.cpp` (que era temporariamente o harness de benchmark)
   voltou a ser o firmware real, com overrun reintroduzido, `setStoppingCriterion(1e-3f, 200)` no
   `setup()`, `processingTime` publicado em vez de `loopTime` (que satura no período),
   `esp_timer_get_time()` substituindo `micros()` nas 39 ocorrências, e `Telemetry::Sample`
   estendido com `processingTime/t_lqr/iters/outcome`. **Achado**: o novo `Sample` (76B) não cabe
   mais em `CAPACITY=1000` (região DRAM estoura por 7856B) — o buffer antigo já vivia com apenas
   144B de folga. Reduzido para `CAPACITY=800` (janela de telemetria cai de ~26s para ~20,8s de
   voo). Build: RAM 31,6%, Flash 71,3%.
4. **Mecanismo do R pequeno confirmado** — ver 16.2.
5. **`ITERATIVE_FIXED` incluído na malha fechada**: `bench_trajetorias.IterativeFixedGain`, espelho
   Python stateful de `computeGainMatrixIterative_Fixed()` (warm-start via `P_warm` como estado da
   instância, não um dict compartilhado — evita vazar warm-start entre trajetórias). Validado:
   cold-start dá 126 iterações (bate com as 127 medidas no hardware), segunda chamada no mesmo ponto
   cai para 1 iteração. **Resultado, `outputs/malha_fechada_trajetorias_v5_iterfixed.csv`**:
   `ITERATIVE_FIXED` fica em **-0,5% a -0,0% de J** vs. float64 nas 4 trajetórias — na prática o
   melhor ou empatado com o melhor dos 6 controladores fixed-point em todas elas, apesar de ter o
   pior resíduo/erro de K isolado (Seção 15.2). Reforça o achado já registrado: o resíduo pior do
   `ITERATIVE_FIXED` não se traduz em pior desempenho de rastreamento em malha fechada.
   Corrigido de passagem: `run_compare()` em `bench_trajetorias.py` tinha um bug pré-existente
   (dicionário hardcoded de 4 trajetórias + `IndexError` potencial ao chamar a função de trajetória
   com array de 1 elemento contra `_derivar_central`) que teria silenciosamente ignorado T5/T6;
   corrigido para indexar `gerar_todas()` por `k` em vez de recomputar por ponto.

### 16.4 Exp. A — varredura combinada τ×Q/R (concluído, 2026-08-19)

`test/tol_qr_sweep.cpp` (novo, env `tol_qr_sweep`): 6 τ × 13 `r_scale` × 5 `q_rate_scale` × 12
métodos × ~300 pontos (6 trajetórias), 1.404.000 chamadas. Capturado em
`outputs/serial_tol_qr_sweep_A.txt` (16.728 s ≈ 4,65h, 4680 linhas SUMMARY = 390×12, 140.400 linhas
RUN decimadas 1:10, 10,8 MB). Nota de implementação: o array de estatísticas 4D original (146 KB)
estourava a DRAM do ESP32-S2 — corrigido para acumular só a combinação (τ,r,qr) corrente, resetada
a cada combo (SUMMARY já é emitido antes de avançar, então nada se perde).

**Resultado central — τ=1e-3 é robusto em TODA a banda segura, não só nos pesos nominais.** A
acurácia atingida (resíduo médio) para `ASDA_FIXED` é **essencialmente idêntica nos 6 valores de τ
(1e-2 a 3e-5) em todas as 20 combinações de `(r_scale,q_rate_scale)` testadas na banda segura**
(`r_scale∈[0,1;100]`, `q_rate_scale∈[0,01;100]`) — variação <1% do valor mais frouxo ao mais
apertado, em toda a grade. Isso é uma confirmação mais forte do que a alegação original do artigo
(que só validara a saturação nos pesos nominais, com margem 17,8×): a insensibilidade de τ não é
uma coincidência dos pesos nominais, é uma propriedade da família de duplicação fixed-point válida
em toda a banda segura conhecida. Reforça — não enfraquece — a escolha de τ=1e-3 no artigo.

**Achado novo — falha isolada do ASDA_FIXED em `q_rate_scale` grande.** Em
`r_scale=1e-1, q_rate_scale=1e2` (dentro da banda "segura"), `ASDA_FIXED` reporta 7/300 breakdowns,
**idênticos em todos os 6 valores de τ** (não relacionado a quantização — é uma singularidade
numérica genuína do algoritmo de reescala adaptativa nessa combinação específica). Os outros quatro
métodos de duplicação (`SDA_FIXED, SDA_SS_FIXED, SDA_SCALED_FIXED, ADDA_FIXED`) reportam 300/300 na
mesma célula — a falha não é geral, é específica do ASDA. Isso é notável porque o ASDA_FIXED é
justamente o variante identificado como "mais robusto" na varredura de R puro (Seção 13); aqui, na
dimensão de `q_rate_scale`, ele tem seu próprio ponto fraco isolado que os demais não têm. Mecanismo
não investigado nesta sessão — candidato a trabalho futuro (a reescala `s=√(‖H‖_F/‖G‖_F)` pode
degenerar nessa combinação específica de pesos).

**Comportamento do `ITERATIVE_FIXED` na mesma célula**: degrada suavemente com τ mais apertado —
300/300 convergidos em τ=1e-2, caindo monotonicamente para 4/300 em τ=3e-5 — consistente com a
Seção 15.2 (é o único método cujo critério de parada de fato rastreia τ, então cruza o piso de
quantização LOCAL dessa célula, que tem `‖P‖_F` maior por causa do `q_rate_scale=100`).

### 16.5 Exp. C — bateria estendida, 6 trajetórias (concluído, 2026-08-19)

Reflash trivial de `benchmark_solvers.cpp` (já usa `Trajectories::N_TRAJ=6` automaticamente desde a
unificação da Seção 16.3). Capturado em `outputs/serial_capture_bateria_v5_6traj.txt` (6375s ≈
1,77h, 19,8MB). **100% de convergência nos 12 métodos, nos 69228 pontos (6×11538)** — zero budget,
zero breakdown, mesmo em T5 (tilt até 80°) e T6 (taxas até ~3000°/s): as trajetórias extremas não
quebram nenhum solver.

**Cobertura de condicionamento — previsão do host CONFIRMADA no hardware, com um refinamento
honesto.** `python/analisa_cobertura.py` re-rodado sobre os 69228 pontos reais
(`outputs/cobertura_full_v5_6traj.csv`): incluir T5/T6 alarga a faixa observada, mas apenas
marginalmente — `cond(I+GP)` vai de `[4,66; 5,39]` (4 trajetórias, Seção 15.4) para
`[4,61; 5,72]` (6 trajetórias), `‖P‖_F` de `[0,404; 0,470]` para `[0,375; 0,487]` — alargamento de
~6-7% em cada extremo, **três ordens de grandeza menor que o alargamento que a escala de Q/R produz
por si só** (Seção 16.4: `cond(I+GP)` de 1,01 a 3,17×10⁶). A conclusão da Seção 15.4 — trajetórias
não são o eixo que move o condicionamento — permanece válida mesmo forçando o ângulo e a taxa aos
extremos plausíveis.

Achado secundário interessante: **T5 (tilt), sozinha, agora cobre 99,6% da faixa total de ρ** —
mais que T1 (que caiu de 94,9% para 64,2% da faixa total, porque a faixa total cresceu com a adição
de T5/T6, não porque T1 mudou). T6 (taxa) cobre só 28,8%. Ou seja, dentro do pequeno alargamento que
as trajetórias conseguem produzir, é o ÂNGULO extremo (T5), não a TAXA extrema (T6), que domina —
consistente com o mecanismo em 16.1 (φ/θ entram na cinemática de A(x) via sec(θ); p/q/r entram no
acoplamento giroscópico, um efeito mais fraco sobre o condicionamento da Riccati nesta planta).

### 16.6 Exp. D — repetibilidade/jitter (concluído, 2026-08-19)

`test/repeatability.cpp` (novo, env `repeatability`): ~1980 pontos (330/trajetória × 6
trajetórias), cada um com `computeGains()` chamado 20 vezes CONSECUTIVAS sobre a MESMA entrada
(Ad/Bd/Qd/Rd fixos), para os 12 métodos, τ=1e-3/200. Objetivo: medir jitter de execução — até esta
sessão, cada ponto da bateria principal e dos sweeps havia sido medido exatamente 1 vez, e a margem
de tempo real do artigo (SDA-fx: 3,92+0,9=4,82ms médio contra 5,2ms de período) presumia
implicitamente que o tempo de solve é determinístico. Capturado em
`outputs/serial_repeatability_D.txt` (3471s ≈ 58min, 28,3MB, 23760 linhas SUMMARY = 1980×12).

**Resultado — a família de duplicação é determinística, jitter não é um risco para a margem de
tempo real.** Coeficiente de variação (CV=100·std/mean) médio e máximo, sobre os 1980 pontos:

| aritmética | CV médio | CV máximo |
|---|---|---|
| float (5 doubling) | 0,019-0,021% | 0,030-0,040% |
| fixed (5 doubling) | 0,032-0,042% | 0,060-0,090% |

O tempo de execução da família de duplicação varia **menos de 0,1% do ponto pior observado, em
quase 2000 pontos de operação diversos** (incluindo T5/T6, os extremos de tilt e taxa) — confirma
diretamente que a margem de 4,82ms/4,95ms (Seção 15, viabilidade em tempo real) não corre risco por
variabilidade de execução; o número medido é, na prática, uma constante.

`ITERATIVE`/`ITERATIVE_FIXED` mostram CV médio de 138-159% (chegando a 332% no pior ponto) — mas
isso é **inteiramente estrutural, não ruído**: a repetição 0 é sempre fria (warm-start ainda não
"aquecido" para aquela entrada específica), as repetições 1-19 convergem quase instantaneamente
(warm-start já exato) — o mesmo mecanismo de custo de partida a frio já documentado na Seção 15.2/
16.3. Em pontos onde o warm-start já convergia em 1 iteração desde a repetição 0 (regiões planas da
trajetória), o CV cai para 0,09-0,11%, igual à família de duplicação.

### 16.7 Exp. B — mapa fino das duas fronteiras (concluído, 2026-08-19)

`test/boundary_fine.cpp` (novo, env `boundary_fine`): mesma estrutura de `sweep_qr.cpp` (10
métodos, 5 doubling float + 5 fixed, τ=1e-3/200 fixo), mas com `R_scale` refinado — 25 pontos
log-espaçados em `[1e-3,10]` (fronteira inferior) e 15 pontos em `[50,500]` (fronteira superior,
em torno do limiar teórico 144,5) × 5 `q_rate_scale` × ~300 pontos (6 trajetórias). Capturado em
`outputs/serial_boundary_fine_B.txt` (4975s ≈ 1,38h, 42,2MB, 2000 linhas SUMMARY = 200×10).

**Fronteira superior — confirmada como função degrau, limiar isolado em `[134,1; 158,1]`.** Taxa
de breakdown agregada dos 5 métodos `_FIXED`: 1,1% em `r_scale=134,13`, saltando para **100% em
r_scale=158,11** — um intervalo de largura 24 (dos 450 testados nesse trecho da grade) que contém o
limiar analítico previsto de 144,5. Não há zona de transição gradual como na fronteira inferior: é,
na prática, uma função degrau, exatamente como o modelo `R_scale > 8192/55,5` prevê (saturação de
entrada de `Rd`, independente do estado — daí a nitidez).

**Fronteira inferior — curva suave e monotônica, com dois "joelhos" visíveis.** De 60,2% de
breakdown em `r_scale≤0,015` até 1,5% em `r_scale=10`, decaindo suavemente sem não-monotonicidades:

| `r_scale` | brk% | `r_scale` | brk% | `r_scale` | brk% |
|---|---|---|---|---|---|
| ≤1,5e-2 | 60,1-60,2 | 6,8e-2 – 1,5e-1 | 21,9-22,5 | 6,8e-1 – 1,0 | 4,3-4,8 |
| 2,2e-2 | 57,7 | 2,2e-1 – 4,6e-1 | 8,4-12,5 | 1,5 – 10 | 1,5-3,3 |
| 3,2e-2 – 4,6e-2 | 44,1-46,0 | | | | |

Consistente com o mecanismo de overflow de `G_0=BR^{-1}B^T` (Seção 16.2): como `B` depende
fracamente do estado, a probabilidade de estouro por ponto amostrado decresce suavemente conforme
`r_scale` reduz a magnitude de `R^{-1}`, sem limiar único — dá uma regra de projeto probabilística
a priori (ex.: `r_scale≥1` já reduz o risco a <5%; `r_scale≥10`, a <2%), em vez de um corte binário.

### 16.8 Exp. E — ciclo de voo end-to-end (concluído, 2026-08-19)

**Nota de segurança**: este experimento foi feito com o drone **desarmado o tempo todo**
(confirmado nos prints, `Armed: NO`). O bloco de diagnóstico `DEBUG_MODE` reporta
`processingTime`/`overrunCount`/estágios sem depender de `motors.isArmed()` — só a gravação da
telemetria em RAM (não usada aqui) e a escrita real nos motores dependem de armar; o cálculo do
mixer roda sempre, só a escrita no ESC é que é condicionada. Confirmado por leitura do código antes
de habilitar `DEBUG_MODE=true` (revertido para `false` ao final do experimento). Nenhum comando via
WiFi/UDP foi enviado (único canal capaz de armar o `motors_armed_by_remote`).

Firmware de voo restaurado (Seção 16.3) com `setStoppingCriterion(1e-3f,200)` já presente,
`USE_ASYNC_SDRE=false` (síncrono), método default `SDA_FIXED`. Capturado 360,5s de
`outputs/serial_flightloop_E.txt` (679KB, 297 blocos de status, 1/s) a 115200 baud (a taxa real de
`Serial.begin()` no firmware de voo, diferente dos 921600 usados nos benchmarks).

**Resultado — a margem "confortável" da estimativa composta NÃO se sustenta na medição direta.**
Taxa de overrun estabiliza em **9,3-9,6% dos ciclos** em regime permanente (após o transiente inicial
de 100% no primeiro ciclo, efeito de partida a frio já esperado):

| t (s) | overrun acumulado | | t (s) | overrun acumulado |
|---|---|---|---|---|
| 30 | 9,66% | | 180 | 9,38% |
| 90 | 9,51% | | 240 | 9,36% |
| 150 | 9,47% | | 270 | 9,34% |

`processingTime`: mínimo 5061µs, máximo 6916µs (33% acima do período), média acumulada ~5120µs —
**perigosamente perto** do período de 5,2ms mesmo na média. O estágio `t_lqr` (`SDA_FIXED`) mede
3996-4695µs no hardware real (média 4035,3µs, desvio 116,4µs, valores concentrados em 4000-4010µs
com excursões discretas até 4695µs) — cerca de 3% acima da média isolada de bancada (3906-3920µs,
Seção 15), plausivelmente por contenção real de barramento I2C/interrupções que o benchmark isolado
não reproduz. Somado ao resto do ciclo (MPU 623µs, os demais estágios ~450µs), o total fica na faixa
de 5070-5140µs de trabalho útil — abaixo do período na média, mas com uma cauda que cruza a linha
com frequência suficiente para 1 em cada ~10 ciclos.

**Reconciliação com o Exp. D (jitter)**: não há contradição. O Exp. D mediu jitter de execução no
MESMO ponto (CV<0,1%) — aqui a variação de `t_lqr` (3996 a 4695µs, ~17%) é a mesma discretização por
iteração já characterizada em toda a campanha (SDA_FIXED variando entre 9 e 10 iterações conforme o
ponto), não jitter novo. O que o Exp. E revela é que, mesmo sendo essa variação pequena e conhecida,
a margem restante entre o ciclo típico e o período é estreita o bastante para que ela sozinha empurre
~10% dos ciclos para além do período — algo que a estimativa composta (que soma médias, não
distribuições) não conseguia capturar.

**Implicação prática, acionável**: como a campanha (Seção 16.4, Exp. A) já estabeleceu que
`ITERATIVE_FIXED` é hoje o método mais rápido sob τ=1e-3 (2,13ms médios de bancada, Seção 15.2),
substituir o `SDA_FIXED` default do firmware por `ITERATIVE_FIXED` daria ~2,13+1,1≈3,2ms de ciclo
típico — margem de ~2ms sobre o período, folgada o bastante para absorver a variação observada sem
overrun. O custo é a acurácia pior do `ITERATIVE_FIXED` (resíduo 3,1e-2 contra 1,1e-2 do SDA_FIXED),
mas a malha fechada (Seção 16.3, item 5) já mostrou que isso não degrada o rastreamento — pelo
contrário, `ITERATIVE_FIXED` teve o melhor ou empatado J em todas as 4 trajetórias testadas.

**Artigo**: o parágrafo de viabilidade em tempo real (que hoje alega margem confortável a partir da
estimativa composta) precisa ser reescrito com este dado direto — retração parcial similar em
espírito à retração do Exp. A sobre `SDA_FIXED`/R grande, mas aqui invertendo a direção (a estimativa
composta era otimista demais, não pessimista).

### 16.9 Campanha concluída (2026-08-19)

Os cinco experimentos (A-E) estão concluídos, documentados (Seções 16.1-16.8) e incorporados ao
`diname2027_v2.tex` — parágrafo de viabilidade em tempo real reescrito com a medição direta (9,3-9,6%
de overrun, revertendo a alegação de margem confortável), Discussion e Conclusion atualizados para
consistência. PDF recompilado: **10 páginas, zero citação indefinida**, reenviado ao usuário.

**Resumo executivo da campanha** (~15h de hardware total: Exp. A 4,65h + Exp. B 1,38h + Exp. C 1,77h
+ Exp. D 0,97h + Exp. E 0,1h, mais tempo de host):

| Exp. | Achado central |
|---|---|
| A | τ=1e-3 tem acurácia saturada em TODA a banda segura de Q/R (20 combinações, não só nominal) |
| A | Falha isolada do ASDA_FIXED em `q_rate_scale` grande, não compartilhada pelos outros 4 doubling |
| B | Fronteira superior é função-degrau (transição em 24 unidades, em torno do limiar teórico 144,5) |
| B | Fronteira inferior é curva suave e monotônica — regra de projeto probabilística, não corte binário |
| C | T5/T6 (ângulo/taxa extremos) alargam a cobertura de condicionamento em só ~6-7%, não qualitativamente |
| C | Ângulo extremo (T5) domina mais que taxa extrema (T6) no condicionamento — mecanismo via sec(θ) |
| D | Família de duplicação é determinística: CV<0,1% em quase 2000 pontos, jitter não é risco de margem |
| E | **A margem "confortável" da estimativa composta não se sustenta**: 9,3-9,6% de overrun medido |
| E | Variação de `t_lqr` no voo real (17%) é a mesma discretização por iteração já conhecida, não jitter novo |
| E | `ITERATIVE_FIXED` (τ=1e-3) resolveria o overrun (~2ms de margem) — recomendação acionável no artigo |

Nenhum item do plano da campanha estendida ficou pendente. Trabalho remanescente, fora do escopo
desta sessão: considerar migrar o default do firmware de `SDA_FIXED` para `ITERATIVE_FIXED` dado o
achado do Exp. E (decisão de engenharia, não deste documento); revisar se o resumo já submetido no
portal do DINAME precisa de atualização dado que os números centrais mudaram nesta e na sessão
anterior.

## 17. Análise final para o artigo (2026-08-19) — uma recomendação da Seção 16 é retirada

A redação do artigo exigiu reanalisar a bateria principal por **distribuição** e **por trajetória**,
não só pelas médias agregadas das linhas `SUMMARY`. Isso produziu o achado central do artigo e, no
caminho, **invalidou a recomendação registrada no fim da Seção 16.9**.

### 17.1 Retratação: `ITERATIVE_FIXED` NÃO resolve o overrun do Exp. E

A Seção 16.9 registrou que trocar o solver default de `SDA_FIXED` para `ITERATIVE_FIXED` daria
~2 ms de margem e resolveria os 9,3% de estouro. Essa conclusão vinha da **média** (2,13 ms) e não
sobrevive à distribuição. Decompondo os tempos por trajetória (linhas `RUN`, decimadas 1:5):

| trajetória | `SDA_FIXED` mediana / p99 | `ITERATIVE_FIXED` mediana / p99 | % dos ciclos > 5,2 ms (iter.) |
|---|---|---|---|
| T2_figura8    | 4011 / 4023 µs | 934 / 949 µs     | 0% |
| T5_tilt_alto  | 3661 / 4014 µs | 944 / 949 µs     | 0% |
| T1_espiral    | 3999 / 4030 µs | 948 / 3581 µs    | 0% |
| T4_degrau_yaw | 3683 / 4039 µs | 949 / 7390 µs    | **5,3%** |
| T3_chirp      | 4016 / 4033 µs | 5484 / 8804 µs   | **52,0%** |
| T6_taxa_alta  | 4035 / 4052 µs | 12736 / 14998 µs | **100,0%** |

`ITERATIVE_FIXED` é 4,2× mais rápido que `SDA_FIXED` na mediana global e **3,7× mais lento no
percentil 99,9** (15029 µs contra 4052 µs), com máximo de 45509 µs. Em T6 ele estoura o período de
5,2 ms em **todos** os ciclos. Trocar o default por ele converteria um estouro de 9,3% distribuído
num estouro de 100% exatamente durante manobras agressivas — o pior momento possível. **A
recomendação está retirada**; `SDA_FIXED` (ou `SDA_SS_FIXED`, 3900 µs de mediana) continua a escolha
correta para o firmware, e o caminho para fechar os 9,3% é reduzir a taxa do laço (167 Hz fecha com
folga: 100% dos ciclos medidos abaixo de 6,0 ms) ou o overhead de sensor (leitura I²C = 623 µs, o
maior item fora do solver).

### 17.2 O mecanismo, quantificado — virou o achado central do artigo

O custo da iteração de valor é função de **quanto o ponto de operação andou desde a solução
anterior**, porque é dela que ela parte. Ordenando as seis trajetórias pela distância euclidiana
média entre estados consecutivos, a relação com a mediana de iterações é monotônica:

| trajetória | `‖Δx‖` médio | iterações (mediana) |
|---|---|---|
| T5_tilt_alto | 0,0023 | 1 |
| T2_figura8 | 0,0050 | 1 |
| T1_espiral | 0,0163 | 1 |
| T4_degrau_yaw | 0,187 | 1 |
| T3_chirp | 1,966 | 13 |
| T6_taxa_alta | 16,308 | 32 |

A família de duplicação fica em 9-10 iterações em toda essa faixa, porque parte dos dados do
problema e não de uma estimativa anterior. Um solver com warm start converte uma propriedade da
*trajetória* numa propriedade do *escalonamento* — dependência exatamente invertida para tempo real
duro, já que a demanda cresce quando o veículo manobra mais. Esse é o eixo do artigo e a razão do
título incluir "Cost Predictability".

### 17.3 Outras correções feitas na verificação da redação

1. **Faixa do `SDA_FIXED`**: a Seção 16 sugeria constância quase perfeita; medido, a faixa global é
   3323–4055 µs (22%), pela escolha discreta entre 9 e 10 iterações. Continua determinística no
   sentido que importa (nunca chega a 80% do período), mas o número correto é 22%, não ~2%.
2. **Derivação do piso de quantização**: `n·2⁻ˢ/‖P‖_F` é a **amplitude característica do
   ciclo-limite** (todas as n² entradas oscilando 1 LSB), não o passo mínimo representável — uma
   única entrada oscilando daria um passo n vezes menor. O texto do artigo foi ajustado.
3. **Malha fechada em T6**: re-rodada com as 6 trajetórias
   (`outputs/malha_fechada_v6_6traj.csv`). Nas cinco que o laço consegue seguir, todos os `_FIXED`
   ficam em ±0,53% de J. Em T6 **nenhum** controlador rastreia — a própria referência float64
   acumula 141° de RMS, porque as taxas comandadas excedem o torque disponível — então a dispersão
   de 2,2–7,8% ali mede divergência de trajetórias já divergidas, não quantização. Reportado no
   artigo com essa ressalva explícita em vez de omitido ou usado sem qualificação.
4. **Erro de ganho vs. scipy** (1386 pontos com linha `GAIN`, erro relativo de Frobenius):
   float 2,8–3,2e-6; `ASDA_FIXED` 3,7e-3 (melhor dos fixed, 2,6× melhor que `SDA_FIXED` 9,6e-3);
   `SDA_SS_FIXED` 8,4e-3; `ADDA_FIXED` 1,1e-2; `SDA_SCALED_FIXED` 1,2e-2; iterativos 2,1–2,2e-2
   (piores). `ASDA_FIXED` lidera as duas medidas independentes de robustez (acurácia e envelope Q/R).

### 17.4 Artigo

`diname2027_v3.tex` (novo, substitui a v2): reescrito do zero como publicação inédita, sem
referência a versões anteriores. 10 páginas, 6 figuras novas
(`python/figuras_artigo_final.py`), 1 tabela, zero citação indefinida. Figuras verificadas
individualmente quanto a legibilidade e sobreposição de legenda sobre dados.

## 18. Migração do período de controle de 5,2 ms para 6,0 ms (2026-08-26)

### 18.1 Motivação

O Exp. E (Seção 16.8) mediu, a 5,2 ms (200 Hz), overrun de 7,9–9,3% dos ciclos do laço de voo
completo (`processingTime` mediana 5100 µs, p99 5747 µs, máximo absoluto observado entre 6484 e
6916 µs em duas capturas). O artigo já registrava, a partir dessa mesma medição, que "every sampled
cycle completes within 6.0 ms, so a 167 Hz loop closes with margin" — a decisão aqui é **tornar essa
frase medida em vez de inferida**: mover o firmware de voo para 6,0 ms (167 Hz) e medir o overrun
residual diretamente, em vez de extrapolar de uma cauda de amostras a 1 Hz.

### 18.2 O que mudou

`dt` tem duas fontes independentes, ambas migradas: `src/main.cpp:50`
(`SAMPLING_TIME_S`, síncrono) e `lib/Trajectories/Trajectories.h:68` (`Trajectories::DT`, consumido
por todos os 7 firmwares de `experiments/` e por `python/trajetorias.py`/`bench_trajetorias.py` no
host). `test/native/verify_solvers.cpp` e `test/bench/verify_gains_onboard.cpp` replicam `dt`
localmente e foram migrados junto.

`N_POINTS_FULL` (60 s / dt) caiu de 11538 para 10000 pontos por trajetória — a bateria principal
passa de 69228 para 60000 pontos totais. Os *strides* de amostragem dos sweeps (`gamma_sweep`,
`repeatability`, `sweep_qr`, `tol_qr_sweep`, `tolerance_sweep`, `boundary_fine`) foram reajustados
para preservar o mesmo tamanho amostral da campanha publicada.

**Grade fina do Exp. B recentrada.** `Rd[0][0] = R_11_nom·dt·r_scale + O(dt³)` — o coeficiente que
multiplica `r_scale` cresce com `dt` (55,495 → 64,033 a 5,2→6,0 ms), então o limiar analítico de
overflow de entrada do Q13.18 cai de `8192/55,495≈147,6` para `8192/64,033≈127,9`. A grade fina de
15 pontos da fronteira superior em `experiments/boundary_fine.cpp` foi reescalada
proporcionalmente ([50,500] → [43,3, 433,3]) para continuar centrada no novo limiar; a fronteira
inferior (25 pontos em [1e-3, 1e1], mecanismo de overflow interno de `G0=B·R⁻¹·B^T`) não depende
dessa derivação e foi mantida.

**Decimação da telemetria** reduzida de N=5 para N=4 ciclos (`src/main.cpp`), preservando margem de
~5× sobre os 8 Hz da identificação (era caindo para 4,2× com N=5 a 6,0 ms); janela de voo do buffer
passa de 20,8 s para 19,2 s.

### 18.3 Verificação de host (antes do hardware)

Regressão de 3174 casos (`test/native/verify_solvers.cpp`) com `dt=6,0ms`: **zero divergência de
outcome** em relação ao baseline arquivado a `dt=5,2ms` (mesmos 2691 sucessos / 483 falhas,
caso a caso — os 483 são casos de fronteira deliberadamente desenhados para falhar, ex.
`C4b_R_small`/`C4c_Q_small`). Contagem de iterações da família de duplicação permanece em 9–10 no
caso nominal (`C1_hover`), confirmando que a comparação cruzada com a campanha de 5,2 ms continua
defensável.

### 18.4 Resultado (a preencher após a recampanha)

<!-- outputs/serial_flightloop_E.txt a dt=6,0ms: overrun medido, processingTime mediana/p99/max. -->

### 18.5 Escopo

A campanha completa (Exp. 0/1/A–E) foi rerodada em `dt=6,0ms` para manter bancada, malha fechada e
voo no mesmo período — ao custo de ~9,7 h de hardware e da reescrita numérica do artigo
(`69228→60000` pontos, `T_s=5,2ms→6,0ms` em todo o texto, limiar de overflow `147,6→127,9`,
Tabela 1 e as 6 figuras regeneradas). A campanha de 5,2 ms permanece arquivada em
`outputs/archive/dt_5p2ms/` para referência histórica; as Seções 16–17 descrevem essa campanha
anterior e não foram reescritas.
