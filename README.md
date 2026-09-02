# SDRE_VECTORIZED

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22256810.svg)](https://doi.org/10.5281/zenodo.22256810)
[![Código: MIT](https://img.shields.io/badge/c%C3%B3digo-MIT-blue.svg)](LICENSE)
[![Dados: CC BY 4.0](https://img.shields.io/badge/dados-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

Controle de atitude de quadricóptero pelo método **SDRE** (*State-Dependent Riccati
Equation*) em **ponto fixo Q13.18**, para microcontroladores **sem FPU**.

O repositório é duas coisas ao mesmo tempo, e é bom saber disso desde o começo:

1. **Um firmware de voo real.** `src/main.cpp` faz o drone voar a 167 Hz, resolvendo
   uma equação de Riccati a cada ciclo de controle num ESP32-S2 que não tem unidade
   de ponto flutuante.
2. **Uma bancada de medição de solvers de Riccati.** Doze solvers (seis algoritmos ×
   duas aritméticas) medidos no mesmo hardware, sobre os mesmos 60000 pontos de
   operação, com o mesmo critério de parada — a base experimental do artigo
   **DINAME 2027**.

O mesmo código de solver (`lib/AUTOLQR/`) roda nos dois: o que voa é o que se mede.

---

## Índice

- [Comece por aqui](#comece-por-aqui)
- [O problema em uma página](#o-problema-em-uma-página)
- [Parte 1 — Voar o drone](#parte-1--voar-o-drone)
- [Parte 2 — Medir os solvers](#parte-2--medir-os-solvers)
- [Parte 3 — Analisar e auditar](#parte-3--analisar-e-auditar)
- [Mapa do repositório](#mapa-do-repositório)
- [Todos os envs do PlatformIO](#todos-os-envs-do-platformio)
- [Onde procurar quando algo dá errado](#onde-procurar-quando-algo-dá-errado)

---

## Comece por aqui

```bash
pip install -r requirements.txt          # dependências Python
pio run -e esp32-s2-saola-1              # compila o firmware de voo (não grava)
```

Depois escolha o caminho:

| Se você quer… | Vá para |
|---|---|
| fazer o drone voar | [Parte 1](#parte-1--voar-o-drone) |
| medir os solvers de Riccati | [Parte 2](#parte-2--medir-os-solvers) |
| refazer os números e figuras do artigo | [Parte 3](#parte-3--analisar-e-auditar) |
| entender por que o projeto existe | a seção abaixo |

---

## O problema em uma página

Estabilizar a rotação de um quadricóptero pequeno exige realimentação acima de 50 Hz.
O método SDRE fornece o ganho ótimo para o estado **atual** — o que significa resolver
uma equação algébrica de Riccati de tempo discreto (DARE) **uma vez por ciclo de
controle**. Num microcontrolador sem FPU, cada operação em ponto flutuante é emulada
em software pelo compilador, e essa resolução vira o custo dominante do sistema.

Duas escolhas atacam isso:

- **Aritmética de ponto fixo Q13.18** — 13 bits inteiros (alcance ±8192) e 18 bits
  fracionários (resolução 2⁻¹⁸ ≈ 3,8×10⁻⁶). Multiplicações acumulam em 64 bits e
  voltam com deslocamento de 18. Roda de 1,9 a 2,6× mais rápido que a emulação de
  ponto flutuante, com a mesma contagem de iterações.
- **Algoritmos de duplicação** (*structure-preserving doubling*) — convergem
  quadraticamente e, o que mais importa para tempo real, num **número fixo de
  iterações independente do ponto de operação**: 9 em todos os 60000 pontos (7 para
  a variante SDA-SS). Já a iteração de valor clássica varia de 1 a 32 iterações
  conforme a manobra, o que a torna inutilizável num orçamento de tempo rígido.

**O resultado central:** perto da convergência, o passo relativo entre iterações passa
a medir o *formato numérico*, não a solução. Existe um piso analítico

$$\tau_{\min} = \frac{n\,2^{-s}}{\|\mathbf{P}\|_F}$$

que neste sistema fica entre 4,70×10⁻⁵ e 6,10×10⁻⁵. Abaixo dele, apertar a tolerância
não compra exatidão nenhuma — só custa tempo — e o sintoma **inverte** conforme a
ordem de convergência: métodos de duplicação atingem um ponto fixo exato em bits e
declaram qualquer tolerância satisfeita (sucesso falso); a iteração de valor entra num
ciclo-limite de ±1 LSB e falha toda tolerância abaixo do piso (fracasso falso).

---

## Parte 1 — Voar o drone

### O que o firmware faz

`src/main.cpp`, a cada ciclo de **6,0 ms (167 Hz)**:

```
lê a IMU (MPU6050 por I²C)  →  0,62 ms
filtro Madgwick (atitude)   →  0,07 ms
monta A(x) da fatoração SDC →  0,001 ms
resolve a DARE em Q13.18    →  3,68 ms   ← o custo dominante
lei de controle u = -Kx     →  0,002 ms
mistura X-quad → 4 motores  →  0,07 ms
```

Mediana de 4,70 ms por ciclo, com folga de 1,3 ms sobre o período. Medido em 10
janelas de 360 s: **21 de 475120 ciclos (0,004%)** passaram do período.

### Hardware

| Item | Especificação |
|---|---|
| Placa | ESP32-S2 Saola-1 (Xtensa LX7 a 240 MHz, **sem FPU**) |
| IMU | MPU6050 (6-DOF) por I²C — **obrigatória**, o firmware trava sem ela |
| Magnetômetro | QMC5883L (opcional, 9-DOF) — desligado por padrão |
| Motores | 4 × brushed, hélice 55 mm, configuração X |
| Braço | 60 mm (braço efetivo 42,4 mm em X) |
| Inércias | Ixx 42,95 / Iyy 37,77 / Izz 76,15 (×10⁻⁶ kg·m²) |

### Gravar o firmware

```bash
pio run -e esp32-s2-saola-1 -t upload --upload-port COM3
pio device monitor -p COM3 -b 115200        # acompanhar
```

### Pilotar

O drone sobe como **ponto de acesso WiFi** e fala o protocolo **CRTP**, o mesmo do
Crazyflie — então o app **ESP-Drone** da Espressif (Android/iOS) funciona direto.

```
Rede:  ESP-DRONE
Senha: 12345678
Porta: UDP 2390
```

**Sequência de voo:**

1. Ligue o drone em superfície plana e **não o mova** — o `setup()` trava o *yaw*
   inicial como referência e calibra o filtro.
2. Conecte o celular à rede `ESP-DRONE`. O LED indica que o sistema está pronto.
3. Abra o app. Ao conectar, os motores continuam **desarmados**.
4. **O primeiro comando recebido arma os motores.** Não há botão de armar separado —
   mexer no stick já arma. Tenha isso em mente ao segurar o drone.
5. Desconectar o app (ou perder o WiFi) desarma os motores e salva a telemetria.

**Convenção dos sticks:** *roll* e *pitch* são ângulo (setpoint direto). **Yaw é
taxa** — o stick comanda velocidade angular em rad/s, integrada num heading
acumulado; stick centralizado mantém o heading atual, não volta ao zero.

### Segurança

- **Failsafe de inclinação a 80°.** Acima disso os motores desarmam e ficam
  **travados**: só reiniciando fisicamente o drone. Isso protege a zona singular
  1/cos(pitch), onde a matriz do sistema explode.
- **Perda de conexão desarma.** Sem comandos, os motores param.
- **Bateria baixa apenas acende o LED**, não desarma em voo — desarmar no ar seria
  pior que pousar com pouca carga.

### Configurar o firmware

No topo de `src/main.cpp`:

```cpp
const bool DEBUG_MODE       = false; // true: prints detalhados de profiling
const bool PRINT_TELEMETRY  = false; // true: stream contínuo roll,pitch,yaw,p,q,r
const bool USE_MAGNETOMETER = false; // true: 9-DOF; false: 6-DOF
const int  CONTROLLER_TYPE  = 0;     // 0 = SDRE, 1 = PID (para comparação)
const bool USE_ASYNC_SDRE   = false; // ver o aviso abaixo
```

> **`USE_ASYNC_SDRE` — deixe em `false`.** Em `true`, a Riccati vai para uma task
> separada do FreeRTOS e o loop cai para 5 ms. Parece melhor, mas descasa o `dt`: a
> discretização (Ad, Bd) e o filtro Madgwick assumem o período do loop, enquanto o
> ganho chega de um solve que levou outro tempo. O modo síncrono a 6,0 ms é o que
> está caracterizado e é o que gerou todos os dados do artigo.

**Pesos Q e R** vêm da regra de Bryson sobre um envelope realista de voo
(`src/main.cpp:127-141`): `Q_ii = 1/(máximo do estado i)²`, com 45° de roll/pitch,
90° de yaw, 300°/s de p e q, 200°/s de r. Alargar o envelope reduz `Q_ang` e melhora
a razão proporcional/derivativa.

### Telemetria

Um buffer circular de **800 amostras** em RAM grava, a cada 4 ciclos (≈24 ms, ~19 s
de voo): atitude medida e de referência, taxas do corpo, os três torques do SDRE e
os quatro ω² dos motores. Ao desarmar, é salvo em LittleFS e recarregado no boot.

Com o drone **desarmado**, pelo monitor serial:

| Tecla | O que faz |
|---|---|
| `D` | despeja o buffer em CSV pela serial |
| `R` | zera o buffer (e persiste o reset) |

### Ferramentas de bancada

Em `test/bench/`, cada uma com seu env:

```bash
pio run -e i2c_scan -t upload -t monitor            # a IMU está no barramento?
pio run -e verify_gains -t upload -t monitor        # confere K calculado a bordo
pio run -e verify_gains_flight -t upload -t monitor # idem, com o firmware de voo
```

As demais (`calibrate_mpu.cpp`, `calibrate_magnetometer.cpp`,
`motor_calibration_test.cpp`, `pwm_freq_sweep.cpp`, `identify_inertia.cpp`,
`test_filter_plotter.cpp`, `wifi_simple_test.cpp`) estão na mesma pasta mas **não têm
env** — para usá-las, copie um dos blocos acima no `platformio.ini` trocando o
`build_src_filter`. Guias em `docs/MOTOR_CALIBRATION_GUIDE.md`,
`docs/LED_BATTERY_GUIDE.md` e `docs/QUICK_TEST.md`.

---

## Parte 2 — Medir os solvers

### Os doze solvers

Todos em `lib/AUTOLQR/`, cada um com uma versão `float` e uma `_Fixed` (Q13.18):

| Solver | O que faz de diferente | Iterações |
|---|---|---|
| **SDA** | duplicação preservadora de estrutura, a referência (Chu et al. 2004) | 9 |
| **SDA-SS** | aplica um deslocamento real γ ao pencil simplético (γ=0,7) | **7** |
| **ASDA** | reescalona (G,H)→(sG,H/s) a cada iteração | 9 |
| **SDA-Scaled** | balanceamento diagonal a partir das normas de linha de A | 9 |
| **ADDA** | inverte (I+GH) e (I+HG) separadamente | 9 |
| **Value iteration** | iteração de valor clássica, convergência **linear** | **1 a 32** |

O firmware de voo usa **SDA-fx** (`computeGains()` chama `"SDA_FIXED"` por padrão).

Cada chamada devolve um de três desfechos, e a distinção importa: `Converged`,
`Budget` (esgotou o orçamento de iterações — **não** é falha numérica) e `Breakdown`
(overflow ou pivô singular — falha real). Tratar os dois últimos como a mesma coisa
inflava artificialmente a contagem de falhas da iteração de valor.

### Rodar a campanha inteira

```bash
python python/run_experiments.py --all
```

São 10 experimentos (~10 h no total), mais a análise, as figuras e o PDF do artigo.
É **retomável**: se parar no meio, rodar de novo pula o que já foi capturado. Guia
completo em [`experiments/README.md`](experiments/README.md).

Um experimento por vez:

```bash
python python/run_experiments.py --only bateria --force --no-pdf
python python/run_experiments.py --only voo --repeat 10 --port COM3
```

### Os oito experimentos

| Chave | Firmware | O que mede | Tempo |
|---|---|---|---|
| `bateria` | `benchmark_solvers.cpp` | os 12 solvers × 60000 pontos: tempo, iterações, resíduo, ganho — é a **Tabela 1** | 91 min |
| `benchmark_s3` | idem, no ESP32-**S3** | a mesma bateria numa placa **com** FPU — é a **Tabela 2** | 25 min |
| `tol_qr` | `tol_qr_sweep.cpp` | 6 τ × 13 escalas de R × 5 de Q × 12 solvers (1,4 M chamadas) | 4 h 40 |
| `fronteiras` | `boundary_fine.cpp` | 40 escalas de R × 5 de Q: onde exatamente o ponto fixo quebra | 83 min |
| `repetibilidade` | `repeatability.cpp` | 20 repetições no mesmo dado: quanto é ruído de máquina | 58 min |
| `sweep_qr` | `sweep_qr.cpp` | mapa grosso de segurança em Q/R, com o pico de magnitude interna | 27 min |
| `tolerancia` | `tolerance_sweep.cpp` | τ de 10⁻² a 10⁻⁶: o que apertar a tolerância compra e custa | 20 min |
| `gamma` | `gamma_sweep.cpp` | escolha do deslocamento γ do SDA-SS, numa grade de 5 valores | 5 min |
| `norma` | `norm_benchmark.cpp` | custo do teste de convergência, ciclo a ciclo | 2 min |
| `voo` | `src/main.cpp` | o ciclo de controle completo, IMU ligada e motores parados | 6 min/janela |

> O experimento `voo` liga `DEBUG_MODE=true` automaticamente, grava, captura e
> **restaura** o firmware limpo ao terminar. Use `--repeat N` para N janelas — elas
> são numeradas a partir do próximo índice livre e **nunca** sobrescrevem uma
> existente. É o único experimento não determinístico da campanha: a mediana do ciclo
> repete, mas a cauda muda entre janelas.

### Cada captura diz de onde veio

Todo firmware de experimento emite, como primeira linha:

```
STAMP,<commit>,<sujo>,<epoch>,<compilado>,<chip>,<rev>,<núcleos>,<MHz>
```

Sem isso, um `.txt` em `outputs/` é indistinguível de outro gerado meses antes por
outro código — foi assim que uma medição anterior a uma otimização sobreviveu em duas
versões do artigo reportando 12,17 ms onde o código atual mede 9,62 ms.

---

## Parte 3 — Analisar e auditar

### Um comando confere tudo

```bash
python python/auditoria.py
```

```
procedencia    OK      de qual commit, chip e clock veio cada captura
numeros        OK      243 checagens contra o dado bruto, 0 divergencias
figuras        OK      as figuras do artigo saíram do dado de hoje
voo            OK      10 janelas, 475120 ciclos
cobertura      194 de 219 numeros do corpo do artigo
```

A linha de **cobertura** é a que evita auto-engano: ela conta quantos números do
artigo têm checagem automática e **imprime os que não têm**. Se alguém acrescentar
uma afirmação sem teste, o número cai. Detalhes em
[`outputs/v8/auditoria_v8.md`](outputs/v8/auditoria_v8.md).

Etapas isoladas, se quiser só uma:

```bash
python python/verifica_procedencia.py     # carimbos das capturas
python python/verifica_numeros_artigo.py --v8
python python/verifica_figuras.py         # figuras × dado
python python/analisa_voo.py --dir outputs/voo
```

### Gerar figuras, tabelas e derivados

```bash
python python/figuras_artigo_final.py --flight-dir outputs/voo
python python/numeros_artigo.py --tabelas    # recalcula as Tabelas 1 e 2 do dado bruto
python python/generate_table2.py             # Tabela 2 (S2 vs S3) em LaTeX e CSV
python python/parse_memory_map.py            # IRAM, RAM, Flash e pilha do firmware
python python/derivados_artigo.py            # resíduo da referência e fidelidade do modelo
```

Cada PDF de figura carrega nos metadados o commit e o SHA-256 de cada captura que
entrou nela — é o que permite `verifica_figuras.py` detectar figura desatualizada,
coisa que o LaTeX nunca acusaria.

### Pacote para revisores

```bash
python python/exporta_dados_revisores.py
```

Gera `zenodo_diname2027/` com as capturas seriais convertidas em CSV tabular, as
tabelas, as métricas de memória, o ciclo de voo janela a janela e um `MANIFEST.md`
com o hash SHA-256 de cada arquivo de origem e o commit que os produziu.

### Regressão no host (sem hardware)

```bash
bash test/native/build.sh      # compila e roda verify_solvers.cpp
```

Compara os solvers contra `scipy.linalg.solve_discrete_are` em dupla precisão.

---

## Mapa do repositório

```
src/main.cpp          firmware de voo — 167 Hz, SDRE-LQR em Q13.18
experiments/          8 firmwares da campanha (o benchmark roda em S2 e S3)
lib/
  AUTOLQR/            os 12 solvers de Riccati + o kernel Q13.18
  Trajectories/       as 6 trajetórias T1-T6 dos pontos de operação
  BiquadFilter/       filtro de 8 Hz nas taxas
  Telemetry/          buffer circular em RAM + persistência em LittleFS
  MotorControl/       mistura X-quad e PWM
  WiFiComm/           ponto de acesso + protocolo CRTP
  PIDController/      controlador alternativo, para comparação
  KalmanFilter/       estimador alternativo
  OpticalFlow/        sensor PMW3901 (fora do caminho de atuação)
  utils/              inicialização de I²C, LEDs, bateria
test/
  native/             regressão dos solvers no host
  bench/              ferramentas de bancada (3 com env)
python/               captura serial, análise, auditoria, figuras
outputs/              capturas seriais brutas (não versionadas)
  voo/                as N janelas do ciclo de voo
  v8/                 derivados da auditoria
docs/                 auditoria dos solvers, guias de hardware
archive/              código de fases anteriores, fora do caminho principal
zenodo_diname2027/    pacote de dados para revisores (regenerável)
```

**Os quatro documentos que valem a leitura:**

| Arquivo | Sobre |
|---|---|
| [`experiments/README.md`](experiments/README.md) | como reproduzir a campanha do zero |
| [`docs/auditoria_solvers_riccati.md`](docs/auditoria_solvers_riccati.md) | por que cada solver é o que é, e os erros já corrigidos |
| [`outputs/v8/auditoria_v8.md`](outputs/v8/auditoria_v8.md) | cobertura de verificação, número a número |
| [`archive/README.md`](archive/README.md) | o que saiu do caminho principal e por quê |

---

## Todos os envs do PlatformIO

```bash
pio run -e <nome> -t upload -t monitor
```

| Env | Compila | Para quê |
|---|---|---|
| `esp32-s2-saola-1` | `src/main.cpp` | **o firmware de voo** |
| `benchmark` | `experiments/benchmark_solvers.cpp` | bateria principal (Tabela 1) |
| `benchmark_s3` | idem, alvo ESP32-S3 | comparação com FPU (Tabela 2) |
| `tolerance_sweep` | `experiments/tolerance_sweep.cpp` | varredura de τ |
| `tol_qr_sweep` | `experiments/tol_qr_sweep.cpp` | τ cruzado com pesos |
| `sweep_qr` | `experiments/sweep_qr.cpp` | mapa grosso de segurança Q/R |
| `boundary_fine` | `experiments/boundary_fine.cpp` | fronteira fina de quebra |
| `gamma_sweep` | `experiments/gamma_sweep.cpp` | escolha do γ do SDA-SS |
| `repeatability` | `experiments/repeatability.cpp` | jitter de execução |
| `norm_benchmark` | `experiments/norm_benchmark.cpp` | custo do teste de convergência |
| `i2c_scan` | `test/bench/i2c_scan.cpp` | achar a IMU no barramento |
| `verify_gains` | `test/bench/verify_gains_onboard.cpp` | conferir K a bordo |
| `verify_gains_flight` | idem | conferir K com o firmware de voo |

---

## Onde procurar quando algo dá errado

| Sintoma | Causa provável |
|---|---|
| firmware trava no boot, nada na serial | **IMU não responde.** O firmware para em `while(1)` de propósito (`lib/utils/utils.cpp:19-24`). Rode `pio run -e i2c_scan -t upload -t monitor`. |
| motores não armam | Nenhum comando chegou ainda — o primeiro comando é que arma. Confira se o app conectou à rede `ESP-DRONE`. |
| motores desarmaram e não rearmam | **Failsafe de inclinação** (acima de 80°). Só reiniciando fisicamente o drone. |
| captura serial com poucos bytes | O firmware travou — quase sempre a IMU. As janelas de voo inválidas vão para `outputs/voo/invalidas/` com o motivo no nome. |
| `verifica_procedencia.py` reprova | Alguma captura veio de outro commit, de outro chip ou fora dos 240 MHz. A tabela diz qual. |
| `verifica_figuras.py` acusa dessincronia | Uma figura no artigo é mais velha que o dado. Regere com `figuras_artigo_final.py --flight-dir outputs/voo`. |
| loop de voo estourando o período | Confira se `USE_ASYNC_SDRE` está em `false` e se o `DEBUG_MODE` não ficou ligado. |

---

## Requisitos

- **PlatformIO** (CLI ou extensão do VS Code)
- **Python 3.10+** com `pip install -r requirements.txt`
- **MiKTeX** ou outra distribuição LaTeX, apenas para recompilar o PDF do artigo

## Licença

| O que | Licença | Arquivo |
|---|---|---|
| Software — `src/`, `experiments/`, `lib/`, `python/`, `scripts/`, `test/` | **MIT** | [`LICENSE`](LICENSE) |
| Dados de medição — capturas seriais, CSVs derivados e figuras | **CC BY 4.0** | [`LICENSE-DATA`](LICENSE-DATA) |

Dependências de terceiros mantêm cada uma a sua própria licença; nenhuma é
redistribuída aqui.

## Como citar

O repositório traz um [`CITATION.cff`](CITATION.cff) — no GitHub, use o botão
*Cite this repository*. Ou cite o depósito diretamente:

> Bassani, G. A. A.; Cardoso, R.; Correa, D. P. F. (2026).
> *Quadrotor Attitude Control by State-Dependent Riccati Equation on FPU-less
> Hardware: data and code.* Zenodo. <https://doi.org/10.5281/zenodo.22256810>

## Publicação

Os dados sustentam o artigo aceito no **DINAME 2027** (ABCM):
*Quadrotor Attitude Control by State-Dependent Riccati Equation on FPU-less Hardware:
Rigid-Body Dynamics, Quantisation Limits and Cost Predictability.*
