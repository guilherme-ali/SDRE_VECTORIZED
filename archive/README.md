# archive/ — fora do caminho principal, nada apagado

Código e saídas que não participam da cadeia ativa da campanha DINAME 2027
(firmware → captura → análise → figuras → artigo). Ficam aqui porque têm valor
histórico ou de bancada, não porque estejam errados. Nenhum script de `python/`
importa daqui.

Movidos em 2026-09-01, quando `python/` tinha 27 arquivos na raiz mais 46 em seis
subpastas e ficara difícil dizer o que era caminho principal e o que era resíduo.

| Pasta | Origem | O que é |
|---|---|---|
| `python_legacy/` | `python/legacy/` | Ferramentas de uma fase anterior do mestrado: simulação de atitude, identificação de inércia por OLS, extração de anotações de PDF, plot de telemetria, geração de figuras da versão antiga do artigo. |
| `python_matriz_otima/` | `python/matriz_otima/` | Busca de pesos Q/R "ótimos" para o SDRE e verificação da matriz nos domínios contínuo e discreto. Substituída pela varredura Q/R em hardware (`experiments/sweep_qr.cpp`). |
| `python_execucao_otima/` | `python/execucao_otima/` | Estudo de execução ótima, um arquivo. |
| `python_simulador/` | `python/simulador/` | Simulador antigo, um arquivo. |
| `python_outputs_antigos/` | `python/outputs/` | Saídas de execuções antigas que ficaram dentro de `python/` em vez de `outputs/`. |
| `test_archive/` | `test/archive/` | `main_backup.cpp` (a discretização analítica de 2ª ordem que `python/bench_trajetorias.py` espelha e cita), matriz de execução, benchmark de dados e teste do solver DARE. |

## O que continua ativo

- `src/` — firmware de voo (env `esp32-s2-saola-1`)
- `experiments/` — os oito firmwares da campanha (nove envs: benchmark roda em S2 e S3)
- `test/bench/` — ferramentas de bancada (tres com env: i2c_scan, verify_gains, verify_gains_flight)
- `lib/` — AUTOLQR (Q13.18), Trajectories, filtros, telemetria
- `python/` — captura, análise, auditoria e figuras
- `test/native/` — regressão dos solvers no host
