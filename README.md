# SDRE_VECTORIZED

Controle de atitude SDRE (State-Dependent Riccati Equation) em ponto fixo Q13.18 para
ESP32-S2 (sem FPU), com solvers do tipo *structure-preserving doubling algorithm*.
Base experimental do artigo DINAME 2027.

## Três pontos de entrada

1. **Firmware de voo** — `src/main.cpp` (env `esp32-s2-saola-1`).
   `pio run -e esp32-s2-saola-1 -t upload --upload-port COMx`

2. **Campanha de medição** — os 7 firmwares em `experiments/` mais o experimento de
   ciclo de voo, orquestrados por `python/run_experiments.py`.
   Guia completo: [`experiments/README.md`](experiments/README.md).
   ```
   pip install -r requirements.txt
   python python/run_experiments.py --all
   ```

3. **Análise/figuras/artigo** — scripts em `python/` (`analisa_*.py`,
   `figuras_artigo_final.py`, `gerar_relatorio_bateria.py`), consumidos automaticamente
   pelo runner acima, ou chamáveis isoladamente sobre os dados já capturados em
   `outputs/`.

## Estrutura

```
src/            firmware de voo (o único que compila por padrão: pio run)
experiments/    firmwares da campanha de medição + README de instruções
test/bench/     utilitários de bancada (calibração, diagnóstico de fiação, etc.)
test/native/    harness de regressão no host (g++, sem hardware)
test/archive/   firmwares obsoletos, mantidos por referência
lib/            bibliotecas do projeto (AUTOLQR, Telemetry, Trajectories, ...)
python/         scripts de análise, figuras e o orquestrador run_experiments.py
python/legacy/  scripts de linhas de trabalho anteriores, não usados no pipeline atual
outputs/        capturas seriais e dados gerados (fora do git); archive/ = superados
docs/           notas de auditoria e decisões de projeto
```
