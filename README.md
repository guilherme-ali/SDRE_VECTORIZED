# SDRE_VECTORIZED

Controle de atitude SDRE (*State-Dependent Riccati Equation*) em ponto fixo Q13.18 para
microcontroladores sem FPU (ESP32-S2), com solvers *structure-preserving doubling* e comparação com ESP32-S3 (com FPU).
Base de dados, firmwares e pipeline experimental do artigo aceito no **DINAME 2027** (ABCM).

## Três pontos de entrada

1. **Firmware de voo (167 Hz / 6.0 ms)** — `src/main.cpp` (env `esp32-s2-saola-1`).
   `pio run -e esp32-s2-saola-1 -t upload --upload-port COMx`

2. **Campanha de medição automatizada** — os 8 firmwares em `experiments/` (incluindo ESP32-S3 e microbenchmark de ciclos da norma), orquestrados por `python/run_experiments.py`.
   Guia completo: [`experiments/README.md`](experiments/README.md).
   ```bash
   pip install -r requirements.txt
   python python/run_experiments.py --all
   ```

3. **Auditoria e Figuras do Artigo** — scripts em `python/` para validação claim-by-claim de todas as afirmações numéricas do artigo e geração das figuras vetoriais:
   ```bash
   # Validação exata de todos os números do artigo contra os dados brutos
   # (--v8 confere também a Tabela 2 com Value iteration e a fração de passos bit-exatos):
   python python/verifica_numeros_artigo.py --v8

   # Geração das figuras vetoriais da publicação (Fig 1, 3, 4 [v7 e v8 de 3 painéis], 5, 6):
   python python/figuras_artigo_final.py --outdir "G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\artigo_diname\Figures"

   # Tabela 2 (S2/S3 x float/fixed) incluindo o Value iteration, ausente da tabela do v7:
   python python/generate_table2.py

   # IRAM/RAM/Flash do firmware de voo a partir do linker map:
   python python/parse_memory_map.py

   # Recalcula do dado bruto TODOS os numeros das tabelas (e emite o LaTeX):
   python python/numeros_artigo.py --tabelas

   # Procedencia: toda captura veio do mesmo commit, do chip certo, a 240 MHz?
   python python/verifica_procedencia.py
   ```

   O ultimo e' o guarda-costas da campanha. Cada firmware de experimento emite,
   como primeira linha da captura, um carimbo
   `STAMP,<commit>,<dirty>,<build_epoch>,<compilado>,<chip>,<rev>,<nucleos>,<mhz>`
   (`scripts/build_stamp.py` + `include/BuildStamp.h`). Sem ele, um `.txt` em
   `outputs/` e' indistinguivel de um gerado meses antes por outro codigo — foi
   assim que uma captura anterior a otimizacao push-through do ADDA sobreviveu
   reportando 12.17 ms onde o codigo atual mede 9.62 ms.

## Pacote de Replicação Zenodo

A pasta `zenodo_diname2027/` contém o pacote pronto para depósito com DOI no Zenodo: CSVs
tabulares derivados das capturas seriais brutas, a Tabela 2 e as métricas de memória, com
hash SHA-256 de cada arquivo de origem em `MANIFEST.md`. Gerar/atualizar com:
```bash
python python/exporta_dados_revisores.py
```

## Estrutura do Repositório

```
src/                 firmware de voo em tempo real (Ts = 6.0 ms, histograma de 50 µs em RAM)
experiments/         firmwares da campanha experimental (S2, S3 com FPU, sweeps de tau e Q/R)
lib/                 bibliotecas C++ (AUTOLQR Q13.18, Trajectories T1-T6, BiquadFilter 8Hz, Telemetry, ...)
python/              scripts de análise, auditoria automatizada e geração das figuras
outputs/             capturas seriais brutas (60k pontos, 47.802 ciclos de voo, sweep 1.4M chamadas)
docs/                documentos de auditoria, guia de hardware e rastreabilidade
```
