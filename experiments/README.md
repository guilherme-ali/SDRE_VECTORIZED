# Campanha de medição — guia de uso

Como reproduzir, do zero, todos os dados que alimentam o artigo DINAME 2027: conectar o
ESP32-S2 no PC, rodar um comando, e obter os resultados prontos para análise (figuras +
`RESULTS.md`), até a recompilação do PDF do artigo.

## TL;DR

```
pip install -r requirements.txt
python python/run_experiments.py --all
```

Isso roda os 8 experimentos (~9,5 h no total), a análise de host, as 6 figuras do
artigo e recompila o PDF. É retomável: se parar no meio, rodar de novo pula o que já
foi capturado.

## Pré-requisitos

- [PlatformIO Core](https://platformio.org/install/cli) instalado e no PATH (`pio`).
- Python 3.10+ com as dependências de `requirements.txt`.
- ESP32-S2 Saola conectado por USB (driver CP210x instalado, se necessário).
- **Somente para o experimento `voo`**: a IMU (MPU6050) precisa estar fisicamente
  conectada e endereçada corretamente — sem ela o firmware trava em `while(1)`
  (`lib/utils/utils.cpp:19-24`) e a captura nunca progride. Os motores permanecem
  desarmados durante toda a medição; nada gira.
- **Para o passo final de PDF**: MiKTeX (ou outra distribuição TeX com `pdflatex` e
  `bibtex` no PATH). Sem isso o runner avisa e para antes do PDF (`--no-pdf` evita o
  aviso deliberadamente).

## O que cada experimento mede

| chave | firmware / env | pergunta científica | saída | ~duração |
|---|---|---|---|---|
| `tolerancia` | `experiments/tolerance_sweep.cpp` | qual tolerância/orçamento de iterações é comparável entre float e ponto fixo? (Exp. 0) | `serial_tolerance_sweep_frobenius.txt` | 20 min |
| `gamma` | `experiments/gamma_sweep.cpp` | qual γ do SDA-SS minimiza iterações sob o critério casado? (Exp. 3) | `serial_gamma_sweep.txt` | 5 min |
| `sweep_qr` | `experiments/sweep_qr.cpp` | onde os solvers `_FIXED` estouram o teto ±8192 do Q13.18 em função da escala de Q/R? | `serial_sweep_qr_v4.txt` | 27 min |
| `tol_qr` | `experiments/tol_qr_sweep.cpp` | τ=1e-3 é robusto em toda a banda segura de pesos? (Exp. A, o carro-chefe) | `serial_tol_qr_sweep_A.txt` | 280 min |
| `fronteiras` | `experiments/boundary_fine.cpp` | mapa fino das duas transições de falha (Exp. B) | `serial_boundary_fine_B.txt` | 83 min |
| `repetibilidade` | `experiments/repeatability.cpp` | variância de tempo no mesmo ponto, 20 repetições (Exp. D) | `serial_repeatability_D.txt` | 58 min |
| `bateria` | `experiments/benchmark_solvers.cpp` | tempo/erro/convergência dos 12 métodos em 60000 pontos, 6 trajetórias (Exp. 1) | `serial_capture_bateria_v5_6traj.txt` | ~90 min |
| `benchmark_s3` | `experiments/benchmark_solvers.cpp` (env `benchmark_s3`, ESP32-S3 com FPU) | mesma bateria no S3, p/ compor a Tabela 2 (S2 vs S3, incl. Value iteration) | `s3/serial_capture_bateria_s3.txt` | ~25 min |
| `voo` | `src/main.cpp` (`DEBUG_MODE=true` temporário) | limite prático de frequência do ciclo completo de voo (Exp. E) | `serial_flightloop_E.txt` | 6 min |

Cada linha corresponde a uma seção de resultados do artigo (`docs/auditoria_solvers_riccati.md`
tem o detalhamento técnico de cada critério).

## Receitas

**Validar o setup em ~5 min** (compila, grava, captura, analisa um experimento curto):
```
python python/run_experiments.py --only gamma --force --no-pdf
```

**Campanha completa** (~9,5 h; retomável — pode interromper e continuar depois):
```
python python/run_experiments.py --all
```

**Só reanalisar** dados já capturados, sem tocar no hardware — o teste mais forte de
que a reorganização não quebrou nada, pois reproduz os números publicados do artigo
puramente a partir de `outputs/`:
```
python python/run_experiments.py --analyze-only
```

**Um experimento isolado**, pulando o PDF:
```
python python/run_experiments.py --only tol_qr --no-pdf
```

**Rodar tudo, exceto o de voo** (por exemplo, sem a IMU disponível no momento):
```
python python/run_experiments.py --all --skip voo
```

**Ver o que seria feito sem tocar em nada**:
```
python python/run_experiments.py --all --dry-run
```

Flags completas: `python python/run_experiments.py --help`.

## Como ler os resultados

- `outputs/RESULTS.md` — status de cada experimento (OK/pulado/falhou), duração real, e
  o status das fases de host (malha fechada, cobertura, relatório da bateria, figuras,
  PDF).
- `outputs/run_<timestamp>.log` — log completo da execução (todo output de `pio` e dos
  scripts de análise).
- As capturas seriais (`outputs/serial_*.txt`) começam com linhas `#` de progresso
  (ecoadas ao vivo durante a captura) e terminam com uma linha `# FIM DE ...` — o
  marcador que o runner usa para saber que o experimento terminou. Linhas `EXP,...` são
  dados em CSV, uma amostra por linha.
- As 6 figuras do artigo são escritas diretamente em
  `G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\artigo_diname\Figures\` por
  `python/figuras_artigo_final.py`.
- `python/gerar_relatorio_bateria.py` escreve `outputs/relatorio_bateria.md` (tempos,
  erro de K, custo em malha fechada) a partir da bateria + malha fechada.

## Solução de problemas

- **Porta não detectada / ambígua**: passe explicitamente com `--port COM3` (Windows)
  ou `--port /dev/ttyUSBx` (Linux). O erro lista as portas vistas pelo sistema.
- **Porta ocupada**: feche o Serial Monitor do PlatformIO/Arduino IDE ou qualquer outro
  processo que já esteja lendo a porta antes de rodar o runner.
- **Falha de gravação (`upload`)**: confirme que a placa está em modo de boot correto;
  o ESP32-S2 Saola normalmente entra sozinho via o circuito CP210x, mas se travar,
  segure BOOT e pressione RESET manualmente.
- **Captura travada (sem marcador em muito tempo)**: o runner usa um teto de 3× a
  duração estimada antes de desistir e marcar o experimento como falho. Um teto
  atingido geralmente indica firmware travado (ex.: `voo` sem IMU conectada) — veja o
  log ao vivo impresso durante a captura.
- **MiKTeX ausente**: o passo de PDF degrada com aviso em vez de derrubar a execução;
  rode com `--no-pdf` para não tentar.
- **`pio` não encontrado**: instale o PlatformIO Core e confirme que está no PATH
  (`pio --version`).

## Reprodutibilidade

As 6 trajetórias de referência (`python/trajetorias.py`) são geradas em forma fechada,
sem número aleatório — a mesma trajetória, a cada execução, byte a byte. Combinado com
firmware determinístico (sem *watchdog jitter* relevante na escala medida), a campanha é
reproduzível no mesmo hardware: recapturar deve reproduzir os mesmos resultados dentro
do ruído de tempo do sistema operacional/USB.
