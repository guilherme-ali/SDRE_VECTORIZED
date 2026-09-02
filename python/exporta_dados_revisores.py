"""
Exporta as capturas seriais brutas (formato CSV ad-hoc, sem cabecalho, pensado
para o firmware) em CSVs tabulares com cabecalho, prontos para acompanhar o
depósito Zenodo do artigo — a pasta `zenodo_diname2027/` que o README.md já
anuncia ("Pacote de Replicação Zenodo") mas que não existe no disco no
momento em que este script foi escrito.

Nao move nem apaga nenhuma captura bruta: os CSVs sao derivados, os arquivos
de outputs/ permanecem a fonte de verdade.

Gera em zenodo_diname2027/:
  tolerance_sweep_runs.csv   uma linha por chamada do Exp. 0 (tau sweep),
                             com rel_step/bit_exact quando presentes
  battery_s2_runs.csv        uma linha por chamada da bateria principal (ESP32-S2)
  battery_s3_runs.csv        idem, ESP32-S3 (se a captura existir)
  table2.csv                 copia de outputs/v8/tabela2_v8.csv (gerar antes com
                             python/generate_table2.py)
  memory_map.csv             metricas de memoria achatadas (gerar antes com
                             python/parse_memory_map.py)
  MANIFEST.md                hash SHA-256 de cada arquivo de origem + commit git

Uso:
    python python/exporta_dados_revisores.py
"""
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
ZENODO = os.path.join(REPO, "zenodo_diname2027")
# O deposito e' dividido em tres para que o revisor saiba o que e' medida e o que
# e' derivacao: raw/ e' o que saiu da placa, derived/ e' o que se calcula dele, e
# code/ e' o programa que fez a conta.
# DOI reservado no Zenodo para este deposito. E' a mesma string que aparece na
# secao "Data and Code Availability" do artigo; python/auditoria.py confere que
# o .tex nao voltou a carregar um marcador no lugar dela.
DOI = "10.5281/zenodo.22256810"
# DOI conceito: resolve sempre para a versao mais recente do deposito. A v1
# (zenodo.22236199) foi criada pela integracao automatica GitHub->Zenodo e
# trazia so' o zip do repositorio, sem as capturas brutas -- ver README do
# deposito.
DOI_CONCEITO = "10.5281/zenodo.22236198"

RAW = os.path.join(ZENODO, "raw")
DERIVED = os.path.join(ZENODO, "derived")
CODE = os.path.join(ZENODO, "code")

SOURCES = {
    "tolerance_sweep": os.path.join(OUT, "serial_tolerance_sweep_frobenius.txt"),
    "battery_s2": os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt"),
    "battery_s3": os.path.join(OUT, "s3", "serial_capture_bateria_s3.txt"),
    "table2_csv": os.path.join(OUT, "v8", "tabela2_v8.csv"),
    "memory_json": os.path.join(OUT, "v8", "memoria_v8.json"),
    "closed_loop": os.path.join(OUT, "malha_fechada_v6_6traj.csv"),
    "gain_schedule": os.path.join(OUT, "ganho_congelado_6traj.csv"),
    "coverage": os.path.join(OUT, "cobertura_full_v5_6traj.csv"),
    # numeros do artigo que so' passaram a ter script na auditoria v8
    "reference_residual": os.path.join(OUT, "v8", "residuo_referencia.csv"),
    "discretisation_fidelity": os.path.join(OUT, "v8", "fidelidade_discretizacao.csv"),
}

# Series temporais de malha fechada: as seis existem em outputs/ (o script
# anterior apontava para um "malha_fechada_serie_T2.csv" que nao existe, e a
# figura do artigo usa T3). Descobertas por glob para nao envelhecer de novo.
SERIES_GLOB = os.path.join(OUT, "malha_fechada_serie_*.csv")

# Ciclo de voo: N execucoes do mesmo binario. E' o unico experimento nao
# deterministico da campanha, entao o pacote leva as N, e nao a que calhou.
VOO_GLOB = os.path.join(OUT, "voo", "voo_run*.txt")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "desconhecido (git rev-parse falhou)"


def export_tolerance_sweep(src, dst):
    if not os.path.isfile(src):
        print("  pulado (nao encontrado): %s" % src)
        return None
    n = 0
    with open(src, encoding="utf-8", errors="replace") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["exp", "tol", "traj", "k", "metodo", "time_us", "iters",
                    "residuo_dare", "outcome", "rel_step", "bit_exact"])
        for line in fin:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 10:
                continue
            exp, tol, traj, k, metodo, time_us, iters, resid, outcome = p[1:10]
            rel_step = p[10] if len(p) >= 11 else ""
            bit_exact = p[11] if len(p) >= 12 else ""
            w.writerow([exp, tol, traj, k, metodo, time_us, iters, resid, outcome,
                        rel_step, bit_exact])
            n += 1
    print("  %s: %d linhas -> %s" % (os.path.basename(src), n, dst))
    return n


def export_battery(src, dst):
    if not os.path.isfile(src):
        print("  pulado (nao encontrado): %s" % src)
        return None
    n = 0
    with open(src, encoding="utf-8", errors="replace") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["traj", "k", "metodo", "time_us", "iters", "residuo_dare",
                    "outcome", "rel_step", "bit_exact"])
        for line in fin:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 8:
                continue
            traj, k, metodo, time_us, iters, resid, outcome = p[1:8]
            rel_step = p[8] if len(p) >= 9 else ""
            bit_exact = p[9] if len(p) >= 10 else ""
            w.writerow([traj, k, metodo, time_us, iters, resid, outcome,
                        rel_step, bit_exact])
            n += 1
    print("  %s: %d linhas -> %s" % (os.path.basename(src), n, dst))
    return n


def export_memory_csv(src_json, dst_csv):
    if not os.path.isfile(src_json):
        print("  pulado (nao encontrado, rodar python/parse_memory_map.py antes): %s" % src_json)
        return None
    with open(src_json, encoding="utf-8") as f:
        d = json.load(f)
    rows = [
        ("map_file", d["map_file"]),
        ("env", d["env"]),
        ("iram_q13_18_code_bytes", d["iram_q13_18"]["code_bytes"]),
        ("iram_q13_18_literal_bytes", d["iram_q13_18"]["literal_bytes"]),
        ("ram_data_bytes", d["ram"]["data_bytes"]),
        ("ram_bss_bytes", d["ram"]["bss_bytes"]),
        ("ram_used_bytes", d["ram"]["used_bytes"]),
        ("ram_total_kb", d["ram"]["total_kb"]),
        ("ram_pct", d["ram"]["pct"]),
        ("flash_text_bytes", d["flash"]["text_bytes"]),
        ("flash_rodata_bytes", d["flash"]["rodata_bytes"]),
        ("flash_used_bytes", d["flash"]["used_bytes"]),
        ("flash_partition_kb", d["flash"]["partition_kb"]),
        ("flash_pct", d["flash"]["pct"]),
    ]
    with open(dst_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    print("  %s -> %s" % (os.path.basename(src_json), dst_csv))
    return len(rows)


def export_flight(paths, dst_runs, dst_hist):
    """Uma linha por execucao + o histograma de 50 us de cada uma, lado a lado.

    O laco de voo nao repete: as mesmas 360 s do mesmo binario deram de 0 a 6
    ciclos acima do periodo. O revisor precisa das N execucoes para ver isso,
    nao de uma media que esconde a dispersao.
    """
    import re
    import statistics as stat

    ESTAGIOS = [(r"LQR .Ganhos.", "dare_solve_us"), ("Leitura MPU", "imu_read_us"),
                ("Matriz Sistema", "sdc_matrix_us"), (r"C.lc. .ngulos", "euler_us")]
    linhas, hists = [], []
    for path in paths:
        txt = open(path, encoding="utf-8", errors="replace").read()
        blocos = txt.split("STATUS DO SISTEMA")
        if len(blocos) < 2:
            continue
        m = re.search(r"HIST_PROC_50US:([0-9,]+)", blocos[-1])
        if not m:
            continue
        h = [int(x) for x in m.group(1).split(",") if x]
        tot = sum(h)
        acc, p50, p99, p999 = 0, None, None, None
        for i, c in enumerate(h):
            acc += c
            f = 100.0 * acc / tot
            if p50 is None and f >= 50:
                p50 = i * 50 / 1000.0
            if p99 is None and f >= 99:
                p99 = i * 50 / 1000.0
            if p999 is None and f >= 99.9:
                p999 = i * 50 / 1000.0
        mx = re.search(r"Processamento_Maximo:\s*(\d+)", blocos[-1])
        ml = re.search(r"Tempo_Medio:\s*([\d.]+)", blocos[-1])
        md = re.search(r"Processamento_Medio:\s*([\d.]+)", blocos[-1])
        carimbo = re.search(r"STAMP,([^,\s]+),(\d+),(\d+)", txt)
        row = {
            "run": os.path.splitext(os.path.basename(path))[0],
            "cycles": tot,
            "median_ms": p50, "p99_ms": p99, "p999_ms": p999,
            "max_ms": int(mx.group(1)) / 1e3 if mx else "",
            "mean_ms": float(md.group(1)) / 1e3 if md else "",
            "loop_period_ms": float(ml.group(1)) / 1e3 if ml else "",
            "cycles_over_period": sum(h[120:]),
            "git_rev": carimbo.group(1) if carimbo else "",
            "build_dirty": carimbo.group(2) if carimbo else "",
        }
        for pat, lbl in ESTAGIOS:
            v = [int(mm.group(1)) for b in blocos[1:]
                 for mm in [re.search(pat + r":\s*(\d+)\s*.s", b)] if mm]
            row[lbl] = stat.median(v) if v else ""
        linhas.append(row)
        hists.append((row["run"], h))

    if not linhas:
        print("  pulado (nenhuma captura de voo encontrada)")
        return
    with open(dst_runs, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print("  %s: %d execucoes" % (os.path.basename(dst_runs), len(linhas)))

    n = max(len(h) for _, h in hists)
    with open(dst_hist, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_lower_ms", "bin_upper_ms"] + [r for r, _ in hists] + ["pooled"])
        for i in range(n):
            col = [h[i] if i < len(h) else 0 for _, h in hists]
            w.writerow(["%.2f" % (i * 0.05), "%.2f" % ((i + 1) * 0.05)] + col + [sum(col)])
    print("  %s: %d bins de 50 us" % (os.path.basename(dst_hist), n))


def copia_capturas_brutas():
    """Leva para raw/ as capturas seriais como saíram da placa.

    Sao a medida em si: os CSVs de derived/ sao reconstruíveis a partir delas
    (`python python/exporta_dados_revisores.py`), o contrario nao. Sem elas, o
    SHA-256 que o MANIFEST publica nao e' conferivel por ninguem de fora.
    """
    os.makedirs(RAW, exist_ok=True)
    copiados, total = [], 0
    padroes = [
        (os.path.join(OUT, "serial_*.txt"), ""),
        (os.path.join(OUT, "s3", "*.txt"), "s3"),
        (os.path.join(OUT, "voo", "voo_run*.txt"), "voo"),
    ]
    for padrao, sub in padroes:
        destino_dir = os.path.join(RAW, sub) if sub else RAW
        os.makedirs(destino_dir, exist_ok=True)
        for src in sorted(glob.glob(padrao)):
            nome = os.path.basename(src)
            # backups locais e capturas pre-carimbo nao entram: nao sustentam
            # numero nenhum do artigo e so' confundiriam quem for conferir
            if "_ANTES" in nome or "_SEMCARIMBO" in nome:
                continue
            dst = os.path.join(destino_dir, nome)
            shutil.copyfile(src, dst)
            copiados.append(os.path.relpath(dst, ZENODO).replace(os.sep, "/"))
            total += os.path.getsize(dst)
    print("  raw/: %d capturas brutas (%.1f MB)" % (len(copiados), total / 1e6))
    return copiados


def arquiva_codigo(commit):
    """Snapshot do repositorio no commit exato, via `git archive`.

    O artigo promete "solver implementations, firmware benchmarks and analysis
    scripts" no deposito. Um zip do commit entrega os tres e fixa a versao: o
    GitHub continua sendo o lugar de trabalhar, mas o que gerou ESTES numeros
    fica congelado aqui.
    """
    curto = commit[:7] if commit and commit != "desconhecido" else "sem-commit"
    # direto na raiz do deposito, e nao dentro de CODE/: o git archive JA' e' um
    # zip, e deixa-lo numa pasta fazia empacota_para_upload() zipar de novo. O
    # code.zip da v2 tem, por isso, um zip dentro dele.
    destino = os.path.join(ZENODO, "code.zip")
    try:
        # exclui archive/ (49 MB de xlsx/png de simulacoes antigas, que nao
        # sustentam numero nenhum do artigo) e o proprio pacote, para nao
        # aninhar o deposito dentro de si mesmo
        subprocess.check_call(
            ["git", "archive", "--format=zip", "-o", destino, "HEAD",
             "--", ".", ":(exclude)archive", ":(exclude)zenodo_diname2027"],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("  code/: git archive falhou (%r); snapshot NAO gerado." % (e,))
        return None
    print("  code.zip: snapshot em %s (%.1f MB)"
          % (curto, os.path.getsize(destino) / 1e6))
    return "code.zip"


def escreve_readme(commit, codigo):
    """README do deposito: quem baixa o zip precisa saber o que e' cada arquivo
    sem ter o repositorio aberto ao lado."""
    linhas = []
    A = linhas.append
    A("# Dados e codigo — SDRE em ponto fixo Q13.18 em hardware sem FPU")
    A("")
    A("Pacote de replicacao do artigo submetido ao **DINAME 2027** (ABCM):")
    A("*Comparative Analysis of Numerical Methods for Solving the Discrete Algebraic")
    A("Riccati Equation in Resource-Constrained Embedded Systems*.")
    A("")
    A("**DOI desta versao:** %s" % DOI)
    A("")
    A("**DOI conceito** (sempre a versao mais recente): %s" % DOI_CONCEITO)
    A("")
    A("Para citar este conjunto de dados use o DOI acima; o artigo o referencia na")
    A("secao *Data and Code Availability*.")
    A("")
    A("Gerado a partir do commit `%s` do repositorio" % commit)
    A("<https://github.com/guilherme-ali/SDRE_VECTORIZED>.")
    A("")
    A("## O que tem aqui")
    A("")
    A("| Pasta | O que e' |")
    A("|---|---|")
    A("| `raw/` | As capturas seriais **como saíram da placa**. Sao a medida em si. |")
    A("| `derived/` | CSVs tabulares calculados de `raw/`, prontos para abrir em qualquer ferramenta. |")
    A("| `code.zip` | Snapshot do repositorio no commit acima: firmware de voo, os 8 firmwares de experimento, os 12 solvers e os scripts de analise. |")
    A("| `MANIFEST.md` | SHA-256 de cada arquivo, para conferir integridade. |")
    A("| `PROVENANCE.md` | Por que oito capturas saem carimbadas `-dirty` e por que continuam rastreaveis. |")
    A("")
    A("## As capturas brutas (`raw/`)")
    A("")
    A("Cada uma comeca com um carimbo que diz de qual firmware ela saiu:")
    A("")
    A("```")
    A("STAMP,<commit>,<arvore suja>,<epoch>,<compilado em>,<chip>,<rev>,<nucleos>,<MHz>")
    A("```")
    A("")
    A("| Arquivo | Experimento |")
    A("|---|---|")
    A("| `serial_capture_bateria_v5_6traj.txt` | bateria principal: 12 solvers x 60000 pontos (Tabela 1, **Figuras 1 e 2**) |")
    A("| `s3/serial_capture_bateria_s3.txt` | a mesma bateria no ESP32-S3, que tem FPU (Tabela 2) |")
    A("| `serial_tolerance_sweep_frobenius.txt` | varredura de tolerancia, 1e-2 a 1e-6 (**Figura 3**) |")
    A("| `serial_tol_qr_sweep_A.txt` | tolerancia cruzada com os pesos Q/R (1,4 M chamadas) |")
    A("| `serial_sweep_qr_v4.txt` | mapa grosso de seguranca Q/R, com pico de magnitude interna |")
    A("| `serial_boundary_fine_B.txt` | fronteira fina de quebra do ponto fixo (**Figura 4**) |")
    A("| `serial_repeatability_D.txt` | 20 repeticoes no mesmo dado: jitter de execucao |")
    A("| `serial_gamma_sweep.txt` | escolha do deslocamento gamma do SDA-SS |")
    A("| `serial_norm_benchmark.txt` | custo do teste de convergencia, ciclo a ciclo |")
    A("| `voo/voo_run1..10.txt` | 10 janelas de 360 s do ciclo de voo completo (**Figura 6**) |")
    A("| `serial_flightloop_E.txt` | copia de `voo/voo_run10.txt` sob o nome legado que alguns scripts ainda esperam |")
    A("")
    A("As dez janelas de voo sao do mesmo binario. E' o unico experimento nao")
    A("deterministico do conjunto: a mediana do ciclo repete (4,70 ms nas dez), a cauda")
    A("nao (de 0 a 6 ciclos acima do periodo por janela).")
    A("")
    A("## Os derivados (`derived/`)")
    A("")
    A("| Arquivo | Conteudo |")
    A("|---|---|")
    A("| `battery_s2_runs.csv`, `battery_s3_runs.csv` | uma linha por chamada de solver: tempo, iteracoes, residuo, desfecho |")
    A("| `tolerance_sweep_runs.csv` | idem, com o passo medido na parada e o sinalizador de exatidao em bits |")
    A("| `flight_cycle_runs.csv` | uma linha por janela de voo |")
    A("| `flight_cycle_histogram.csv` | histograma de 50 us das 10 janelas, lado a lado e agregado |")
    A("| `closed_loop_cost.csv` | custo acumulado por solver e trajetoria |")
    A("| `closed_loop_series_T*.csv` | series temporais das 6 trajetorias |")
    A("| `gain_update_schedule.csv` | efeito de recalcular K a cada N ciclos, ou congelar |")
    A("| `operating_point_coverage.csv` | os 60000 pontos com condicionamento e norma de P |")
    A("| `reference_residual.csv` | residuo da referencia float64 (scipy) nos mesmos pontos |")
    A("| `discretisation_fidelity.csv` | fidelidade do modelo discreto por trajetoria |")
    A("| `table2.csv`, `memory_map.csv` | Tabela 2 e metricas de memoria do firmware |")
    A("| `memoria_v8.json` | as mesmas metricas de memoria na forma que a auditoria le |")
    A("")
    A("## Refazer tudo")
    A("")
    A("Sem hardware, a partir so' deste pacote. Descompacte `code.zip` e trabalhe")
    A("dentro do snapshot.")
    A("")
    A("**Passo 1 — as capturas brutas.** De `raw.zip` para `outputs/` do snapshot,")
    A("mantendo os nomes: `raw/*.txt` em `outputs/`, `raw/s3/` em `outputs/s3/` e")
    A("`raw/voo/` em `outputs/voo/`.")
    A("")
    A("**Passo 2 — os derivados.** Os CSVs de `derived/` foram renomeados para ingles")
    A("na publicacao, e os scripts procuram os nomes originais. Renomeie os da tabela")
    A("(o conteudo e' identico, so' o nome muda):")
    A("")
    A("| Em `derived/` | Nome que o codigo espera |")
    A("|---|---|")
    A("| `operating_point_coverage.csv` | `outputs/cobertura_full_v5_6traj.csv` |")
    A("| `closed_loop_cost.csv` | `outputs/malha_fechada_v6_6traj.csv` |")
    A("| `closed_loop_series_T*.csv` | `outputs/malha_fechada_serie_T*.csv` |")
    A("| `gain_update_schedule.csv` | `outputs/ganho_congelado_6traj.csv` |")
    A("| `reference_residual.csv` | `outputs/v8/residuo_referencia.csv` |")
    A("| `discretisation_fidelity.csv` | `outputs/v8/fidelidade_discretizacao.csv` |")
    A("| `table2.csv` | `outputs/v8/tabela2_v8.csv` |")
    A("| `memoria_v8.json` | `outputs/v8/memoria_v8.json` |")
    A("")
    A("**Passo 3 — rodar.**")
    A("")
    A("```bash")
    A("pip install -r requirements.txt")
    A("python python/auditoria.py                  # procedencia, numeros e figuras")
    A("python python/numeros_artigo.py --tabelas   # recalcula as Tabelas 1 e 2")
    A("python python/figuras_artigo_final.py --outdir figuras --flight-dir outputs/voo")
    A("```")
    A("")
    A("O `--outdir` e' obrigatorio: sem ele o script escreve na pasta do artigo dos")
    A("autores. A etapa `figuras` da auditoria compara com o `.tex`, que nao vem no")
    A("pacote, entao ela e' a unica que nao roda de fora; as demais rodam.")
    A("")
    A("Conferido em 2026-09-01: com estes passos as seis figuras do artigo saem")
    A("**identicas** as publicadas, pixel a pixel.")
    A("")
    A("Para refazer as medidas do zero, com as placas: `python python/run_experiments.py --all`")
    A("(~10 h). Detalhes em `experiments/README.md` dentro do snapshot.")
    A("")
    A("## Licenca")
    A("")
    A("| O que | Licenca |")
    A("|---|---|")
    A("| Software (tudo em `code.zip`) | MIT — ver `LICENSE` dentro do snapshot |")
    A("| Dados de medicao (`raw/` e `derived/`) | CC BY 4.0 — ver `LICENSE-DATA` |")
    A("")
    A("Atribuicao sugerida:")
    A("")
    A("> Bassani, G. A. A.; Cardoso, R.; Correa, D. P. F. (2026).")
    A("> *Comparative Analysis of Numerical Methods for Solving the Discrete Algebraic")
    A("> Riccati Equation in Resource-Constrained Embedded Systems: data and code.* Zenodo.")
    A("> https://doi.org/%s" % DOI)
    A("")
    with open(os.path.join(ZENODO, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print("  README.md do deposito escrito")



def escreve_procedencia(commit):
    """PROVENANCE.md solto na raiz + o patch da arvore de trabalho em raw/.

    `verifica_procedencia.py` reprova oito das nove capturas com ARVORE SUJA, e
    isso e' verdade: a campanha rodou de madrugada com as tres alteracoes do
    teste de parada ainda por commitar. O registro de por que o dado continua
    rastreavel vivia so' em outputs/v9/, que e' gitignored -- nao entrava no
    code.zip nem em parte alguma do pacote. Sem ele, quem rodar a auditoria a
    partir do deposito ve' oito reprovacoes sem explicacao.
    """
    origem = os.path.join(OUT, "v9")
    evid = os.path.join(RAW, "provenance")
    os.makedirs(evid, exist_ok=True)
    levados = []
    for nome in ("arvore_de_trabalho_94af585.patch",
                 "arvore_de_trabalho_94af585.status"):
        src = os.path.join(origem, nome)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(evid, nome))
            levados.append(nome)

    L = []
    A = L.append
    A("# Procedencia das capturas / Provenance of the captures")
    A("")
    A("**In English:** the audit script reports eight of the nine captures as")
    A("taken from a modified working tree (`94af585-dirty`). That is accurate.")
    A("All nine came from the *same* build, the exact working-tree diff is")
    A("included in `raw/provenance/`, and after the change was committed one")
    A("experiment was recaptured from a clean tree: seven measured quantities")
    A("came back identical to the hundredth of a microsecond. The defect is in")
    A("the label, not in the data. Details below, in Portuguese.")
    A("")
    A("## O que aconteceu")
    A("")
    A("A campanha rodou das 00:16 as 10:48 de 02/09/2026 com as tres alteracoes")
    A("do teste de parada ainda na arvore de trabalho. Todas as capturas sairam")
    A("carimbadas `94af585-dirty`, e `python/verifica_procedencia.py` as reprova.")
    A("")
    A("## Por que o dado continua rastreavel")
    A("")
    A("**1. As nove vieram do mesmo build.** O proprio `verifica_procedencia`")
    A("confirma que nao ha mistura de binarios: o defeito e' de rotulo.")
    A("")
    A("**2. O diff exato esta' aqui.** `raw/provenance/arvore_de_trabalho_94af585.patch`")
    A("(3 arquivos, 270 linhas) e' o estado da arvore no momento da campanha;")
    A("`.status` registra `HEAD = 94af585`. Aplicado sobre 94af585, reproduz o")
    A("binario que gerou estas capturas.")
    A("")
    A("**3. Foi verificado empiricamente.** Depois do commit `abe5e08`, o")
    A("experimento `norma` foi recapturado com a arvore limpa:")
    A("")
    A("| Grandeza | suja (`94af585-dirty`) | limpa (`abe5e08`) |")
    A("|---|---|---|")
    A("| Norma atual (soft-div) | 215,12 us | 215,12 us |")
    A("| Norma otimizada (soft-mul) | 70,24 us | 70,24 us |")
    A("| Norma inteira (int64) | 5,47 us | 5,47 us |")
    A("| Norma do caminho float | 46,21 us | 46,21 us |")
    A("| SDA-fx iteracao pura | 276,23 us | 276,23 us |")
    A("| ADDA-fx iteracao pura | 421,45 us | 421,45 us |")
    A("| VI-fx iteracao pura | 151,38 us | 151,38 us |")
    A("")
    A("Sete grandezas identicas ate o centesimo de microssegundo: o binario sujo")
    A("era o binario que virou `abe5e08`.")
    A("")
    A("## Situacao de cada captura")
    A("")
    A("| Experimento | Carimbo | Situacao |")
    A("|---|---|---|")
    A("| `norma` | `abe5e08` (limpo) | recapturado apos o commit |")
    A("| `voo` | `94af585-dirty` | **sujo esperado**: o runner liga `DEBUG_MODE` de proposito |")
    A("| `bateria`, `benchmark_s3`, `tolerancia`, `gamma`, `sweep_qr`, `tol_qr`, "
      "`fronteiras`, `repetibilidade` | `94af585-dirty` | sujo, com a equivalencia acima |")
    A("")
    A("## Ao rodar a auditoria a partir deste deposito")
    A("")
    A("`python python/auditoria.py` reprova a etapa `procedencia` por este motivo,")
    A("e apenas por ele. As demais etapas --- 253 checagens numericas, as figuras")
    A("e o ciclo de voo --- passam com zero divergencias.")
    A("")
    caminho = os.path.join(ZENODO, "PROVENANCE.md")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(chr(10).join(L))
    print("  PROVENANCE.md escrito (%d evidencia(s) em raw/provenance/)"
          % len(levados))
    return caminho


def empacota_para_upload():
    """Compacta raw/, derived/ e code/ em tres zips na raiz do deposito.

    O Zenodo aceita ARQUIVOS, nao pastas: arrastar a arvore inteira achata a
    estrutura e o revisor perde a distincao entre medida e derivacao. Tres zips
    preservam a divisao e, mais util, deixam quem so' quer conferir as tabelas
    baixar 40 MB em vez de 193 MB.

    README.md e MANIFEST.md ficam soltos de proposito: aparecem na pagina do
    deposito sem que ninguem precise baixar nada.
    """
    saidas = []
    for pasta in (RAW, DERIVED):
        if not os.path.isdir(pasta):
            continue
        base = os.path.join(ZENODO, os.path.basename(pasta))
        caminho = shutil.make_archive(base, "zip", root_dir=ZENODO,
                                      base_dir=os.path.basename(pasta))
        shutil.rmtree(pasta)
        saidas.append((os.path.basename(caminho), os.path.getsize(caminho)))
    for nome, tam in saidas:
        print("  %-14s %6.1f MB" % (nome, tam / 1e6))
    return saidas


def escreve_manifesto(commit):
    """SHA-256 das origens e de cada arquivo que sobe para o deposito.

    Chamada DEPOIS de empacota_para_upload(): so' assim ela ve os zips finais.
    """
    manifest_path = os.path.join(ZENODO, "MANIFEST.md")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# Manifesto do pacote de dados — DINAME 2027 (v8)\n\n")
        f.write("DOI: %s%s" % (DOI, chr(10) * 2))
        f.write("Gerado por `python/exporta_dados_revisores.py` a partir do commit "
                "`%s` deste repositório.\n\n" % commit)
        f.write("Hash SHA-256 dos arquivos de origem (capturas seriais brutas em `outputs/`, "
                "não movidas nem alteradas por este script):\n\n")
        f.write("| Arquivo de origem | SHA-256 |\n|---|---|\n")
        fontes = dict(SOURCES)
        for p in sorted(glob.glob(VOO_GLOB)):
            fontes["voo_" + os.path.splitext(os.path.basename(p))[0]] = p
        for key, path in fontes.items():
            if os.path.isfile(path):
                f.write("| `%s` | `%s` |\n" % (os.path.relpath(path, REPO), sha256_of(path)))
            else:
                f.write("| `%s` | (nao encontrado nesta maquina) |\n" % os.path.relpath(path, REPO))
        f.write("\nArquivos deste deposito (confira com `sha256sum`):\n\n")
        for raiz, _, arquivos in os.walk(ZENODO):
            for name in sorted(arquivos):
                if name in ("MANIFEST.md", "README.md", "PROVENANCE.md"):
                    continue
                p = os.path.join(raiz, name)
                rel = os.path.relpath(p, ZENODO).replace(os.sep, "/")
                f.write("- `%s` (%d bytes, sha256 `%s`)\n" % (rel, os.path.getsize(p), sha256_of(p)))
    return manifest_path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Monta o pacote de dados do Zenodo.")
    ap.add_argument("--sem-zip", action="store_true",
                    help="deixa raw/, derived/ e code/ como pastas, sem compactar "
                         "(util para inspecionar antes de subir)")
    ap.add_argument("--doi", default=None,
                    help="DOI desta versao (ex.: 10.5281/zenodo.22256810). Cada "
                         "versao nova no Zenodo ganha o seu; reserve-o no rascunho "
                         "(New version -> o DOI ja' aparece) e passe aqui, em vez "
                         "de editar a constante a mao.")
    ap.add_argument("--doi-conceito", default=None,
                    help="DOI conceito, que resolve sempre para a versao mais "
                         "recente. So' muda se o deposito for recriado do zero.")
    args = ap.parse_args()

    global DOI, DOI_CONCEITO
    if args.doi:
        DOI = args.doi.strip().replace("https://doi.org/", "")
    if args.doi_conceito:
        DOI_CONCEITO = args.doi_conceito.strip().replace("https://doi.org/", "")
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", DOI):
        raise SystemExit("DOI com formato inesperado: %r" % DOI)
    print("DOI desta versao: %s  |  conceito: %s" % (DOI, DOI_CONCEITO))

    os.makedirs(ZENODO, exist_ok=True)
    # Nada de uma execucao anterior sobrevive: uma rodada --sem-zip deixa raw/,
    # derived/ e code/ soltos, e a rodada seguinte herdava esse estado. Foi por
    # aí que o MANIFEST da v2 acabou carimbando os zips da geracao anterior.
    for pasta in (RAW, DERIVED, CODE):
        if os.path.isdir(pasta):
            print("  limpando sobra de execucao anterior: %s/"
                  % os.path.basename(pasta))
            shutil.rmtree(pasta)
    os.makedirs(DERIVED, exist_ok=True)
    print("Exportando para %s" % ZENODO)

    export_tolerance_sweep(SOURCES["tolerance_sweep"],
                            os.path.join(DERIVED, "tolerance_sweep_runs.csv"))
    export_battery(SOURCES["battery_s2"], os.path.join(DERIVED, "battery_s2_runs.csv"))
    export_battery(SOURCES["battery_s3"], os.path.join(DERIVED, "battery_s3_runs.csv"))

    table2_dst = os.path.join(DERIVED, "table2.csv")
    if os.path.isfile(SOURCES["table2_csv"]):
        shutil.copyfile(SOURCES["table2_csv"], table2_dst)
        print("  copiado table2.csv")
    else:
        print("  pulado (rodar python/generate_table2.py antes): %s" % SOURCES["table2_csv"])

    export_memory_csv(SOURCES["memory_json"], os.path.join(DERIVED, "memory_map.csv"))
    # o JSON tambem, e nao so' o CSV achatado: a etapa de memoria da auditoria le
    # o JSON, e sem ele quem baixa o pacote nao consegue rodar essa checagem.
    if os.path.isfile(SOURCES["memory_json"]):
        shutil.copyfile(SOURCES["memory_json"],
                        os.path.join(DERIVED, "memoria_v8.json"))
        print("  copiado memoria_v8.json")

    # ciclo de voo: N execucoes do mesmo binario (unico experimento nao deterministico)
    export_flight(sorted(glob.glob(VOO_GLOB),
                         key=lambda p: int(re.search(r"voo_run(\d+)", p).group(1))),
                  os.path.join(DERIVED, "flight_cycle_runs.csv"),
                  os.path.join(DERIVED, "flight_cycle_histogram.csv"))

    # malha fechada e cobertura: ja sao CSV tabulares na origem, copiados como estao
    for key, name in (("closed_loop", "closed_loop_cost.csv"),
                      ("gain_schedule", "gain_update_schedule.csv"),
                      ("coverage", "operating_point_coverage.csv"),
                      ("reference_residual", "reference_residual.csv"),
                      ("discretisation_fidelity", "discretisation_fidelity.csv")):
        src = SOURCES[key]
        if os.path.isfile(src):
            # copyfile e' binario de proposito: a copia em modo texto convertia
            # a quebra de linha no Windows e o SHA-256 do arquivo publicado
            # deixava de bater com o da origem que o proprio MANIFEST lista.
            shutil.copyfile(src, os.path.join(DERIVED, name))
            print("  copiado %s" % name)
        else:
            print("  pulado (nao encontrado): %s" % src)

    # as seis series temporais de malha fechada, descobertas por glob
    for src in sorted(glob.glob(SERIES_GLOB)):
        name = "closed_loop_series_" + os.path.basename(src).split("serie_")[1]
        shutil.copyfile(src, os.path.join(DERIVED, name))
        print("  copiado %s" % name)

    # o que o artigo promete no deposito, e que ate 2026-09-01 nao estava la:
    brutas = copia_capturas_brutas()
    commit = git_commit()
    codigo = arquiva_codigo(commit)
    escreve_readme(commit, codigo)
    escreve_procedencia(commit)

    # A compactacao vem ANTES do manifesto, de proposito. Ate 2026-09-01 era o
    # contrario, e o manifesto carimbava os zips da geracao anterior: o
    # MANIFEST.md publicado na v2 lista SHA-256 que nao correspondem a nenhum
    # dos tres zips publicados (e um tamanho errado para o code.zip). Os zips
    # estavam integros; errado estava o documento que serve para provar isso.
    if not args.sem_zip:
        print(chr(10) + "Compactando para upload no Zenodo:")
        empacota_para_upload()

    manifest_path = escreve_manifesto(commit)
    print("MANIFEST.md escrito em %s" % manifest_path)

    if not args.sem_zip:
        print(chr(10) + "Suba estes itens no deposito %s:" % DOI)
        for n in ("README.md", "PROVENANCE.md", "MANIFEST.md", "raw.zip",
                  "derived.zip", "code.zip"):
            p = os.path.join(ZENODO, n)
            if os.path.isfile(p):
                print("   %-14s %6.1f MB" % (n, os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
