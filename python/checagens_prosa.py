# -*- coding: utf-8 -*-
"""Checagens dos numeros que vivem no CORPO do artigo, nao nas tabelas.

Por que este arquivo existe: ate a auditoria de 2026-09-01, `verifica_numeros_artigo.py`
cobria as Tabelas 1 e 2, as razoes de speedup, o piso de quantizacao e o ciclo de
voo -- 77 checagens. O corpo do artigo tem ~245 numeros distintos, e os que ficavam
de fora eram conferidos a mao, quando eram. Foi assim que sobreviveram ao v7 um
jitter de "0,019 %" que o dado nunca mostrou (o minimo real e' 0,023 %) e um
residuo de referencia de "5,7e-15" contra 4,8e-15 medido. Numero sem teste
envelhece em silencio; e' esse o furo que este modulo fecha.

Cada entrada abaixo e' uma afirmacao do texto com a fonte bruta ao lado. Ao
recapturar a campanha, rode `python python/verifica_numeros_artigo.py --v8` e o
que tiver saido do lugar aparece como XX.

Importado por verifica_numeros_artigo.py; nao roda sozinho.
"""
import csv
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")


def _linhas(path, prefixo):
    """Linhas que contem `prefixo`, tolerando lixo de sincronizacao serial colado
    no inicio da linha (a captura comeca com a placa ja transmitindo)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            i = ln.find(prefixo)
            if i >= 0:
                yield ln[i:].rstrip("\n").split(",")


# ---------------------------------------------------------------------------
def balanco_das_janelas(check):
    """Sec. The complete control cycle: as dez janelas de 360 s.

    O v8 dizia "360 s each, 47475 to 47646 cycles per window at a mean period of
    6.002 ms" -- e 47475 x 6,002 ms = 285 s, nao 360. A diferenca sao os blocos
    periodicos de status que a captura imprime, e que a estatistica de ciclo
    exclui. O texto passou a dizer isso; aqui esta a checagem que faltava, para
    o numero novo nao envelhecer em silencio como os outros.
    """
    import glob
    try:
        import analisa_voo
    except ImportError:
        return
    caminhos = sorted(glob.glob(os.path.join(OUT, "voo", "voo_run*.txt")))
    janelas = [j for j in (analisa_voo.analisa(c) for c in caminhos) if j]
    if len(janelas) < 2:
        return
    ciclos = [j["ciclos"] for j in janelas]
    segundos = [j["ciclos"] * j["periodo"] / 1000.0 for j in janelas]
    check("prosa: ciclos por janela, minimo (47475)", 47475, min(ciclos), tol=0.001)
    check("prosa: ciclos por janela, maximo (47646)", 47646, max(ciclos), tol=0.001)
    check("prosa: periodo medio das janelas (6.002 ms)", 6.002,
          st.mean(j["periodo"] for j in janelas), tol=0.001)
    check("prosa: segundos de ciclo por janela (285 s)", 285.0,
          st.mean(segundos), tol=0.005)
    check("prosa: ciclos somados nas dez janelas (475120)", 475120, sum(ciclos), tol=0.001)
    check("prosa: estouros do periodo, total (21)", 21, sum(j["estouros"] for j in janelas))
    check("prosa: estouros do periodo (0.004%)", 0.004,
          100.0 * sum(j["estouros"] for j in janelas) / sum(ciclos), tol=0.13)


# ---------------------------------------------------------------------------
def jitter(check):
    """Sec. Cost predictability: 'mean coefficient of variation between 0.023%
    (SDA-SS) and 0.078% (SDA-Scaled-fx) per doubling solver, with no single point
    exceeding 0.14%'. Fonte: serial_repeatability_D.txt, coluna cv_pct."""
    cv = defaultdict(list)
    for p in _linhas(os.path.join(OUT, "serial_repeatability_D.txt"), "SUMMARY,"):
        if len(p) < 11 or p[1] == "traj":
            continue
        try:
            cv[p[3]].append(float(p[8]))
        except ValueError:
            pass
    dbl = {m: v for m, v in cv.items() if "ITERATIVE" not in m}
    if not dbl:
        return
    medios = {m: st.mean(v) for m, v in dbl.items()}
    check("prosa: CV medio minimo (0.023%, SDA-SS)", 0.023, min(medios.values()))
    check("prosa: CV medio maximo (0.078%, SDA-Scaled-fx)", 0.078, max(medios.values()))
    check("prosa: nenhum ponto acima de 0.14%", 0.14, max(max(v) for v in dbl.values()))
    check("prosa: pontos por solver na repetibilidade (2004)", 2004,
          st.median([len(v) for v in dbl.values()]))


def referencia_float64(check):
    """Sec. Reference solution: residuo DARE da referencia entre 4.8e-15 e 3.0e-13,
    mediana 6.3e-14. Fonte: outputs/v8/residuo_referencia.csv
    (gerar com python/derivados_artigo.py referencia)."""
    path = os.path.join(OUT, "v8", "residuo_referencia.csv")
    if not os.path.isfile(path):
        print("[INFO] %s ausente (rode python python/derivados_artigo.py referencia)."
              % os.path.relpath(path, REPO))
        return
    r = [float(x["dare_residual_rel"]) for x in csv.DictReader(open(path, encoding="utf-8"))]
    check("prosa: residuo da referencia, minimo (4.8e-15)", 4.8e-15, min(r), tol=0.06)
    check("prosa: residuo da referencia, mediana (6.3e-14)", 6.3e-14, float(np.median(r)), tol=0.06)
    check("prosa: residuo da referencia, maximo (3.0e-13)", 3.0e-13, max(r), tol=0.06)
    check("prosa: pontos da referencia (60000)", 60000, len(r))


def fidelidade_discretizacao(check):
    """Sec. Operating points: ||Ac*Ts||_2 abaixo de 0.05 em T1/T2/T4/T5, 0.114 em
    T3 e 0.279 em T6; termo desprezado <2e-5, 2.5e-4 e 3.6e-3.
    Fonte: outputs/v8/fidelidade_discretizacao.csv."""
    path = os.path.join(OUT, "v8", "fidelidade_discretizacao.csv")
    if not os.path.isfile(path):
        print("[INFO] %s ausente (rode python python/derivados_artigo.py fidelidade)."
              % os.path.relpath(path, REPO))
        return
    d = {x["traj"]: (float(x["norm_Ac_Ts_2"]), float(x["termo_desprezado_rel"]))
         for x in csv.DictReader(open(path, encoding="utf-8"))}
    brandas = [t for t in d if t.split("_")[0] in ("T1", "T2", "T4", "T5")]
    check("prosa: ||AcTs|| maximo em T1/T2/T4/T5 (<0.05)", 0.05,
          max(d[t][0] for t in brandas), tol=0.25)
    check("prosa: ||AcTs|| em T3 (0.114)", 0.114, d["T3_chirp"][0])
    check("prosa: ||AcTs|| em T6 (0.279)", 0.279, d["T6_taxa_alta"][0])
    check("prosa: termo desprezado em T1/T2/T4/T5 (<2e-5)", 2e-5,
          max(d[t][1] for t in brandas), tol=0.10)
    check("prosa: termo desprezado em T3 (2.5e-4)", 2.5e-4, d["T3_chirp"][1], tol=0.05)
    check("prosa: termo desprezado em T6 (3.6e-3)", 3.6e-3, d["T6_taxa_alta"][1], tol=0.05)


def predictability(check, bat_path):
    """Sec. Cost predictability: SDA-fx entre 3.67 e 3.69 ms e nunca 62% do periodo;
    VI-fx de 0.94 ms (T2) a 12.79 ms (T6), chamadas ate 41.8 ms (60.8 ms em float),
    estourando o periodo em 43.9% do chirp e 100% de T6; medianas de iteracao da VI
    1, 1, 13, 32. Fonte: a bateria principal."""
    tempo = defaultdict(lambda: defaultdict(list))
    iters = defaultdict(lambda: defaultdict(list))
    for p in _linhas(bat_path, "RUN,"):
        if len(p) < 8 or p[1] == "traj":
            continue
        tempo[p[3]][p[1]].append(int(p[4]) / 1000.0)
        iters[p[3]][p[1]].append(int(p[5]))
    if not tempo:
        return
    sda = tempo["SDA_FIXED"]
    med = [st.median(v) for v in sda.values()]
    todos = [x for v in sda.values() for x in v]
    check("prosa: SDA-fx mediana minima por traj (3.67 ms)", 3.67, min(med))
    check("prosa: SDA-fx mediana maxima por traj (3.69 ms)", 3.69, max(med))
    check("prosa: SDA-fx nunca atinge 62% do periodo", 62.0, 100.0 * max(todos) / 6.0, tol=0.01)

    vi, vif = tempo["ITERATIVE_FIXED"], tempo["ITERATIVE"]
    check("prosa: VI-fx mediana em T2 (0.94 ms)", 0.94, st.median(vi["T2_figura8"]))
    check("prosa: VI-fx mediana em T6 (12.79 ms)", 12.79, st.median(vi["T6_taxa_alta"]))
    check("prosa: VI-fx chamada mais longa (41.8 ms)", 41.8, max(x for v in vi.values() for x in v))
    check("prosa: VI float chamada mais longa (60.8 ms)", 60.8, max(x for v in vif.values() for x in v))
    for traj, alvo in (("T3_chirp", 43.9), ("T6_taxa_alta", 100.0)):
        v = vi[traj]
        check("prosa: VI-fx acima do periodo em %s (%.1f%%)" % (traj.split("_")[0], alvo),
              alvo, 100.0 * sum(1 for x in v if x > 6.0) / len(v))
    for traj, alvo in (("T5_tilt_alto", 1), ("T4_degrau_yaw", 1), ("T3_chirp", 13), ("T6_taxa_alta", 32)):
        check("prosa: iteracoes medianas da VI em %s (%d)" % (traj.split("_")[0], alvo),
              alvo, st.median(iters["ITERATIVE_FIXED"][traj]), tol=0.01)

    # 'ASDA-fx reaches 10 iterations somewhere in the battery where its
    # single-precision counterpart never exceeds 9' -- vem das linhas SUMMARY,
    # que cobrem os 60000 pontos, nao a subamostra de 12000 das linhas RUN.
    max_it = {}
    for p in _linhas(bat_path, "SUMMARY,"):
        if len(p) < 8 or p[1] == "traj_ou_ALL":
            continue
        if p[1] == "ALL":
            try:
                max_it[p[2]] = int(p[7])
            except ValueError:
                pass
    if max_it:
        check("prosa: ASDA-fx atinge 10 iteracoes", 10, max_it.get("ASDA_FIXED"), tol=0.01)
        check("prosa: ASDA float nunca passa de 9", 9, max_it.get("ASDA"), tol=0.01)


def teste_de_convergencia(check):
    """Sec. Stopping test: 214.8 us / 51551 ciclos, 129.8 us nas 72 conversoes a
    433 ciclos cada, 2.4 us na raiz; 276.2 / 421.4 / 151.4 us de aritmetica pura,
    logo 43.7% / 33.8% / 58.7% da iteracao; alternativas a 69.9 us (3.1x) e
    5.5 us (39x). Fonte: serial_norm_benchmark.txt."""
    path = os.path.join(OUT, "serial_norm_benchmark.txt")
    if not os.path.isfile(path):
        return
    txt = open(path, encoding="utf-8", errors="replace").read()

    def num(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None

    atual = num(r"Norma atual .*?:\s*([\d.]+) us")
    ciclos = num(r"Norma atual .*?:\s*[\d.]+ us \(\s*(\d+) ciclos")
    div = num(r"72 chamadas q2f \(div\):\s*([\d.]+) us")
    cyc_div = num(r"([\d.]+) cyc/div")
    raiz = num(r"sqrtf isolado \(soft\):\s*([\d.]+) us")
    otim = num(r"Norma otimizada \(soft-mul\):\s*([\d.]+) us")
    inteira = num(r"Norma inteira \(int64_t\):\s*([\d.]+) us")
    sda = num(r"SDA-fx iter pura:\s*([\d.]+) us")
    adda = num(r"ADDA-fx iter pura:\s*([\d.]+) us")
    vi = num(r"VI-fx iter pura:\s*([\d.]+) us")

    check("prosa: custo do teste (214.8 us)", 214.8, atual)
    check("prosa: custo do teste em ciclos (51551)", 51551, ciclos)
    check("prosa: 72 conversoes int->float (129.8 us)", 129.8, div)
    check("prosa: ciclos por divisao em software (433)", 433, cyc_div)
    check("prosa: raiz quadrada (2.4 us)", 2.4, raiz)
    check("prosa: aritmetica pura SDA-fx (276.2 us)", 276.2, sda)
    check("prosa: aritmetica pura ADDA-fx (421.4 us)", 421.4, adda)
    check("prosa: aritmetica pura VI-fx (151.4 us)", 151.4, vi)
    if atual and sda:
        check("prosa: teste = 43.7% da iteracao do SDA-fx", 43.7, 100 * atual / (atual + sda))
        check("prosa: teste = 33.8% da iteracao do ADDA-fx", 33.8, 100 * atual / (atual + adda))
        check("prosa: teste = 58.7% da iteracao da VI-fx", 58.7, 100 * atual / (atual + vi))
        check("prosa: reciproco pre-computado (69.9 us, 3.1x)", 3.1, atual / otim)
        check("prosa: teste inteiro (5.5 us, 39x)", 39.0, atual / inteira)


def mapa_de_seguranca(check):
    """Sec. Safe region: dentro de R_scale em [0.1, 10], ASDA-fx tem zero breakdowns
    em 19500 execucoes contra 7.8-13.6% das outras quatro; a fronteira superior da
    1.9% em R_scale=116.2 e 100% em 137.0. Fonte: serial_boundary_fine_B.txt."""
    path = os.path.join(OUT, "serial_boundary_fine_B.txt")
    if not os.path.isfile(path):
        return
    faixa = defaultdict(lambda: [0, 0])
    fronteira = defaultdict(lambda: [0, 0])
    for p in _linhas(path, "SUMMARY"):
        if len(p) != 8:
            continue
        try:
            rs, m, nbk, cnt = float(p[1]), p[3], int(p[6]), int(p[7])
        except ValueError:
            continue
        if 0.1 <= rs <= 10.0:
            a = faixa[m]
            a[0] += nbk
            a[1] += cnt
        # a fronteira do artigo agrega as CINCO variantes de ponto fixo (e' um
        # limite do formato Q13.18, nao de um algoritmo), 5 x 5 q x 300 pontos
        if m.endswith("_FIXED"):
            a = fronteira[round(rs, 1)]
            a[0] += nbk
            a[1] += cnt
    if not faixa:
        return
    fx = {m: 100.0 * v[0] / v[1] for m, v in faixa.items() if m.endswith("_FIXED")}
    outros = {m: v for m, v in fx.items() if m != "ASDA_FIXED"}
    check("prosa: execucoes na faixa usavel (19500)", 19500, faixa["ASDA_FIXED"][1])
    check("prosa: ASDA-fx sem breakdown na faixa usavel (0)", 0.0, fx.get("ASDA_FIXED"), tol=0.001)
    check("prosa: menor taxa de breakdown das outras (7.8%)", 7.8, min(outros.values()))
    check("prosa: maior taxa de breakdown das outras (13.6%)", 13.6, max(outros.values()))
    for rs, alvo in ((116.2, 1.9), (137.0, 100.0)):
        v = fronteira.get(round(rs, 1))
        if v:
            check("prosa: breakdown em R_scale=%.1f (%.1f%%)" % (rs, alvo),
                  alvo, 100.0 * v[0] / v[1], tol=0.05)


def gamma(check):
    """Sec. Solvers: gamma=0.7 corta de 9 iteracoes (gamma=0.1) para 7 com o menor
    residuo em ponto fixo (9.14e-3 contra 1.34e-2); gamma=0.9 converge em 5 mas
    quebra em 194 de 1824 pontos. Fonte: serial_gamma_sweep.txt."""
    path = os.path.join(OUT, "serial_gamma_sweep.txt")
    if not os.path.isfile(path):
        return
    g = {}
    for p in _linhas(path, "SUMMARY,"):
        if len(p) < 12 or p[3] != "SDA_SS_FIXED":
            continue
        try:
            g[float(p[2])] = (float(p[6]), float(p[7]), int(p[10]))  # iters, residuo, breakdown
        except ValueError:
            pass
    if not g:
        return
    check("prosa: iteracoes de SDA-SS-fx em gamma=0.1 (9)", 9, g[0.1][0], tol=0.01)
    check("prosa: iteracoes de SDA-SS-fx em gamma=0.7 (7)", 7, g[0.7][0], tol=0.01)
    check("prosa: iteracoes de SDA-SS-fx em gamma=0.9 (5)", 5, g[0.9][0], tol=0.01)
    check("prosa: residuo em gamma=0.7 (9.14e-3)", 9.14e-3, g[0.7][1])
    check("prosa: residuo em gamma=0.1 (1.34e-2)", 1.34e-2, g[0.1][1])
    check("prosa: breakdown em gamma=0.9 (194 de 1824)", 194, g[0.9][2], tol=0.01)


def malha_fechada(check):
    """Sec. Closed-loop: |dJ/J| <= 0.26% em T1-T5 para a familia doubling e 0.44%
    com a VI-fx; rho maximo 0.9868; recomputar a cada 20 ciclos custa no maximo
    +0.05% e reduz 0.75% (T1) e 2.05% (T3); congelar K custa 22.94% (T1), 3.65%
    (T2), 1.11% (T4) e reduz 1.19% em T3."""
    mf = os.path.join(OUT, "malha_fechada_v6_6traj.csv")
    if os.path.isfile(mf):
        rows = list(csv.DictReader(open(mf, encoding="utf-8")))
        ref = {r["traj"]: float(r["J_total"]) for r in rows if r["controller"] == "SDA_float64"}
        dbl, vi = [], []
        for r in rows:
            c, tj = r["controller"], r["traj"]
            if c == "SDA_float64" or tj.startswith("T6"):
                continue
            d = abs(100.0 * (float(r["J_total"]) - ref[tj]) / ref[tj])
            (vi if c == "ITERATIVE_FIXED" else dbl).append(d)
        if dbl:
            check("prosa: |dJ/J| maximo do doubling em T1-T5 (0.26%)", 0.26, max(dbl), tol=0.05)
            check("prosa: |dJ/J| maximo com VI-fx em T1-T5 (0.44%)", 0.44, max(vi), tol=0.05)

    cob = os.path.join(OUT, "cobertura_full_v5_6traj.csv")
    if os.path.isfile(cob):
        rows = list(csv.DictReader(open(cob, encoding="utf-8")))
        check("prosa: raio espectral maximo (0.9868)", 0.9868,
              max(float(r["rho"]) for r in rows), tol=0.001)
        tilt = max(math.degrees(math.hypot(float(r["phi"]), float(r["theta"]))) for r in rows)
        taxa = max(math.degrees(max(abs(float(r["p"])), abs(float(r["q"])), abs(float(r["r"]))))
                   for r in rows)
        check("prosa: inclinacao maxima do envelope (80 graus)", 80.0, tilt, tol=0.01)
        check("prosa: taxa maxima do envelope (3000 graus/s)", 3000.0, taxa, tol=0.02)

    gc = os.path.join(OUT, "ganho_congelado_6traj.csv")
    if os.path.isfile(gc):
        d = defaultdict(dict)
        for r in csv.DictReader(open(gc, encoding="utf-8")):
            d[r["schedule"]][r["traj"]] = float(r["cost_increase_pct"])
        t15 = [t for t in d["frozen_start"] if not t.startswith("T6")]
        e20 = d["every_20_cycles"]
        check("prosa: a cada 20 ciclos, pior acrescimo em T1-T5 (0.05%)", 0.05,
              max(e20[t] for t in t15), tol=0.05)
        check("prosa: a cada 20 ciclos, reducao em T1 (0.75%)", -0.75, e20["T1_espiral"], tol=0.02)
        check("prosa: a cada 20 ciclos, reducao em T3 (2.05%)", -2.05, e20["T3_chirp"], tol=0.02)
        fz = d["frozen_start"]
        check("prosa: congelar K custa em T1 (22.94%)", 22.94, fz["T1_espiral"], tol=0.01)
        check("prosa: congelar K custa em T2 (3.65%)", 3.65, fz["T2_figura8"], tol=0.01)
        check("prosa: congelar K custa em T4 (1.11%)", 1.11, fz["T4_degrau_yaw"], tol=0.02)
        check("prosa: congelar K REDUZ o custo em T3 (-1.19%)", -1.19, fz["T3_chirp"], tol=0.02)


def memoria(check):
    """Sec. Implementation: 2390 bytes (2.33 KB) de IRAM no kernel Q13.18;
    101.8 KB de 320 KB de RAM (31.8%). Fonte: outputs/v8/memoria_v8.json
    (gerar com python/parse_memory_map.py apos compilar o firmware de voo)."""
    path = os.path.join(OUT, "v8", "memoria_v8.json")
    if not os.path.isfile(path):
        print("[INFO] %s ausente (rode python python/parse_memory_map.py)."
              % os.path.relpath(path, REPO))
        return
    j = json.load(open(path, encoding="utf-8"))
    check("prosa: IRAM do kernel Q13.18 (2390 bytes)", 2390, j["iram_q13_18"]["code_bytes"], tol=0.01)
    check("prosa: IRAM do kernel Q13.18 (2.33 KB)", 2.33, j["iram_q13_18"]["code_kb"], tol=0.01)
    check("prosa: RAM do firmware de voo (101.8 KB)", 101.8, j["ram"]["used_kb"], tol=0.01)
    check("prosa: RAM do firmware de voo (31.8%)", 31.8, j["ram"]["pct"], tol=0.01)


def tolerancia(check, ts_loader):
    """Sec. Tolerance: apertar tau de 1e-2 a 1e-6 muda o residuo em +6.6% (SDA-fx),
    +5.2% (SDA-Scaled-fx) e -0.2% (ASDA-fx), no maximo 6.8%; custa +19.3% (SDA-fx),
    entre +9.9% e +29.0% na familia. Passo medido: 5.8e-4 em tau=1e-3, 3.0e-5 em
    1e-4, zero em 1824/1824 a partir de 1e-5; ADDA-fx 1823/1824; ASDA-fx 63/1824
    em 1e-6 com passo 5.4e-7. Residuo e' MEDIANA (painel a), tempo e' MEDIA
    (painel b) -- as duas estatisticas que a figura mostra."""
    ts = ts_loader()
    if not ts:
        return
    FX = ["SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"]
    dres, dt = {}, {}
    for m in FX:
        d = ts.get(("0a", m))
        if not d or 1e-2 not in d or 1e-6 not in d:
            return
        r0 = st.median([x["resid"] for x in d[1e-2]])
        r1 = st.median([x["resid"] for x in d[1e-6]])
        t0 = st.mean([x["time_us"] for x in d[1e-2]])
        t1 = st.mean([x["time_us"] for x in d[1e-6]])
        dres[m] = 100.0 * (r1 - r0) / r0
        dt[m] = 100.0 * (t1 - t0) / t0
    check("prosa: residuo de SDA-fx ao apertar tau (+6.6%)", 6.6, dres["SDA_FIXED"], tol=0.03)
    check("prosa: residuo de SDA-Scaled-fx (+5.2%)", 5.2, dres["SDA_SCALED_FIXED"], tol=0.03)
    check("prosa: residuo de ASDA-fx (-0.2%)", -0.2, dres["ASDA_FIXED"], tol=0.25)
    check("prosa: maior mudanca de residuo na familia (6.8%)", 6.8, max(dres.values()), tol=0.03)
    check("prosa: tempo de SDA-fx ao apertar tau (+19.3%)", 19.3, dt["SDA_FIXED"], tol=0.02)
    check("prosa: menor acrescimo de tempo (+9.9%)", 9.9, min(dt.values()), tol=0.02)
    check("prosa: maior acrescimo de tempo (+29.0%)", 29.0, max(dt.values()), tol=0.02)

    sda = ts[("0a", "SDA_FIXED")]
    check("prosa: passo de SDA-fx em tau=1e-3 (5.8e-4)", 5.8e-4,
          st.median([x["step"] for x in sda[1e-3]]), tol=0.02)
    check("prosa: passo de SDA-fx em tau=1e-4 (3.0e-5)", 3.0e-5,
          st.median([x["step"] for x in sda[1e-4]]), tol=0.02)

    def frac(m, tol_val, sub="0a"):
        d = ts.get((sub, m), {}).get(tol_val)
        return (sum(1 for x in d if x["bit_exact"]), len(d)) if d else (None, None)

    for m, tol_val, alvo, rot in (("SDA_FIXED", 1e-5, 1824, "SDA-fx @1e-5"),
                                  ("SDA_SCALED_FIXED", 1e-5, 1824, "SDA-Scaled-fx @1e-5"),
                                  ("ADDA_FIXED", 1e-5, 1823, "ADDA-fx @1e-5"),
                                  ("ASDA_FIXED", 1e-6, 63, "ASDA-fx @1e-6")):
        n_be, n = frac(m, tol_val)
        if n:
            check("prosa: passos bit-exatos de %s (%d de %d)" % (rot, alvo, n),
                  alvo, n_be, tol=0.01)
    check("prosa: passo de ASDA-fx em tau=1e-6 (5.4e-7)", 5.4e-7,
          st.median([x["step"] for x in ts[("0a", "ASDA_FIXED")][1e-6]]), tol=0.03)

    for tol_val, alvo in ((1e-4, 600), (1e-5, 287), (1e-6, 132)):
        d = ts.get(("0b", "ITERATIVE_FIXED"), {}).get(tol_val)
        if d:
            check("prosa: VI-fx converge em tau=%.0e (%d de 600)" % (tol_val, alvo),
                  alvo, sum(1 for x in d if x["outcome"] == 0), tol=0.01)


def tabela1_exatidao(check, bat_path):
    """Tab. 1, colunas de residuo e de erro de ganho -- so' as colunas de tempo e
    iteracoes eram conferidas. O residuo sai da propria captura; o e_K sai de
    numeros_artigo.gain_errors(), que resolve a referencia NO PONTO QUE O
    DISPOSITIVO REPORTOU (linhas PT), nao num ponto regenerado em Python."""
    RES = {  # metodo: residuo DARE mediano publicado
        "SDA": 2.2e-6, "SDA_SS": 8.8e-7, "ADDA": 2.2e-6, "SDA_SCALED": 2.2e-6,
        "ASDA": 2.2e-6, "ITERATIVE": 2.9e-2,
        "SDA_FIXED": 1.0e-2, "SDA_SCALED_FIXED": 1.3e-2, "SDA_SS_FIXED": 8.4e-3,
        "ASDA_FIXED": 5.9e-3, "ADDA_FIXED": 1.2e-2, "ITERATIVE_FIXED": 2.8e-2,
    }
    EK = {  # metodo: erro relativo de ganho publicado
        "SDA": 1.9e-6, "SDA_SS": 1.9e-6, "ADDA": 1.9e-6, "SDA_SCALED": 1.8e-6,
        "ASDA": 1.8e-6, "ITERATIVE": 1.8e-2,
        "SDA_FIXED": 8.9e-3, "SDA_SCALED_FIXED": 1.0e-2, "SDA_SS_FIXED": 8.9e-3,
        "ASDA_FIXED": 3.2e-3, "ADDA_FIXED": 9.9e-3, "ITERATIVE_FIXED": 1.9e-2,
    }
    res = defaultdict(list)
    for p in _linhas(bat_path, "RUN,"):
        if len(p) < 8 or p[1] == "traj":
            continue
        try:
            res[p[3]].append(float(p[6]))
        except ValueError:
            pass
    for m, alvo in RES.items():
        if res.get(m):
            check("Tab1 %s residuo" % m, alvo, st.median(res[m]), tol=0.06)

    try:
        import numeros_artigo

        eK = numeros_artigo.gain_errors()
    except Exception as e:
        print("[INFO] erro de ganho indisponivel (%r); Tab. 1 conferida so' no residuo." % (e,))
        return
    for m, alvo in EK.items():
        if eK.get(m):
            check("Tab1 %s e_K" % m, alvo, st.median(eK[m]), tol=0.06)


def tabela2_completa(check, t2, t3):
    """Tab. 2: as colunas S2 e as duas razoes. As colunas S3 ja eram conferidas
    em verifica_numeros_artigo; aqui fecham-se as quatro restantes."""
    LINHAS = {  # metodo: (S2-float, S2-fx, S2fx/S3fx, S3fx/S3float)
        "SDA": (8.92, 3.68, 1.33, 2.61),
        "SDA_SS": (9.40, 3.92, 1.27, 2.60),
        "ADDA": (9.61, 5.00, 1.27, 3.81),
        "SDA_SCALED": (9.76, 3.81, 1.36, 2.55),
        "ASDA": (10.21, 4.21, 1.35, 2.81),
        "ITERATIVE": (1.03, 0.96, 1.65, 1.92),
    }
    # a linha da value iteration nao entra no TAB2 de verifica_numeros_artigo
    # (que so' cobre a familia doubling); suas duas colunas S3 sao conferidas aqui
    if t3.get("ITERATIVE") and t3.get("ITERATIVE_FIXED"):
        check("Tab2 Value iteration S3-float", 0.30, st.median(t3["ITERATIVE"]) / 1e3, tol=0.02)
        check("Tab2 Value iteration S3-fx", 0.58, st.median(t3["ITERATIVE_FIXED"]) / 1e3, tol=0.02)

    for m, (s2f, s2x, rp, ra) in LINHAS.items():
        fx = m + "_FIXED"
        if not (t2.get(m) and t2.get(fx) and t3.get(m) and t3.get(fx)):
            continue
        check("Tab2 %s S2-float" % m, s2f, st.median(t2[m]) / 1e3)
        check("Tab2 %s S2-fx" % m, s2x, st.median(t2[fx]) / 1e3)
        check("Tab2 %s S2-fx/S3-fx" % m, rp, st.median(t2[fx]) / st.median(t3[fx]))
        check("Tab2 %s S3-fx/S3-float" % m, ra, st.median(t3[fx]) / st.median(t3[m]))


def deslocamento_e_rastreamento(check):
    """Sec. Cost predictability: as iteracoes medianas da VI crescem com a
    distancia media entre estados consecutivos -- 2.6e-3 (T5), 0.21 (T4),
    2.26 (T3), 18.7 (T6). E Sec. Closed-loop: em T3, perto de 1.9 Hz, o
    realizado atrasa 5.2 graus sobre 27.7 de amplitude, e as duas aritmeticas
    diferem 0.12 grau."""
    cob = os.path.join(OUT, "cobertura_full_v5_6traj.csv")
    if os.path.isfile(cob):
        por_traj = defaultdict(list)
        for r in csv.DictReader(open(cob, encoding="utf-8")):
            por_traj[r["traj"]].append(
                [float(r[c]) for c in ("phi", "theta", "p", "q", "r")])
        ALVO = {"T5_tilt_alto": 2.6e-3, "T4_degrau_yaw": 0.21,
                "T3_chirp": 2.26, "T6_taxa_alta": 18.7}
        for traj, alvo in ALVO.items():
            v = por_traj.get(traj)
            if not v or len(v) < 2:
                continue
            d = [math.dist(v[i], v[i - 1]) for i in range(1, len(v))]
            check("prosa: ||dx|| medio em %s (%g)" % (traj.split("_")[0], alvo),
                  alvo, st.mean(d), tol=0.06)

    serie = os.path.join(OUT, "malha_fechada_serie_T3_chirp.csv")
    if not os.path.isfile(serie):
        return
    rows = list(csv.DictReader(open(serie, encoding="utf-8")))
    if not rows or "t" not in rows[0]:
        print("[INFO] serie de T3 sem coluna de tempo; rastreamento nao conferido.")
        return
    # Mesma janela da Fig. 5(a) (fig5_closed_loop, window=(12.0, 14.5)): perto de
    # 1.9 Hz no chirp, que vai de 0.2 a 8 Hz em 60 s.
    jan = [r for r in rows if 12.0 <= float(r["t"]) <= 14.5]
    if not jan:
        return
    ref = [float(r["phi_ref_deg"]) for r in jan]
    f64 = [float(r["phi_SDA_float64_deg"]) for r in jan]
    fx = [float(r["phi_SDA_FIXED_deg"]) for r in jan]

    # 27.7 e' a amplitude do REALIZADO (o comando tem 25.0; o laco sobrepassa).
    check("prosa: amplitude do realizado em T3 (27.7 graus)", 27.7,
          (max(f64) - min(f64)) / 2.0, tol=0.02)
    # 5.2 e 0.12 sao erros MEDIOS na janela, nao maximos.
    atraso = st.mean([abs(a - b) for a, b in zip(ref, f64)])
    dif = st.mean([abs(a - b) for a, b in zip(f64, fx)])
    check("prosa: atraso medio do realizado em T3 (5.2 graus)", 5.2, atraso, tol=0.03)
    check("prosa: diferenca media entre aritmeticas em T3 (0.12 grau)", 0.12, dif, tol=0.05)
    check("prosa: essa diferenca e' 2% do erro de rastreamento", 2.0,
          100.0 * dif / atraso, tol=0.25)


def magnitude_interna(check):
    """Sec. Implementation: 'at nominal weights the largest intermediate magnitude
    reached by any fixed-point variant over the full battery is 1051'. Unica
    captura que instrumenta isso: serial_sweep_qr_v4.txt, coluna max_abs_seen_pico,
    filtrada em r_scale=1 e q_rate_scale=1."""
    path = os.path.join(OUT, "serial_sweep_qr_v4.txt")
    if not os.path.isfile(path):
        return
    picos = []
    for p in _linhas(path, "SUMMARY,"):
        if len(p) < 9 or p[1] == "r_scale":
            continue
        try:
            r, q = float(p[1]), float(p[2])
        except ValueError:
            continue
        if abs(r - 1.0) < 1e-9 and abs(q - 1.0) < 1e-9 and p[3].endswith("_FIXED"):
            try:
                picos.append(float(p[8]))
            except ValueError:
                pass
    if picos:
        check("prosa: maior magnitude interna em pesos nominais (1051)", 1051.0,
              max(picos), tol=0.01)
        check("prosa: teto do formato Q13.18 (8192)", 8192.0, 2.0 ** 13, tol=0.001)
        check("prosa: margem ate o teto (fator 8)", 8.0, 8192.0 / max(picos), tol=0.05)


def alocacoes_de_heap(check):
    """Sec. Platform: 'the single-precision solver performs 18 heap allocations
    per call, at a measured 7.2 us per allocate/free pair (~130 us per solve)'.

    As 18 chamadas a `new` estao em AutoLQR::computeGainMatrixSDA() (13 no setup,
    1 em B_Rinv, 4 no calculo final do ganho) -- contadas aqui do proprio fonte,
    para que uma alteracao no solver derrube a checagem. O custo do par vem do
    item 7 de serial_norm_benchmark.txt."""
    fonte = os.path.join(REPO, "lib", "AUTOLQR", "AutoLQR.cpp")
    if os.path.isfile(fonte):
        txt = open(fonte, encoding="utf-8", errors="replace").read()
        i = txt.find("bool AutoLQR::computeGainMatrixSDA()")
        j = txt.find(chr(10) + "bool AutoLQR::", i + 1)
        corpo = txt[i:j if j > 0 else len(txt)]
        check("prosa: alocacoes de heap no SDA float (18)", 18,
              corpo.count("new "), tol=0.01)

    bench = os.path.join(OUT, "serial_norm_benchmark.txt")
    if os.path.isfile(bench):
        m = re.search(r"new float\[72\] \+ delete\[\]:\s*([\d.]+) us",
                      open(bench, encoding="utf-8", errors="replace").read())
        if m:
            par = float(m.group(1))
            check("prosa: custo de um par new/delete (7.2 us)", 7.2, par, tol=0.02)
            check("prosa: custo das alocacoes por solve (130 us)", 130.0, 18 * par, tol=0.02)


def derivados_da_tabela1(check, t2, r2):
    """Sec. Reference/Results: ASDA-fx e' 2.75x mais exato que SDA-fx e custa
    0.53 ms a mais; tau=1e-3 fica 16x acima do pior piso; ||P||_F fica perto
    de 0.43."""
    if t2.get("SDA_FIXED") and t2.get("ASDA_FIXED"):
        check("prosa: custo extra do ASDA-fx (0.53 ms)", 0.53,
              (st.median(t2["ASDA_FIXED"]) - st.median(t2["SDA_FIXED"])) / 1e3, tol=0.05)
    try:
        import numeros_artigo

        eK = numeros_artigo.gain_errors()
        if eK.get("SDA_FIXED") and eK.get("ASDA_FIXED"):
            check("prosa: ASDA-fx mais exato que SDA-fx (2.75x)", 2.75,
                  st.median(eK["SDA_FIXED"]) / st.median(eK["ASDA_FIXED"]), tol=0.05)
    except Exception:
        pass

    cob = os.path.join(OUT, "cobertura_full_v5_6traj.csv")
    if os.path.isfile(cob):
        npf = [float(r["normP_F"]) for r in csv.DictReader(open(cob, encoding="utf-8"))]
        check("prosa: ||P||_F tipica (0.43)", 0.43, st.median(npf), tol=0.02)
        piso_pior = 6 * 2 ** -18 / min(npf)
        check("prosa: tau=1e-3 fica 16x acima do pior piso", 16.0,
              1e-3 / piso_pior, tol=0.05)


def pilha(check):
    """Sec. Implementation: 1376 bytes na entrada do SDA-fx, 1424 no laco de
    duplicacao, 1264 na inversao, 4064 (3.97 KB) no pico aninhado. Fonte:
    outputs/v8/memoria_v8.json, secao 'pilha' (gerar com parse_memory_map.py
    apos 'pio run -e esp32-s2-saola-1'; o env ja passa -fstack-usage)."""
    path = os.path.join(OUT, "v8", "memoria_v8.json")
    if not os.path.isfile(path):
        return
    j = json.load(open(path, encoding="utf-8"))
    p = j.get("pilha")
    if not p:
        print("[INFO] memoria_v8.json sem secao 'pilha': recompilar o firmware de voo "
              "e rodar python python/parse_memory_map.py.")
        return
    ALVO = {"computeGainMatrixSDA_Fixed": 1376, "doubling_loop_q": 1424,
            "invert_q": 1264}
    for e in p["etapas"]:
        alvo = ALVO.get(e["simbolo"])
        if alvo and e["bytes"]:
            check("prosa: quadro de pilha de %s (%d bytes)" % (e["simbolo"], alvo),
                  alvo, e["bytes"], tol=0.01)
    check("prosa: pico de pilha de um solve (4064 bytes)", 4064,
          p["pico_aninhado_bytes"], tol=0.01)
    check("prosa: pico de pilha de um solve (3.97 KB)", 3.97,
          p["pico_aninhado_kb"], tol=0.01)


def estabilizabilidade(check):
    """Sec. SDC factorisation: 'over all 60000 points the controllability matrix
    M_C = [B, AB, A^2 B] maintains full rank 6 (min sigma = 0.6685)'.

    Decimado 1:20 por custo; o minimo e' estavel nessa amostragem (o mesmo 0.6685
    sai de 1:50 e de 1:20), e o posto e' conferido em todos os pontos amostrados.
    """
    cob = os.path.join(OUT, "cobertura_full_v5_6traj.csv")
    if not os.path.isfile(cob):
        return
    try:
        from bench_trajetorias import build_Ad_Bd_Qd_Rd
    except Exception as e:
        print("[INFO] controlabilidade nao conferida (%r)." % (e,))
        return

    rows = list(csv.DictReader(open(cob, encoding="utf-8")))
    smin, posto_cheio = float("inf"), True
    for r in rows[::20]:
        Ad, Bd, _, _ = build_Ad_Bd_Qd_Rd(*[float(r[c]) for c in
                                           ("phi", "theta", "p", "q", "r")])
        Mc = np.hstack([Bd, Ad @ Bd, Ad @ Ad @ Bd])
        smin = min(smin, float(np.linalg.svd(Mc, compute_uv=False)[-1]))
        if np.linalg.matrix_rank(Mc) < 6:
            posto_cheio = False
    check("prosa: min sigma da controlabilidade (0.6685)", 0.6685, smin, tol=0.01)
    check("prosa: M_C tem posto 6 em todos os pontos", 6, 6 if posto_cheio else 0, tol=0.01)


def overflow_de_entrada(check):
    """Sec. Safe region: 'input conversion overflow R_d[0][0] = 64.033 R_scale >
    8192 at R_scale = 127.9'. E' uma identidade: 8192/64.033 = 127.9. Confere-se
    a conta e o coeficiente contra o R_d que o modelo monta em pesos nominais."""
    check("prosa: R_scale de overflow de entrada (127.9)", 127.9, 8192.0 / 64.033, tol=0.005)

    # 64.033 e' o COEFICIENTE de r_scale, nao o R_d[0][0] total: em
    # Rd = Rc*DT + B'QB*(DT^3/3), o segundo termo acrescenta O(q_rate_scale) e
    # leva o total a 65.46 em pesos nominais. O firmware documenta a distincao em
    # experiments/boundary_fine.cpp:171. Comparar o total contra o coeficiente
    # seria comparar grandezas diferentes; o que se confere aqui e' que a
    # fronteira PREVISTA (127.9) cai dentro da fronteira MEDIDA.
    path = os.path.join(OUT, "serial_boundary_fine_B.txt")
    if not os.path.isfile(path):
        return
    taxa = defaultdict(lambda: [0, 0])
    for p in _linhas(path, "SUMMARY"):
        if len(p) != 8 or not p[3].endswith("_FIXED"):
            continue
        try:
            rs, nbk, cnt = float(p[1]), int(p[6]), int(p[7])
        except ValueError:
            continue
        a_ = taxa[rs]
        a_[0] += nbk
        a_[1] += cnt
    if not taxa:
        return
    limiar = 8192.0 / 64.033
    abaixo = [r for r in taxa if r < limiar and taxa[r][0] < taxa[r][1]]
    acima = [r for r in taxa if r > limiar and taxa[r][0] == taxa[r][1]]
    check("prosa: fronteira prevista cai entre a ultima grade parcial e a primeira total",
          1.0, 1.0 if (abaixo and acima and max(abaixo) < limiar < min(acima)) else 0.0,
          tol=0.01)


def norma_absolutos(check):
    """Sec. Stopping test: as duas alternativas ao teste atual, em valor absoluto
    (as razoes 3.1x e 39x ja eram conferidas): 69.9 us com reciproco precomputado
    e 5.5 us com o teste inteiro."""
    path = os.path.join(OUT, "serial_norm_benchmark.txt")
    if not os.path.isfile(path):
        return
    txt = open(path, encoding="utf-8", errors="replace").read()
    for pad, alvo, rot in (
            (r"Norma otimizada \(soft-mul\):\s*([\d.]+) us", 69.9, "reciproco precomputado"),
            (r"Norma inteira \(int64_t\):\s*([\d.]+) us", 5.5, "teste inteiro")):
        m = re.search(pad, txt)
        if m:
            check("prosa: %s (%.1f us)" % (rot, alvo), alvo, float(m.group(1)), tol=0.02)


def todas(check, bat_path, ts_loader, t2=None, t3=None, r2=None):
    jitter(check)
    balanco_das_janelas(check)
    tabela1_exatidao(check, bat_path)
    if t2 and t3:
        tabela2_completa(check, t2, t3)
        derivados_da_tabela1(check, t2, r2)
    deslocamento_e_rastreamento(check)
    magnitude_interna(check)
    alocacoes_de_heap(check)
    pilha(check)
    estabilizabilidade(check)
    overflow_de_entrada(check)
    norma_absolutos(check)
    referencia_float64(check)
    fidelidade_discretizacao(check)
    predictability(check, bat_path)
    teste_de_convergencia(check)
    mapa_de_seguranca(check)
    gamma(check)
    malha_fechada(check)
    memoria(check)
    tolerancia(check, ts_loader)
