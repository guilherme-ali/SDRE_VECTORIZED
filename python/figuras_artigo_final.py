"""
Figuras do artigo DINAME 2027 — geradas a partir dos dados da campanha final.

Fontes de dados (todas medidas no ESP32-S2, exceto onde indicado):
  outputs/serial_capture_bateria_v5_6traj.txt   bateria principal (60000 pts x 12 metodos)
  outputs/serial_tol_qr_sweep_A.txt             tau x Q/R (390 combinacoes x 12 metodos)
  outputs/serial_boundary_fine_B.txt            mapa fino das fronteiras (200 x 10)
  outputs/serial_flightloop_E.txt               ciclo de voo completo (360 s)
  outputs/cobertura_full_v5_6traj.csv           condicionamento (host, scipy)

Convencoes de legibilidade (pedido explicito):
  - largura 6.0 in ~= 152 mm, incluida no .tex com width=150mm -> render ~1:1,
    entao fonte 8-9 pt no grafico aparece como 8-9 pt na pagina.
  - nenhuma legenda sobre area de dados: todas em cantos vazios verificados,
    ou fora dos eixos.
  - paleta segura para daltonismo (Okabe-Ito).
  - escala linear sempre que a faixa permite; log so quando a mensagem E' a
    diferenca de ordem de grandeza (Fig. 3a, Fig. 4).

Uso: python figuras_artigo_final.py [--outdir <dir>]
"""

import argparse
import csv
import hashlib
import json
import subprocess
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")

BAT = os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt")
TOLQR = os.path.join(OUT, "serial_tol_qr_sweep_A.txt")
BOUND = os.path.join(OUT, "serial_boundary_fine_B.txt")
FLIGHT = os.path.join(OUT, "serial_flightloop_E.txt")
COBER = os.path.join(OUT, "cobertura_full_v5_6traj.csv")
TOLSWEEP = os.path.join(OUT, "serial_tolerance_sweep_frobenius.txt")

# Okabe-Ito
C_FLOAT = "#0072B2"   # azul
C_FIXED = "#D55E00"   # vermelhao
C_VI    = "#009E73"   # verde
C_GREY  = "#555555"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

TRAJS = ["T1_espiral", "T2_figura8", "T3_chirp", "T4_degrau_yaw",
         "T5_tilt_alto", "T6_taxa_alta"]
TRAJ_LBL = {"T1_espiral": "T1 spiral", "T2_figura8": "T2 figure-8",
            "T3_chirp": "T3 chirp", "T4_degrau_yaw": "T4 yaw step",
            "T5_tilt_alto": "T5 high tilt", "T6_taxa_alta": "T6 high rate"}

DOUBLING = ["SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA"]
LBL = {"SDA": "SDA", "SDA_SS": "SDA-SS", "ASDA": "ASDA",
       "SDA_SCALED": "SDA-Scaled", "ADDA": "ADDA", "ITERATIVE": "Value iter."}

PERIOD_US = 6000.0

plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# --------------------------------------------------------------------------
# Carga de dados (cacheada em memoria — os arquivos somam ~70 MB)
# --------------------------------------------------------------------------
def load_battery():
    """RUN,traj,k,metodo,time_us,iters,residuo,outcome  (decimado 1:5)."""
    t = defaultdict(list)          # metodo -> [tempo]
    tt = defaultdict(list)         # (metodo,traj) -> [tempo]
    it = defaultdict(list)         # (metodo,traj) -> [iters]
    with open(BAT, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 8:
                continue
            try:
                traj, m, us, its = p[1], p[3], int(p[4]), int(p[5])
            except ValueError:
                continue
            t[m].append(us)
            tt[(m, traj)].append(us)
            it[(m, traj)].append(its)
    return t, tt, it


def load_tolqr():
    """SUMMARY,tau,r,qr,metodo,nconv,nbud,nbrk,count,mean_us,mean_res."""
    d = defaultdict(dict)
    with open(TOLQR, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("SUMMARY"):
                continue
            p = line.strip().split(",")
            if len(p) != 11:
                continue
            _, tau, rs, qr, m, nc, nb, nbk, cnt, mus, mres = p
            d[(rs, qr, m)][tau] = (int(nc), int(cnt), float(mus), float(mres))
    return d


def load_boundary():
    """SUMMARY,r,qr,metodo,nconv,nbud,nbrk,count -> brk% por (r,metodo)."""
    agg = defaultdict(lambda: [0, 0])
    with open(BOUND, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("SUMMARY"):
                continue
            p = line.strip().split(",")
            if len(p) != 8:
                continue
            _, rs, qr, m, nc, nb, nbk, cnt = p
            a = agg[(float(rs), m)]
            a[0] += int(nbk)
            a[1] += int(cnt)
    return agg


def load_flight(paths=None):
    """Consolida N execucoes do ciclo de voo numa unica populacao de ciclos.

    Ao contrario dos benchmarks de solver, que repetem bit-a-bit, o laco de voo
    varia entre execucoes: I2C, WiFi e os blocos de impressao deslocam a cauda.
    Cinco capturas do mesmo binario deram de 0 a 6 ciclos acima do periodo. Uma
    figura feita de UMA captura reporta, portanto, a amostra que calhou de ser
    gravada, e nao o comportamento do sistema.

    Somar os histogramas de 50 us e' exato (os bins sao os mesmos em todas as
    execucoes), entao a ECDF agregada e' a ECDF real do conjunto de ciclos; as
    amostras de estagio de todas as execucoes entram na mesma lista.
    """
    if paths is None:
        paths = [FLIGHT]
    names = [("LEDs", "LEDs"), ("Bateria", "Battery"), ("WiFi/UDP", "WiFi/UDP"),
             ("Leitura MPU", "IMU read"), ("Filtro Madgwick", "Madgwick"),
             (r"C.lc. .ngulos", "Euler"), ("Matriz Sistema", "SDC matrix"),
             (r"LQR .Ganhos.", "DARE solve"), (r"L.gica Controle", "Control law"),
             (r"C.lc. Omega.", "Mixer"), ("Set Motores", "Motor write")]
    proc, stages = [], defaultdict(list)
    hist_soma = None
    meta = {"execucoes": 0, "arquivos": [], "ciclos_por_execucao": [],
            "estouros_por_execucao": []}

    for path in paths:
        txt = open(path, encoding="utf-8", errors="replace").read()
        blocks = txt.split("STATUS DO SISTEMA")
        if len(blocks) < 2:
            print("  [!] sem blocos de status, ignorada:", os.path.basename(path))
            continue
        m_hist = re.search(r"HIST_PROC_50US:([0-9,]+)", blocks[-1])
        if m_hist:
            counts = [int(x) for x in m_hist.group(1).split(",") if x]
            if sum(counts) > 0:
                if hist_soma is None:
                    hist_soma = list(counts)
                else:
                    if len(counts) > len(hist_soma):
                        hist_soma += [0] * (len(counts) - len(hist_soma))
                    for i, c in enumerate(counts):
                        hist_soma[i] += c
                meta["ciclos_por_execucao"].append(sum(counts))
                meta["estouros_por_execucao"].append(sum(counts[120:]))  # bin 120 = 6.00 ms
        for b in blocks[1:]:
            m = re.search(r"Tempo_Processamento:\s*(\d+)", b)
            if m:
                proc.append(int(m.group(1)))
            for pat, lbl in names:
                mm = re.search(pat + r":\s*(\d+)\s*.s", b)
                if mm:
                    stages[lbl].append(int(mm.group(1)))
        meta["execucoes"] += 1
        meta["arquivos"].append(os.path.basename(path))
        meta.setdefault("caminhos", []).append(path)

    hist_cdf_p = hist_cdf_y = None
    if hist_soma:
        tot = sum(hist_soma)
        hist_cdf_p = (np.arange(len(hist_soma)) * 50) / 1000.0  # ms
        hist_cdf_y = 100.0 * np.cumsum(hist_soma) / tot
        meta["ciclos"] = tot
        meta["estouros"] = sum(hist_soma[120:])
        meta["hist"] = hist_soma
    return proc, stages, (hist_cdf_p, hist_cdf_y), meta


def load_coverage():
    d = defaultdict(lambda: defaultdict(list))
    with open(COBER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in ("rho", "cond_IGP", "normP_F"):
                d[r["traj"]][c].append(float(r[c]))
    return d


def load_tolerance_sweep(path=TOLSWEEP):
    """RUN,<0a|0b>,tol,traj,k,metodo,time_us,iters,residuo,outcome[,rel_step[,bit_exact]].

    Tolerante a três formatos de captura (10/11/12 campos): capturas antigas
    não têm rel_step nem bit_exact (pré-instrumentação); capturas de transição
    têm rel_step mas não bit_exact (o formato encontrado em
    outputs/serial_tolerance_sweep_frobenius.txt no início da v8 — produzido
    por um código que não estava mais na árvore); capturas pós-v8 têm os dois.
    Onde o campo não existe, o valor fica None — os consumidores devem tratar.

    Retorna dict (exp, metodo) -> tol -> lista de dicts com time_us/iters/
    residuo/outcome/step/bit_exact, para os dois sub-experimentos 0a (família
    de duplicação) e 0b (value iteration).
    """
    out = defaultdict(lambda: defaultdict(list))
    n_missing_step = 0
    n_missing_bitexact = 0
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 10:
                continue
            exp = p[1]
            try:
                tol = float(p[2])
                traj, k, metodo = p[3], int(p[4]), p[5]
                time_us, iters, resid, outcome = int(p[6]), int(p[7]), float(p[8]), int(p[9])
            except ValueError:
                continue
            step = None
            bit_exact = None
            if len(p) >= 11:
                try:
                    step = float(p[10])
                except ValueError:
                    step = None
            else:
                n_missing_step += 1
            if len(p) >= 12:
                try:
                    bit_exact = bool(int(p[11]))
                except ValueError:
                    bit_exact = None
            else:
                n_missing_bitexact += 1
            out[(exp, metodo)][tol].append({
                "traj": traj, "k": k, "time_us": time_us, "iters": iters,
                "resid": resid, "outcome": outcome, "step": step, "bit_exact": bit_exact,
            })
    if n_missing_step or n_missing_bitexact:
        print("  aviso: serial_tolerance_sweep_frobenius.txt tem %d linhas sem rel_step e "
              "%d sem bit_exact (captura anterior ao firmware instrumentado — "
              "recapturar com 'python python/run_experiments.py --only tolerancia --force' "
              "para o painel (c) completo)." % (n_missing_step, n_missing_bitexact))
    return out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


# --------------------------------------------------------------------------
# Fig. 1 — as seis trajetorias
# --------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Procedencia das figuras
# ---------------------------------------------------------------------------
_PROV_REGISTRO = {}


def _git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=REPO, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        sujo = subprocess.call(["git", "diff", "--quiet"], cwd=REPO,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) != 0
        return rev + ("-dirty" if sujo else "")
    except Exception:
        return "desconhecido"


def _sha12(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "ausente"
    return h.hexdigest()[:12]


def _metadados(nome_pdf, fontes):
    """Metadados PDF com commit e hash de cada captura de origem.

    O backend PDF do matplotlib aceita Title/Author/Subject/Keywords/Creator;
    e' o unico lugar que sobrevive ao arquivo sair do repositorio.
    """
    fontes = [f for f in fontes if f]
    itens = ["%s=%s" % (os.path.relpath(f, REPO).replace(os.sep, "/"), _sha12(f))
             for f in fontes]
    rev = _git_rev()
    _PROV_REGISTRO[nome_pdf] = {"commit": rev, "fontes": itens}
    return {
        "Title": nome_pdf,
        "Author": rev,
        "Subject": "; ".join(itens) if itens else "sem fonte declarada",
        "Keywords": "python/figuras_artigo_final.py",
        "Creator": "SDRE_VECTORIZED / figuras_artigo_final.py",
    }


def _grava_registro_proveniencia():
    destino = os.path.join(OUT, "v8", "figuras_procedencia.json")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(_PROV_REGISTRO, f, indent=2, ensure_ascii=False)
    print("  procedencia das figuras:", os.path.relpath(destino, REPO))


def fig1_envelope(outdir, cover):
    """(a) envelope de operacao percorrido; (b) condicionamento numerico
    resultante. A mensagem e' que (a) varia muito entre trajetorias e (b)
    quase nao varia."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "python"))
    import trajetorias as trj
    dados = trj.gerar_todas()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.28))
    for i, name in enumerate(TRAJS):
        d = dados[name]
        tilt = np.rad2deg(np.hypot(np.asarray(d["phi"]), np.asarray(d["theta"])))
        rate = np.rad2deg(np.sqrt(np.asarray(d["p"]) ** 2 + np.asarray(d["q"]) ** 2
                                  + np.asarray(d["r"]) ** 2))
        sl = slice(None, None, 11)
        ax1.scatter(tilt[sl], np.maximum(rate[sl], 1.0), s=1.2, alpha=0.30,
                    color=PALETTE[i], edgecolors="none", rasterized=True)
        c = cover[name]
        ax2.scatter(c["normP_F"][::11], c["cond_IGP"][::11], s=1.2, alpha=0.30,
                    color=PALETTE[i], edgecolors="none", rasterized=True)
    ax1.set_yscale("log")
    ax1.set_xlabel("tilt magnitude (deg)")
    ax1.set_ylabel(r"body-rate magnitude (deg/s)")
    ax1.set_title("(a) operating envelope covered", fontsize=8.5)
    ax1.set_xlim(-3, 95)
    ax1.set_ylim(0.8, 6000)
    ax2.set_xlabel(r"$\|\mathbf{P}\|_F$")
    ax2.set_ylabel(r"$\mathrm{cond}(\mathbf{I}+\mathbf{G}\mathbf{P})$")
    ax2.set_title("(b) resulting numerical conditioning", fontsize=8.5)
    ax2.set_xlim(0.34, 0.52)
    ax2.set_ylim(5.3, 7.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=PALETTE[i],
                      label=TRAJ_LBL[t]) for i, t in enumerate(TRAJS)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.045), columnspacing=1.1,
               handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    p = os.path.join(outdir, "fig1_envelope.pdf")
    fig.savefig(p, dpi=300, metadata=_metadados("fig1_envelope.pdf", [COBER]))
    plt.close(fig)
    print("  ok", p)


# --------------------------------------------------------------------------
# Fig. 2 — tempo de execucao, familia de duplicacao (float vs ponto fixo)
# --------------------------------------------------------------------------
def fig2_timing(outdir, tall):
    med_f = [st.median(tall[m]) / 1000.0 for m in DOUBLING]
    med_x = [st.median(tall[m + "_FIXED"]) / 1000.0 for m in DOUBLING]
    hi_f = [pct(tall[m], 0.999) / 1000.0 for m in DOUBLING]
    hi_x = [pct(tall[m + "_FIXED"], 0.999) / 1000.0 for m in DOUBLING]

    y = np.arange(len(DOUBLING))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 2.28))
    ax.barh(y + h / 2, med_f, height=h, color=C_FLOAT, label="single precision",
            xerr=[[0] * len(med_f), np.array(hi_f) - np.array(med_f)],
            error_kw=dict(ecolor="0.25", lw=0.8, capsize=2))
    ax.barh(y - h / 2, med_x, height=h, color=C_FIXED, label="Q13.18 fixed point",
            xerr=[[0] * len(med_x), np.array(hi_x) - np.array(med_x)],
            error_kw=dict(ecolor="0.25", lw=0.8, capsize=2))
    for i, (a, b) in enumerate(zip(med_f, med_x)):
        ax.text(a + 0.15, i + h / 2, "%.2f" % a, va="center", fontsize=7.5, color=C_FLOAT)
        ax.text(b + 0.15, i - h / 2, "%.2f" % b, va="center", fontsize=7.5, color=C_FIXED)
        ax.text(12.05, i, r"$\times$%.2f" % (a / b), va="center", ha="left",
                fontsize=8, fontweight="bold", color="0.15")
    ax.set_yticks(y)
    ax.set_yticklabels([LBL[m] for m in DOUBLING])
    ax.set_xlabel("solve time (ms) — bar: median, whisker: 99.9th percentile")
    ax.set_xlim(0, 13.9)
    ax.set_ylim(-0.65, len(DOUBLING) - 0.05)
    ax.text(12.05, len(DOUBLING) - 0.30, "speed-up", fontsize=8,
            fontweight="bold", color="0.15", ha="left", va="center")
    # legenda FORA da area de dados (acima dos eixos)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.01), ncol=2,
              frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_timing.pdf")
    fig.savefig(p, metadata=_metadados("fig2_timing.pdf", [BAT]))
    plt.close(fig)
    print("  ok", p)


# --------------------------------------------------------------------------
# Fig. 3 — previsibilidade do custo (achado central)
# --------------------------------------------------------------------------
def fig3_predictability(outdir, tt, it):
    # (b) variacao media do estado entre pontos consecutivos
    dx = defaultdict(list)
    prev = None
    with open(COBER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t, k = r["traj"], int(r["k"])
            x = (float(r["phi"]), float(r["theta"]),
                 float(r["p"]), float(r["q"]), float(r["r"]))
            if prev and prev[0] == t and k == prev[1] + 1:
                dx[t].append(math.dist(x, prev[2]))
            prev = (t, k, x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.42))

    # --- (a) distribuicao do tempo por trajetoria ---
    pos = np.arange(len(TRAJS))
    w = 0.34
    d_sda = [np.array(tt[("SDA_FIXED", t)]) / 1000.0 for t in TRAJS]
    d_vi = [np.array(tt[("ITERATIVE_FIXED", t)]) / 1000.0 for t in TRAJS]
    bp1 = ax1.boxplot(d_sda, positions=pos - w / 2, widths=w, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", lw=1.0),
                      whis=(1, 99))
    bp2 = ax1.boxplot(d_vi, positions=pos + w / 2, widths=w, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", lw=1.0),
                      whis=(1, 99))
    for b in bp1["boxes"]:
        b.set_facecolor(C_FIXED)
        b.set_linewidth(0.6)
    for b in bp2["boxes"]:
        b.set_facecolor(C_VI)
        b.set_linewidth(0.6)
    ax1.axhline(PERIOD_US / 1000.0, color="0.2", ls="--", lw=1.0)
    ax1.set_yscale("log")
    ax1.set_ylim(0.6, 60)
    ax1.set_xticks(pos)
    ax1.set_xticklabels([TRAJ_LBL[t].split()[0] for t in TRAJS], fontsize=7.5)
    ax1.set_ylabel("solve time (ms), log scale")
    ax1.set_title("(a) cost per trajectory", fontsize=8.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax1.legend(handles=[Patch(facecolor=C_FIXED, label="SDA-fx"),
                        Patch(facecolor=C_VI, label="Value iter.-fx"),
                        Line2D([], [], color="0.2", ls="--", lw=1.0,
                               label="6.0 ms control period")],
               loc="upper left", frameon=True, framealpha=0.95,
               edgecolor="0.8", fontsize=7.0, handlelength=1.8,
               borderpad=0.35, labelspacing=0.3)

    # --- (b) iteracoes vs variacao do ponto de operacao ---
    xs = [st.mean(dx[t]) for t in TRAJS]
    y_vi = [st.median(it[("ITERATIVE_FIXED", t)]) for t in TRAJS]
    y_sda = [st.median(it[("SDA_FIXED", t)]) for t in TRAJS]
    o = np.argsort(xs)
    xs = np.array(xs)[o]
    y_vi = np.array(y_vi)[o]
    y_sda = np.array(y_sda)[o]
    lbl = [TRAJ_LBL[TRAJS[i]].split()[0] for i in o]
    ax2.plot(xs, y_vi, "o-", color=C_VI, lw=1.2, ms=4.5, label="Value iter.-fx")
    ax2.plot(xs, y_sda, "s-", color=C_FIXED, lw=1.2, ms=4.5, label="SDA-fx")
    for x, yv, l in zip(xs, y_vi, lbl):
        ax2.annotate(l, (x, yv), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=6.8, color="0.25")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 42)
    ax2.set_xlabel(r"mean $\|\Delta x\|$ between consecutive points")
    ax2.set_xlim(1.3e-3, 45)
    ax2.set_ylabel("median iterations")
    ax2.set_title("(b) iterations vs. operating-point motion", fontsize=8.5)
    # legenda no canto que fica vazio (as curvas sobem da esquerda p/ direita)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.8",
               fontsize=7.5)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_predictability.pdf")
    fig.savefig(p, metadata=_metadados("fig3_predictability.pdf", [BAT]))
    plt.close(fig)
    print("  ok", p)


# --------------------------------------------------------------------------
# Fig. 4 — tolerancia pedida x acuracia atingida, e o piso de quantizacao
# --------------------------------------------------------------------------
def fig4_tolerance(outdir, tq, cover):
    taus = ["1e-02", "3e-03", "1e-03", "3e-04", "1e-04", "3e-05"]
    tvals = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
    cell = ("1e+00", "1e+00")  # pesos nominais

    # piso de quantizacao n*2^-s/||P||_F sobre TODOS os pontos medidos
    allP = [v for t in cover for v in cover[t]["normP_F"]]
    floor_lo = 6 * (2.0 ** -18) / max(allP)
    floor_hi = 6 * (2.0 ** -18) / min(allP)

    series = [("SDA_FIXED", "SDA-fx", C_FIXED, "-", "o"),
              ("ASDA_FIXED", "ASDA-fx", "#CC79A7", "-", "s"),
              ("ITERATIVE_FIXED", "Value iter.-fx", C_VI, "-", "^"),
              ("SDA", "SDA (float)", C_FLOAT, "--", "o"),
              ("ITERATIVE", "Value iter. (float)", "#56B4E9", "--", "^")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.38))
    for key, lab, col, ls, mk in series:
        d = tq.get((cell[0], cell[1], key), {})
        y = [d[t][3] for t in taus if t in d]
        x = [tv for t, tv in zip(taus, tvals) if t in d]
        if y:
            ax1.plot(x, y, ls=ls, marker=mk, color=col, lw=1.2, ms=4, label=lab)
    ax1.axvspan(floor_lo, floor_hi, color="0.75", alpha=0.55, lw=0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel(r"requested tolerance $\tau$")
    ax1.set_ylabel("achieved DARE residual")
    ax1.set_title("(a) accuracy does not follow the tolerance", fontsize=8.5)
    ax1.set_ylim(3e-7, 1.2)
    ax1.text(math.sqrt(floor_lo * floor_hi), 0.34, "quantisation\nfloor",
             fontsize=6.8, color="0.25", ha="center", va="center")

    # (b) custo em tempo de apertar tau — banda segura inteira
    band_r = ["1e-01", "1e+00", "1e+01", "1e+02"]
    band_q = ["1e-02", "1e-01", "1e+00", "1e+01", "1e+02"]
    for key, lab, col, ls, mk in [("SDA_FIXED", "SDA-fx", C_FIXED, "-", "o"),
                                  ("ASDA_FIXED", "ASDA-fx", "#CC79A7", "-", "s"),
                                  ("ADDA_FIXED", "ADDA-fx", "#E69F00", "-", "d")]:
        y = []
        for t in taus:
            v = [tq[(r, q, key)][t][2] for r in band_r for q in band_q
                 if t in tq.get((r, q, key), {}) and tq[(r, q, key)][t][0] > 0]
            y.append(st.mean(v) / 1000.0 if v else np.nan)
        ax2.plot(tvals, y, ls=ls, marker=mk, color=col, lw=1.2, ms=4, label=lab)
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel(r"requested tolerance $\tau$")
    ax2.set_ylabel("mean solve time (ms)")
    ax2.set_title("(b) but the cost does", fontsize=8.5)
    ax2.set_ylim(3.4, 6.6)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.8",
               fontsize=7.5)
    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=5, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.1, handlelength=2.0,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    p = os.path.join(outdir, "fig4_tolerance.pdf")
    fig.savefig(p, metadata=_metadados("fig4_tolerance.pdf", [TOLQR, COBER]))
    plt.close(fig)
    print("  ok", p)
    return floor_lo, floor_hi


# --------------------------------------------------------------------------
# Fig. 4 v8 — tres paineis: residuo, tempo E o passo medido na terminacao
# (novo painel (c)), todos da MESMA fonte (serial_tolerance_sweep_frobenius.txt,
# pesos nominais, tau de 1e-2 a 1e-6), em vez de misturar tolerance_sweep com
# tol_qr_sweep como a Fig. 4 do v7 fazia (ver plano da v8).
# --------------------------------------------------------------------------
_V8_SERIES = [
    ("SDA_FIXED", "0a", "SDA-fx", C_FIXED, "-", "o"),
    ("ASDA_FIXED", "0a", "ASDA-fx", "#CC79A7", "-", "s"),
    ("ITERATIVE_FIXED", "0b", "Value iter.-fx", C_VI, "-", "^"),
    ("SDA", "0a", "SDA (float)", C_FLOAT, "--", "o"),
    ("ITERATIVE", "0b", "Value iter. (float)", "#56B4E9", "--", "^"),
]


def _tolsweep_series_stats(ts, exp, metodo, taus):
    """Para (exp,metodo), agrega por tau: mediana do residuo/passo (só sobre
    convergidos — mesma convenção do SUMMARY do firmware, ver TolStats::add()
    em experiments/tolerance_sweep.cpp) e tempo médio (ms), mais a fração de
    passos bit-exatos (ΔH==0), essa sobre TODAS as tentativas daquele tau."""
    per_tau = ts.get((exp, metodo), {})
    out = {}
    for tau in taus:
        rows = per_tau.get(tau, [])
        if not rows:
            continue
        conv = [r for r in rows if r["outcome"] == 0]
        resid = [r["resid"] for r in conv]
        steps = [r["step"] for r in conv if r["step"] is not None]
        # bit_exact==None (captura pre-instrumentacao): usa step==0.0 exato como
        # proxy — um passo de Frobenius relativo em float só é exatamente zero
        # quando todo ΔH_ij==0.0, o que é precisamente o que bit_exact mede.
        n_bit_exact = sum(
            1 for r in rows
            if r["bit_exact"] or (r["bit_exact"] is None and r["step"] == 0.0)
        )
        n_bit_exact_known = sum(1 for r in rows if r["bit_exact"] is not None)
        out[tau] = {
            "n_total": len(rows),
            "n_conv": len(conv),
            "resid_median": st.median(resid) if resid else None,
            "time_ms_mean": (st.mean(r["time_us"] for r in conv) / 1000.0) if conv else None,
            "step_median": st.median(steps) if steps else None,
            "n_bit_exact": n_bit_exact,
            "n_bit_exact_known": n_bit_exact_known,
        }
    return out


def fig4_tolerance_v8(outdir, ts, cover):
    """3 painéis lado a lado, mesma fonte de dados (tau nominal, 1e-2..1e-6):
    (a) residuo DARE atingido, (b) tempo médio, (c) passo medido na
    terminação (novo) — com a faixa do piso analítico e marcadores 'x' nos
    pontos bit-exatos (ΔH==0 em Q13.18, não representável em escala log)."""
    taus = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    have_any_step = any(
        r.get("step") is not None
        for (exp, metodo) in ts for tau_rows in ts[(exp, metodo)].values() for r in tau_rows
    )
    have_any_bitexact = any(
        r.get("bit_exact") is not None
        for (exp, metodo) in ts for tau_rows in ts[(exp, metodo)].values() for r in tau_rows
    )
    if not have_any_step:
        print("  aviso: sem rel_step em serial_tolerance_sweep_frobenius.txt — "
              "painel (c) da fig4_tolerance_v8 ficará vazio. Recapturar com o "
              "firmware instrumentado (ver docstring de load_tolerance_sweep()).")

    allP = [v for t in cover for v in cover[t]["normP_F"]]
    floor_lo = 6 * (2.0 ** -18) / max(allP)
    floor_hi = 6 * (2.0 ** -18) / min(allP)

    # 6.7 in = 170 mm = \textwidth da classe xxidiname: incluida no .tex com
    # width=170mm, renderiza 1:1, entao 8.5 pt no grafico sai 8.5 pt na pagina.
    # (O v7 autorava a 6.0 in e incluia com width=108mm — reducao a 71%, que
    # fazia a fonte cair para ~6 pt na pagina impressa.)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.7, 2.45))

    y_floor = 3e-7  # piso do eixo (c) p/ plotar step==0 (bit-exato), não representável em log
    zero_fracs = []  # (tau, metodo, n_bit_exact, n_total) — p/ nota de rodapé, evita poluir o grafico
    for series_idx, (metodo, exp, lab, col, ls, mk) in enumerate(_V8_SERIES):
        s = _tolsweep_series_stats(ts, exp, metodo, taus)
        xs_a = [t for t in taus if t in s and s[t]["resid_median"] is not None]
        ys_a = [s[t]["resid_median"] for t in xs_a]
        if ys_a:
            ax1.plot(xs_a, ys_a, ls=ls, marker=mk, color=col, lw=1.2, ms=4, label=lab)

        xs_b = [t for t in taus if t in s and s[t]["time_ms_mean"] is not None]
        ys_b = [s[t]["time_ms_mean"] for t in xs_b]
        if ys_b:
            ax2.plot(xs_b, ys_b, ls=ls, marker=mk, color=col, lw=1.2, ms=4, label=lab)

        # painel (c): pontos com passo>0 em escala log normal; passo==0 (bit-exato)
        # não é representável em log — vai para o piso do eixo com marcador 'x'.
        xs_pos, ys_pos = [], []
        xs_zero = []
        for t in taus:
            if t not in s or s[t]["step_median"] is None:
                continue
            v = s[t]["step_median"]
            if v > 0:
                xs_pos.append(t)
                ys_pos.append(v)
            else:
                xs_zero.append(t)
        if xs_pos:
            ax3.plot(xs_pos, ys_pos, ls=ls, marker=mk, color=col, lw=1.2, ms=4, label=lab)
        if xs_zero:
            # leve jitter vertical por série (log scale) — evita marcadores 'x' de séries
            # diferentes empilhados exatamente no mesmo pixel no mesmo tau.
            y_jit = y_floor * (1.0 + 0.22 * series_idx)
            ax3.plot(xs_zero, [y_jit] * len(xs_zero), ls="none", marker="x", color=col,
                     ms=6, mew=1.5, zorder=5)
            for t in xs_zero:
                zero_fracs.append((t, lab, s[t]["n_bit_exact"], s[t]["n_total"]))

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel(r"requested tolerance $\tau$")
    ax1.set_ylabel("achieved DARE residual")
    ax1.set_title("(a) achieved residual", fontsize=8.5)

    ax2.set_xscale("log")
    # log em y: a iteracao de valor varre 1->120 ms enquanto a familia doubling
    # se move de 3.5 a 5.9 ms. Em escala linear a subida do doubling (o custo de
    # apertar tau, que e' o ponto do painel) fica invisivel no rodape do eixo.
    ax2.set_yscale("log")
    ax2.invert_xaxis()
    ax2.axhline(PERIOD_US / 1000.0, color="0.35", ls="--", lw=0.9, zorder=0)
    # canto inferior direito: abaixo das curvas fixed-point (~4.4 ms) e longe
    # das de value iteration (>100 ms) — o unico canto do painel sem dado.
    ax2.text(taus[-1], 1.35, "6.0 ms period", fontsize=6.3,
             color="0.35", ha="right", va="center")
    ax2.set_xlabel(r"requested tolerance $\tau$")
    ax2.set_ylabel("mean solve time (ms)")
    ax2.set_title("(b) mean solve time", fontsize=8.5)

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.invert_xaxis()
    ax3.set_ylim(y_floor * 0.6, 1.5)
    ax3.axhspan(floor_lo, floor_hi, color="0.75", alpha=0.55, lw=0, zorder=0)
    ax3.plot(taus, taus, color="0.5", ls=":", lw=1.0, zorder=1)
    ax3.text(taus[1], taus[1] * 1.6, r"$y=\tau$", fontsize=6.5, color="0.4",
             ha="left", va="bottom", rotation=32)
    ax3.text(math.sqrt(floor_lo * floor_hi), floor_hi * 2.6, "quantisation\nfloor",
             fontsize=6.5, color="0.25", ha="center", va="bottom")
    ax3.set_xlabel(r"requested tolerance $\tau$")
    ax3.set_ylabel(r"measured step $\|\Delta \mathbf{H}\|_F/\|\mathbf{H}\|_F$")
    ax3.set_title("(c) step at termination", fontsize=8.5)

    # ticks explicitos: com 3 paineis estreitos o locator automatico do log
    # mostrava so 1e-3 e 1e-5, escondendo os extremos da varredura.
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([1e-2, 1e-4, 1e-6])
        ax.set_xticks([1e-3, 1e-5], minor=True)
        ax.set_xticklabels([], minor=True)

    from matplotlib.lines import Line2D
    h1, l1 = ax1.get_legend_handles_labels()
    h1 = h1 + [Line2D([], [], ls="none", marker="x", color="0.25", ms=6, mew=1.5)]
    l1 = l1 + [r"bit-exact ($\Delta H=0$), panel (c)"]
    fig.legend(h1, l1, loc="lower center", ncol=3, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.09), columnspacing=1.1, handlelength=2.0,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    p = os.path.join(outdir, "fig4_tolerance_v8.pdf")
    fig.savefig(p, metadata=_metadados("fig4_tolerance_v8.pdf", [TOLSWEEP, COBER]))
    plt.close(fig)
    print("  ok", p)
    if not have_any_bitexact:
        print("  aviso: sem bit_exact em serial_tolerance_sweep_frobenius.txt — os marcadores "
              "'x' do painel (c) usaram step==0.0 como proxy (equivalente na prática, mas a "
              "fração relatada abaixo é inferida, não medida diretamente). Recapturar para o dado real.")
    if zero_fracs:
        print("  passos bit-exatos (painel c, fora do grafico p/ nao poluir):")
        for t, lab, n_be, n_tot in zero_fracs:
            print("    tau=%.0e  %-16s  %d/%d" % (t, lab, n_be, n_tot))
    return floor_lo, floor_hi


# --------------------------------------------------------------------------
# Fig. 5 — envelope seguro das matrizes de peso
# --------------------------------------------------------------------------
def fig5_safety(outdir, agg):
    """(a) envelope completo com as duas fronteiras; (b) zoom na banda util,
    onde a diferenca entre variantes e' o resultado de interesse."""
    methods = [("SDA_FIXED", "SDA-fx", C_FIXED, "o"),
               ("SDA_SS_FIXED", "SDA-SS-fx", "#E69F00", "v"),
               ("ASDA_FIXED", "ASDA-fx", "#CC79A7", "s"),
               ("SDA_SCALED_FIXED", "SDA-Scaled-fx", "#56B4E9", "d"),
               ("ADDA_FIXED", "ADDA-fx", C_VI, "^")]
    rs = sorted({r for (r, m) in agg})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.33),
                                   gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    for key, lab, col, mk in methods:
        y = [100.0 * agg[(r, key)][0] / agg[(r, key)][1] for r in rs]
        ax1.plot(rs, y, marker=mk, color=col, lw=1.1, ms=3.0, label=lab, alpha=0.9)
        sub = [(r, v) for r, v in zip(rs, y) if 0.06 <= r <= 20]
        ax2.plot([r for r, _ in sub], [v for _, v in sub], marker=mk, color=col,
                 lw=1.2, ms=4.0, alpha=0.9)
    ax1.axvline(127.9, color="0.2", ls="--", lw=1.1)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$R_\mathrm{scale}$ (nominal $=1$)")
    ax1.set_ylabel("breakdown rate (%)")
    ax1.set_ylim(-5, 112)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_title("(a) full weighting range", fontsize=8.5)
    ax1.annotate("$\\mathbf{R}_d$ input\noverflow at $127.9$",
                 xy=(127.9, 50), xytext=(1.5, 60), fontsize=6.8, color="0.15",
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.7))
    ax1.text(1.1e-3, 68, r"$\mathbf{G}_0$ setup overflow", fontsize=6.8,
             color="0.15", ha="left")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$R_\mathrm{scale}$ (usable band)")
    ax2.set_ylabel("breakdown rate (%)")
    ax2.set_ylim(-1.2, 26)
    ax2.set_title("(b) zoom: usable band", fontsize=8.5)
    ax2.annotate("ASDA-fx", xy=(1.1, 0.45), xytext=(2.6, 13.5), fontsize=7.5,
                 color="#CC79A7", fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=0.9))
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.045), columnspacing=1.0,
               handlelength=1.8, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = os.path.join(outdir, "fig5_safety_envelope.pdf")
    fig.savefig(p, metadata=_metadados("fig5_safety_envelope.pdf", [BOUND]))
    plt.close(fig)
    print("  ok", p)


# --------------------------------------------------------------------------
# Fig. 5 (malha fechada) — rastreamento e penalidade de custo.
#
# A versao anterior deste PDF (Figures/fig5_closed_loop.pdf, usada no v7) NAO
# tinha script no repositorio que a reproduzisse, e trazia "<= 0.26%" fixo no
# titulo, vindo de uma rodada antiga (malha_fechada_trajetorias_v4.csv). Esta
# funcao regenera a figura a partir dos dados correntes:
#   outputs/malha_fechada_v6_6traj.csv      penalidade por trajetoria (painel b)
#   outputs/malha_fechada_serie_T2.csv      serie temporal de T2 (painel a),
#     produzida por python/malha_fechada_trajetorias.py --series-traj T2_figura8
# O limite nao e' mais escrito a mao: e' calculado do proprio dado e impresso.
# --------------------------------------------------------------------------
MALHA = os.path.join(OUT, "malha_fechada_v6_6traj.csv")
# T3 (chirp), nao T2: T2 e' a trajetoria mais facil das seis (RMS 0.098 deg nas
# duas aritmeticas), onde as curvas coincidem trivialmente e o grafico nao prova
# nada. T3 e' a mais exigente entre as de modelo discreto fiel, e e' a mesma em
# que o artigo ancora a alegacao de previsibilidade — usar a mesma trajetoria
# nas duas figuras mantem o argumento coerente.
SERIE_TRAJ = "T3_chirp"
SERIE_CSV = os.path.join(OUT, "malha_fechada_serie_%s.csv" % SERIE_TRAJ)

# Ordem de linhas do heatmap do painel (b). Nao ha cor por controlador aqui: a
# versao em barras agrupadas (6 controladores x 5 trajetorias = 30 barras) era
# ilegivel — o leitor nao conseguia casar barra com trajetoria, e a paleta
# Okabe-Ito nao tem 6 tons categoricos seguros. O heatmap troca "casar cor com
# legenda" por "ler o numero na celula", e ainda comporta as SEIS trajetorias.
FX_CTRLS = [("SDA_FIXED", "SDA-fx"),
            ("SDA_SS_FIXED", "SDA-SS-fx"),
            ("ASDA_FIXED", "ASDA-fx"),
            ("SDA_SCALED_FIXED", "SDA-Scaled-fx"),
            ("ADDA_FIXED", "ADDA-fx"),
            ("ITERATIVE_FIXED", "Value iter.-fx")]

# Divergente Okabe-Ito: azul (negativo) -> cinza neutro -> vermelhao (positivo).
# Mesmos polos ja usados no artigo para float/fixed; validados em
# scripts/validate_palette.js (DeltaE 21.9 protan, 31.2 visao normal, todos os
# checks PASS). Midpoint neutro, nao um terceiro matiz.
_DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_div", ["#0072B2", "#8FBFDD", "#F0EFEC", "#EBA77C", "#D55E00"])


# Janela escolhida a partir do dado, nao a olho: em 12-14.5 s o chirp esta a
# ~1.9 Hz (cerca de 5 ciclos, ainda legiveis), o atraso de rastreamento e' 5.2 deg
# sobre uma amplitude de 27.7 deg — visivel, mostrando o controlador sob carga —
# enquanto a distancia float64<->Q13.18 e' 0.12 deg, 2% disso. Janelas mais tarde
# (>=28 s) empilham 8+ ciclos e o atraso passa de 13 deg, o que desvia a atencao
# do ponto do painel; janelas antes de 8 s sao faceis demais.
def fig5_closed_loop(outdir, window=(12.0, 14.5)):
    if not (os.path.isfile(MALHA) and os.path.isfile(SERIE_CSV)):
        print("  [pulado] fig5_closed_loop: faltam %s e/ou %s "
              "(rodar python/malha_fechada_trajetorias.py)" %
              (os.path.basename(MALHA), os.path.basename(SERIE_CSV)))
        return None

    J = defaultdict(dict)
    with open(MALHA, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            J[r["traj"]][r["controller"]] = float(r["J_total"])

    ser = defaultdict(list)
    with open(SERIE_CSV, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            for c in rd.fieldnames:
                ser[c].append(float(r[c]))

    # 6.0 in como as demais figuras: incluidas todas com a mesma largura no .tex,
    # autorar todas no mesmo tamanho mantem a fonte impressa igual entre elas.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.32),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.30]))

    # ---- (a) rastreamento numa janela do chirp -------------------------------
    # Janela curta, nao os 60 s: a 60 s o chirp vira um borrao solido e nada se
    # le. A janela default fica na metade alta da varredura, onde rastrear e'
    # mais dificil — se as duas aritmeticas coincidem ali, coincidem no resto.
    t = np.array(ser["t"])
    m = (t >= window[0]) & (t <= window[1])
    ax1.plot(t[m], np.array(ser["phi_ref_deg"])[m], color="0.30", lw=2.4,
             solid_capstyle="round", label="commanded", zorder=1)
    if "phi_SDA_float64_deg" in ser:
        ax1.plot(t[m], np.array(ser["phi_SDA_float64_deg"])[m], color=C_FLOAT,
                 lw=1.3, label="float64", zorder=2)
    if "phi_SDA_FIXED_deg" in ser:
        ax1.plot(t[m], np.array(ser["phi_SDA_FIXED_deg"])[m], color=C_FIXED,
                 lw=1.3, ls=(0, (3, 2)), label="Q13.18 (SDA-fx)", zorder=3)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("roll (deg)")
    ax1.set_title("(a) tracking on T3 (chirp), %g-%g s" % window, fontsize=8.5)
    ax1.set_xlim(*window)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=3,
               frameon=False, fontsize=6.8, handlelength=1.9,
               columnspacing=1.0, handletextpad=0.4)

    # ---- (b) heatmap da penalidade de custo, TODAS as seis trajetorias -------
    keys = [k for k, _ in FX_CTRLS]
    labs = [l for _, l in FX_CTRLS]
    M = np.full((len(keys), len(TRAJS)), np.nan)
    for i, k in enumerate(keys):
        for j, t_ in enumerate(TRAJS):
            ref, cur = J[t_].get("SDA_float64"), J[t_].get(k)
            if ref and cur:
                M[i, j] = 100.0 * (cur / ref - 1.0)

    faithful = [j for j, t_ in enumerate(TRAJS) if t_ != "T6_taxa_alta"]
    worst_dbl = np.nanmax(np.abs(M[:-1, faithful]))   # doubling, T1-T5
    worst_all = np.nanmax(np.abs(M[:, faithful]))     # + value iteration, T1-T5

    # SymLog: T1-T5 vivem em +-0.44%, T6 chega a +8.6%. Numa escala linear T6
    # satura tudo e T1-T5 viram uma mancha uniforme; em symlog os dois regimes
    # coexistem. A cor e' apoio ao padrao — o numero exato esta em cada celula.
    norm = SymLogNorm(linthresh=0.5, vmin=-9, vmax=9, base=10)
    im = ax2.imshow(M, cmap=_DIVERGING, norm=norm, aspect="auto")
    ax2.set_xticks(range(len(TRAJS)))
    ax2.set_xticklabels([TRAJ_LBL[t_].split()[0] for t_ in TRAJS], fontsize=7.5)
    ax2.set_yticks(range(len(labs)))
    ax2.set_yticklabels(labs, fontsize=7.2)
    ax2.set_title(r"(b) $\Delta J/J_\mathrm{ref}$ (%) vs. float64", fontsize=8.5)
    ax2.set_xticks(np.arange(-0.5, len(TRAJS), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(labs), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=1.1)
    ax2.grid(which="major", visible=False)
    ax2.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            v = M[i, j]
            # tinta escura sobre celula clara, branca sobre celula saturada
            ink = "white" if abs(v) > 2.0 else "0.12"
            if abs(v) >= 1.0:
                s = "%+.1f" % v
            elif abs(v) < 0.005:
                s = "0.00"          # evita o "-0.00" que o %+.2f produzia
            else:
                s = "%+.2f" % v
            ax2.text(j, i, s, ha="center", va="center", fontsize=6.3, color=ink)
    # T6 e' qualitativamente diferente (modelo discreto menos fiel): marcado
    # com uma moldura, em vez de simplesmente omitido como na versao anterior.
    j6 = TRAJS.index("T6_taxa_alta")
    ax2.add_patch(plt.Rectangle((j6 - 0.5, -0.5), 1, len(labs), fill=False,
                                 edgecolor="0.25", lw=1.3, ls=(0, (3, 2)), zorder=5))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = os.path.join(outdir, "fig5_closed_loop_v8.pdf")
    fig.savefig(p, metadata=_metadados("fig5_closed_loop_v8.pdf", [MALHA, SERIE_CSV]))
    plt.close(fig)
    print("  ok", p)
    print("     |dJ/J| maximo T1-T5: doubling %.3f%%, incluindo value iter. %.3f%%"
          % (worst_dbl, worst_all))
    return worst_dbl, worst_all


# --------------------------------------------------------------------------
# Fig. 6 — ciclo de voo completo
# --------------------------------------------------------------------------
def fig6_flight(outdir, proc, stages, hist_tuple=None, meta=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.24),
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    if hist_tuple and hist_tuple[0] is not None:
        p_bins, cdf_y = hist_tuple
        ax1.step(p_bins, cdf_y, where="post", color=C_FLOAT, lw=1.6)
    else:
        p = np.sort(np.array(proc) / 1000.0)
        cdf = 100.0 * np.arange(1, len(p) + 1) / len(p)
        ax1.step(p, cdf, where="post", color=C_FLOAT, lw=1.6)
    ax1.axvline(6.0, color="0.25", ls=":", lw=1.2)
    ax1.set_xlabel("cycle processing time (ms)")
    ax1.set_ylabel("cycles below abscissa (%)")
    ax1.set_title("(a) complete control cycle", fontsize=8.5)
    ax1.set_xlim(4.60, 6.15)
    ax1.set_ylim(0, 119)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.text(6.05, 45, "6.0 ms\n(167 Hz)\ncontrol period", fontsize=7, color="0.25",
             ha="left", va="center")

    order = ["DARE solve", "IMU read", "WiFi/UDP", "Euler", "Madgwick",
             "Mixer", "Motor write", "LEDs", "Battery", "SDC matrix",
             "Control law"]
    order = [o for o in order if stages.get(o)]
    vals = [st.median(stages[o]) for o in order]
    keep = [(o, v) for o, v in zip(order, vals) if v >= 1]
    o2 = [k[0] for k in keep][::-1]
    v2 = [k[1] for k in keep][::-1]
    cols = [C_FIXED if o == "DARE solve" else C_GREY for o in o2]
    yy = np.arange(len(o2))
    ax2.barh(yy, v2, color=cols, height=0.68)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(o2, fontsize=7.2)
    ax2.set_xlabel(r"median stage time ($\mu$s)")
    ax2.set_title("(b) where the cycle time goes", fontsize=8.5)
    ax2.set_xlim(0, max(v2) * 1.34)
    for y, v in zip(yy, v2):
        ax2.text(v + max(v2) * 0.02, y, "%d" % v, va="center", fontsize=7)
    fig.tight_layout()
    pth = os.path.join(outdir, "fig6_flight_cycle.pdf")
    fontes = (meta or {}).get("caminhos") or [FLIGHT]
    fig.savefig(pth, metadata=_metadados("fig6_flight_cycle.pdf", fontes))
    plt.close(fig)
    print("  ok", pth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=r"G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS"
                                        r"\DINAME_2027\artigo_diname\Figures")
    ap.add_argument("--with-fig2", action="store_true",
                    help="regenera a Fig. 2 (barras de tempo), removida do v5")
    ap.add_argument("--legacy-fig4", action="store_true",
                    help="pula a fig4_tolerance_v8 (3 paineis, painel (c) novo) e gera só a Fig. 4 do v7")
    ap.add_argument("--no-fig4-v7", action="store_true",
                    help="pula a Fig. 4 do v7 (2 paineis) — só a v8")
    ap.add_argument("--flight-dir",
                    help="pasta com voo_run*.txt; a Fig. 6 passa a agregar todas as "
                         "execucoes em vez de usar apenas outputs/serial_flightloop_E.txt")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("carregando dados...")
    tall, tt, it = load_battery()
    tq = load_tolqr()
    agg = load_boundary()
    voos = None
    if args.flight_dir:
        voos = sorted(glob.glob(os.path.join(args.flight_dir, "voo_run*.txt")),
                      key=lambda p: int(re.search(r"voo_run(\d+)", p).group(1)))
        if not voos:
            raise SystemExit("--flight-dir sem voo_run*.txt: " + args.flight_dir)
    proc, stages, hist_tuple, flight_meta = load_flight(voos)
    cover = load_coverage()
    ts = load_tolerance_sweep()
    print("gerando figuras em", args.outdir)
    fig1_envelope(args.outdir, cover)
    # Fig. 2 (barras de tempo) removida no v5: repetia as duas primeiras colunas
    # da Tabela 1 sem acrescentar informacao, e o artigo esta no teto de paginas.
    if args.with_fig2:
        fig2_timing(args.outdir, tall)
    fig3_predictability(args.outdir, tt, it)
    if not args.no_fig4_v7:
        lo, hi = fig4_tolerance(args.outdir, tq, cover)
        print("piso de quantizacao medido (Fig.4 v7): [%.2e, %.2e]" % (lo, hi))
    if not args.legacy_fig4:
        lo8, hi8 = fig4_tolerance_v8(args.outdir, ts, cover)
        print("piso de quantizacao medido (Fig.4 v8): [%.2e, %.2e]" % (lo8, hi8))
    fig5_closed_loop(args.outdir)
    fig5_safety(args.outdir, agg)
    fig6_flight(args.outdir, proc, stages, hist_tuple, flight_meta)
    _grava_registro_proveniencia()


if __name__ == "__main__":
    main()
