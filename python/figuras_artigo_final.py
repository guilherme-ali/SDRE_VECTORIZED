"""
Figuras do artigo DINAME 2027 — geradas a partir dos dados da campanha final.

Fontes de dados (todas medidas no ESP32-S2, exceto onde indicado):
  outputs/serial_capture_bateria_v5_6traj.txt   bateria principal (69228 pts x 12 metodos)
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
import math
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")

BAT = os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt")
TOLQR = os.path.join(OUT, "serial_tol_qr_sweep_A.txt")
BOUND = os.path.join(OUT, "serial_boundary_fine_B.txt")
FLIGHT = os.path.join(OUT, "serial_flightloop_E.txt")
COBER = os.path.join(OUT, "cobertura_full_v5_6traj.csv")

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

PERIOD_US = 5200.0

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


def load_flight():
    txt = open(FLIGHT, encoding="utf-8", errors="replace").read()
    blocks = txt.split("STATUS DO SISTEMA")
    proc, stages = [], defaultdict(list)
    names = [("LEDs", "LEDs"), ("Bateria", "Battery"), ("WiFi/UDP", "WiFi/UDP"),
             ("Leitura MPU", "IMU read"), ("Filtro Madgwick", "Madgwick"),
             (r"C.lc. .ngulos", "Euler"), ("Matriz Sistema", "SDC matrix"),
             (r"LQR .Ganhos.", "DARE solve"), (r"L.gica Controle", "Control law"),
             (r"C.lc. Omega.", "Mixer"), ("Set Motores", "Motor write")]
    for b in blocks[1:]:
        m = re.search(r"Tempo_Processamento:\s*(\d+)", b)
        if m:
            proc.append(int(m.group(1)))
        for pat, lbl in names:
            mm = re.search(pat + r":\s*(\d+)\s*.s", b)
            if mm:
                stages[lbl].append(int(mm.group(1)))
    return proc, stages


def load_coverage():
    d = defaultdict(lambda: defaultdict(list))
    with open(COBER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in ("rho", "cond_IGP", "normP_F"):
                d[r["traj"]][c].append(float(r[c]))
    return d


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


# --------------------------------------------------------------------------
# Fig. 1 — as seis trajetorias
# --------------------------------------------------------------------------
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
    ax2.set_ylim(4.4, 5.95)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=PALETTE[i],
                      label=TRAJ_LBL[t]) for i, t in enumerate(TRAJS)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.045), columnspacing=1.1,
               handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    p = os.path.join(outdir, "fig1_envelope.pdf")
    fig.savefig(p, dpi=300)
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
    fig.savefig(p)
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
                               label="5.2 ms control period")],
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
    fig.savefig(p)
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
    fig.savefig(p)
    plt.close(fig)
    print("  ok", p)
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
    ax1.axvline(147.6, color="0.2", ls="--", lw=1.1)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$R_\mathrm{scale}$ (nominal $=1$)")
    ax1.set_ylabel("breakdown rate (%)")
    ax1.set_ylim(-5, 112)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_title("(a) full weighting range", fontsize=8.5)
    ax1.annotate("$\\mathbf{R}_d$ input\noverflow at $147.6$",
                 xy=(147.6, 50), xytext=(1.5, 60), fontsize=6.8, color="0.15",
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
    fig.savefig(p)
    plt.close(fig)
    print("  ok", p)


# --------------------------------------------------------------------------
# Fig. 6 — ciclo de voo completo
# --------------------------------------------------------------------------
def fig6_flight(outdir, proc, stages):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.24),
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    p = np.sort(np.array(proc) / 1000.0)
    cdf = 100.0 * np.arange(1, len(p) + 1) / len(p)
    ax1.step(p, cdf, where="post", color=C_FLOAT, lw=1.6)
    ax1.axvline(5.2, color="#D55E00", ls="--", lw=1.2)
    ax1.axvline(6.0, color="0.25", ls=":", lw=1.2)
    ax1.set_xlabel("cycle processing time (ms)")
    ax1.set_ylabel("cycles below abscissa (%)")
    ax1.set_title("(a) complete control cycle", fontsize=8.5)
    ax1.set_xlim(4.95, 6.30)
    ax1.set_ylim(0, 119)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.text(5.15, 45, "5.2 ms\n(200 Hz)", fontsize=7, color="#D55E00",
             ha="right", va="center")
    ax1.text(6.05, 45, "6.0 ms\n(167 Hz)", fontsize=7, color="0.25",
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
    fig.savefig(pth)
    plt.close(fig)
    print("  ok", pth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=r"G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS"
                                        r"\DINAME_2027\artigo_diname\Figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("carregando dados...")
    tall, tt, it = load_battery()
    tq = load_tolqr()
    agg = load_boundary()
    proc, stages = load_flight()
    cover = load_coverage()
    print("gerando figuras em", args.outdir)
    fig1_envelope(args.outdir, cover)
    fig2_timing(args.outdir, tall)
    fig3_predictability(args.outdir, tt, it)
    lo, hi = fig4_tolerance(args.outdir, tq, cover)
    fig5_safety(args.outdir, agg)
    fig6_flight(args.outdir, proc, stages)
    print("piso de quantizacao medido: [%.2e, %.2e]" % (lo, hi))


if __name__ == "__main__":
    main()
