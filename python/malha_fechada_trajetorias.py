"""
Malha fechada: simula o laço SDRE completo (realimentação de erro,
ganho recalculado a cada passo por linearização no estado ATUAL — extended
linearization / SDRE clássico) seguindo as 4 trajetórias de
python/trajetorias.py, comparando o controlador de referência
(SDA em float64, `verifica_solvers.sda_reference`) contra os 5 solvers
`_FIXED` emulados em Q13.18 (`fixedpoint_q.py`).

Resolve a crítica do revisor 2 do CBA 2026 ("a abordagem estatística de
malha aberta não captura a verdadeira trajetória sequencial contínua de um
drone em voo... o artigo seria significativamente mais forte se fosse
incluída uma simulação em malha fechada") — mas com um escopo deliberadamente
mais estreito que experiments/benchmark_solvers.cpp: aqui o objetivo não é medir
tempo de execução (isso já está coberto pela bateria on-device), e sim se a
QUANTIZAÇÃO de cada variante _FIXED degrada mensuravelmente o rastreamento
de trajetória e o custo acumulado, quando comparada ao float64 de
referência.

Simplificação assumida e documentada: a dinâmica "verdadeira" simulada é o
próprio modelo SDC discretizado (Ad,Bd por ponto, os mesmos usados para
montar a DARE) — não uma simulação não linear separada. O controle é
realimentação de erro u = -K(x - x_ref), com K recalculado a cada passo por
linearização no estado atual (não no estado de referência) — é a forma
usual de SDRE "extended linearization" tracking. Isso NÃO é uma prova de
estabilidade nem uma simulação de voo de alta fidelidade (essa já existe,
para um cenário isolado, em atitude_sim.py); é o experimento mínimo que
falta para responder à crítica do R2 com uma trajetória sequencial real sob
realimentação, mantendo o escopo do artigo em "solvers da DARE", não em
"simulador de voo".

Uso:
    python malha_fechada_trajetorias.py [--saida outputs/malha_fechada.csv]
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import trajetorias as trj
from bench_trajetorias import (
    build_Ad_Bd_Qd_Rd, k_from_P, FIXED_SOLVERS, IterativeFixedGain,
    MAX_TAU_ROLL, MAX_TAU_PITCH, MAX_TAU_YAW,
)
from verifica_solvers import sda_reference

TAU_LIMITS = np.array([MAX_TAU_ROLL, MAX_TAU_PITCH, MAX_TAU_YAW])


def angle_diff(a, b):
    """a - b, tratando o wrap de +-pi (psi de T4 cruza a descontinuidade)."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def simulate_closed_loop(traj_dados, controller_name, gain_fn):
    """gain_fn(Ad,Bd,Qd,Rd) -> (P, ok). Retorna dict com séries e métricas
    agregadas. ok=False no passo k reaproveita o K do passo anterior —
    mesma convenção do firmware real (ver FixedPointQ.h)."""
    N = len(traj_dados["t"])
    x = np.array([traj_dados["phi"][0], traj_dados["theta"][0], traj_dados["psi"][0],
                  traj_dados["p"][0], traj_dados["q"][0], traj_dados["r"][0]])

    err_hist = np.zeros((N, 3))  # phi,theta,psi
    att_hist = np.zeros((N, 3))  # atitude REALIZADA (phi,theta,psi) — Fig. 5(a)
    ref_hist = np.zeros((N, 3))  # atitude COMANDADA, para o mesmo grafico
    J_total = 0.0
    K_prev = None
    n_fallback = 0
    n_total = 0

    for k in range(N):
        x_ref = np.array([traj_dados["phi"][k], traj_dados["theta"][k], traj_dados["psi"][k],
                           traj_dados["p"][k], traj_dados["q"][k], traj_dados["r"][k]])
        e = x - x_ref
        e[2] = angle_diff(x[2], x_ref[2])
        err_hist[k] = e[:3]
        att_hist[k] = x[:3]
        ref_hist[k] = x_ref[:3]

        # Linearizacao no estado ATUAL (extended linearization / SDRE classico)
        phi_c = max(min(x[0], math.radians(75)), math.radians(-75))
        theta_c = max(min(x[1], math.radians(75)), math.radians(-75))
        Ad, Bd, Qd, Rd = build_Ad_Bd_Qd_Rd(phi_c, theta_c, x[3], x[4], x[5])

        n_total += 1
        P, ok_or_st = gain_fn(Ad, Bd, Qd, Rd)
        ok = P is not None
        if ok:
            K = k_from_P(Ad, Bd, Qd, Rd, P)
            K_prev = K
        else:
            n_fallback += 1
            if K_prev is None:
                K = np.zeros((3, 6))
            else:
                K = K_prev

        u = -K @ e
        u = np.clip(u, -TAU_LIMITS, TAU_LIMITS)

        J_total += float(e @ Qd @ e + u @ Rd @ u)

        x = Ad @ x + Bd @ u

    rms_deg = np.sqrt(np.mean(np.rad2deg(err_hist) ** 2, axis=0))
    return {
        "controller": controller_name,
        "rms_phi_deg": rms_deg[0], "rms_theta_deg": rms_deg[1], "rms_psi_deg": rms_deg[2],
        "rms_total_deg": float(np.sqrt(np.mean(np.rad2deg(err_hist) ** 2))),
        "J_total": J_total,
        "fallback_rate": n_fallback / max(n_total, 1),
        "n_fallback": n_fallback, "n_total": n_total,
        # séries (não vão para o CSV agregado; usadas só pelo dump de --series-traj)
        "_att_hist": att_hist, "_ref_hist": ref_hist,
    }


def _sda_float_gain(Ad, Bd, Qd, Rd):
    P, _ = sda_reference(Ad, Bd, Qd, Rd)
    return P, True


CONTROLLERS = {"SDA_float64": _sda_float_gain}
CONTROLLERS.update(FIXED_SOLVERS)

# ITERATIVE_FIXED fica FORA de CONTROLLERS de propósito: precisa de uma
# instância nova de IterativeFixedGain por (trajetória, controlador) para que
# o warm-start (estado interno da instância) não vaze de uma simulação para a
# próxima — CONTROLLERS.items() reutilizaria a MESMA função/objeto em todo o
# laço de main(), o que misturaria warm-start entre trajetórias diferentes.
# Ver bench_trajetorias.IterativeFixedGain e o item 0.5 do plano da campanha
# estendida (docs/auditoria_solvers_riccati.md, Seção 15).
STATEFUL_CONTROLLER_FACTORIES = {"ITERATIVE_FIXED": lambda: IterativeFixedGain()}


def main(saida, series_trajs=None, series_dir=None, series_ctrls=("SDA_float64", "SDA_FIXED")):
    """series_trajs/series_dir: além do CSV agregado, grava a série temporal de
    atitude comandada vs. realizada de CADA trajetória listada (um CSV por
    trajetória), para os controladores em series_ctrls. É a fonte do painel (a)
    da figura de malha fechada, que antes não tinha script que a reproduzisse.
    Grava todas por padrão: a simulação leva ~17 min e escolher a trajetória a
    exibir depois não deve exigir rodá-la de novo."""
    dados = trj.gerar_todas()
    rows = []
    series = defaultdict(dict)   # traj -> ctrl -> (att, ref, t)
    t_start = time.time()
    for traj_nome, d in dados.items():
        for ctrl_nome, fn in CONTROLLERS.items():
            t0 = time.time()
            res = simulate_closed_loop(d, ctrl_nome, fn)
            res["traj"] = traj_nome
            res["tempo_sim_s"] = time.time() - t0
            rows.append(res)
            if series_trajs and traj_nome in series_trajs and ctrl_nome in series_ctrls:
                series[traj_nome][ctrl_nome] = (res["_att_hist"], res["_ref_hist"], d["t"])
            print("%-14s %-16s rms_total=%.3f deg  J=%.4e  fallback=%d/%d  (%.1fs)" % (
                traj_nome, ctrl_nome, res["rms_total_deg"], res["J_total"],
                res["n_fallback"], res["n_total"], res["tempo_sim_s"]))
        for ctrl_nome, factory in STATEFUL_CONTROLLER_FACTORIES.items():
            fn = factory()  # instancia nova -- warm-start comeca zerado, mesma
                             # convencao de "controlador armado do zero" usada
                             # para os demais (nenhum deles carrega estado
                             # entre trajetorias nesta simulacao)
            t0 = time.time()
            res = simulate_closed_loop(d, ctrl_nome, fn)
            res["traj"] = traj_nome
            res["tempo_sim_s"] = time.time() - t0
            rows.append(res)
            print("%-14s %-16s rms_total=%.3f deg  J=%.4e  fallback=%d/%d  (%.1fs)" % (
                traj_nome, ctrl_nome, res["rms_total_deg"], res["J_total"],
                res["n_fallback"], res["n_total"], res["tempo_sim_s"]))

    os.makedirs(os.path.dirname(saida), exist_ok=True)
    fieldnames = ["traj", "controller", "rms_phi_deg", "rms_theta_deg", "rms_psi_deg",
                  "rms_total_deg", "J_total", "n_fallback", "n_total", "fallback_rate", "tempo_sim_s"]
    with open(saida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("\ngravado: %s (%d linhas, %.1f min totais)" % (saida, len(rows), (time.time() - t_start) / 60.0))

    if series_dir and series:
        os.makedirs(series_dir, exist_ok=True)
        for traj_nome, per_ctrl in series.items():
            any_ctrl = next(iter(per_ctrl))
            t_vec = per_ctrl[any_ctrl][2]
            cols = ["t", "phi_ref_deg", "theta_ref_deg", "psi_ref_deg"]
            for c in per_ctrl:
                cols += ["phi_%s_deg" % c, "theta_%s_deg" % c, "psi_%s_deg" % c]
            out = os.path.join(series_dir, "malha_fechada_serie_%s.csv" % traj_nome)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                ref = per_ctrl[any_ctrl][1]
                for k in range(len(t_vec)):
                    row = [t_vec[k]] + [math.degrees(v) for v in ref[k]]
                    for c in per_ctrl:
                        row += [math.degrees(v) for v in per_ctrl[c][0][k]]
                    w.writerow(["%.6g" % v for v in row])
            print("gravado: %s (%s)" % (out, ", ".join(per_ctrl)))

    print("\n=== Degradacao relativa (fixed vs. SDA_float64), por trajetoria ===")
    by_traj = {}
    for row in rows:
        by_traj.setdefault(row["traj"], {})[row["controller"]] = row
    for traj_nome, ctrls in by_traj.items():
        base = ctrls.get("SDA_float64")
        if base is None:
            continue
        print("\n%s (referencia float64: rms=%.3f deg, J=%.4e)" % (traj_nome, base["rms_total_deg"], base["J_total"]))
        for ctrl_nome, row in ctrls.items():
            if ctrl_nome == "SDA_float64":
                continue
            drms = row["rms_total_deg"] - base["rms_total_deg"]
            dj = (row["J_total"] / base["J_total"] - 1.0) * 100.0 if base["J_total"] > 0 else float("nan")
            print("  %-18s rms=%.3f deg (%+.3f)  J %+.1f%%  fallback=%d" % (
                ctrl_nome, row["rms_total_deg"], drms, dj, row["n_fallback"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "malha_fechada_v6_6traj.csv"))
    ap.add_argument("--series-trajs", default="all",
                     help="trajetorias cuja serie temporal e' exportada (lista por virgula, ou 'all')")
    ap.add_argument("--series-dir", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    args = ap.parse_args()
    todas = list(trj.gerar_todas().keys())
    trajs = todas if args.series_trajs == "all" else [t.strip() for t in args.series_trajs.split(",") if t.strip()]
    main(args.saida, series_trajs=set(trajs), series_dir=args.series_dir)
