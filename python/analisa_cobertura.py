"""
Cobertura de condicionamento numérico das 4 trajetórias (Exp. 4 / Parte 4 do
plano de correção do artigo, docs/auditoria_solvers_riccati.md Seção 15).

Pergunta do usuário: 4 trajetórias garantem a melhor avaliação possível, ou
vale acrescentar mais? Um protótipo em 231 pontos amostrados sugeriu que as
4 trajetórias produzem condicionamento numérico quase idêntico apesar de
perfis de atitude muito diferentes — este script refaz essa medição nos
46152 pontos completos (4 trajetórias x 11538 pontos, mesma discretização
usada em test/benchmark_solvers.cpp / tolerance_sweep.cpp / gamma_sweep.cpp),
sem decimação, usando scipy.linalg.solve_discrete_are como referência
float64 (mesma referência-ouro de bench_trajetorias.py).

Três métricas por ponto de operação:
  - rho:        raio espectral de malha fechada, max(|eig(Ad - Bd@K)|)
  - cond_IGP:   numero de condicao de (I + G@P), G = Bd @ Rd^-1 @ Bd^T
                (a mesma matriz cuja inversao domina o SDA-SS_Fixed via Phi)
  - normP_F:    norma de Frobenius de P (solucao da DARE)

Uso:
    python analisa_cobertura.py [--saida outputs/cobertura_full.csv]

Sem --saida, grava em outputs/cobertura_full.csv por padrão. Imprime uma
tabela resumo (min/max/media) por trajetoria e um veredito honesto: se as
4 trajetorias cobrem faixas de condicionamento essencialmente iguais, isso
e reportado como tal (nao inventa diversidade que a medicao nao mostra).
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, os.path.dirname(__file__))
import trajetorias as trj
from bench_trajetorias import build_Ad_Bd_Qd_Rd, k_from_P


def analisa_ponto(phi, theta, p, q, r):
    Ad, Bd, Qd, Rd = build_Ad_Bd_Qd_Rd(phi, theta, p, q, r)
    try:
        P = solve_discrete_are(Ad, Bd, Qd, Rd)
    except Exception:
        return None
    K = k_from_P(Ad, Bd, Qd, Rd, P)
    Acl = Ad - Bd @ K
    eig = np.linalg.eigvals(Acl)
    rho = float(np.max(np.abs(eig)))

    n = Ad.shape[0]
    G = Bd @ np.linalg.solve(Rd, Bd.T)
    IGP = np.eye(n) + G @ P
    cond_igp = float(np.linalg.cond(IGP))

    normP_F = float(np.linalg.norm(P, ord="fro"))
    return rho, cond_igp, normP_F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="outputs/cobertura_full_v5_6traj.csv")
    ap.add_argument("--stride", type=int, default=1,
                     help="1 = todos os 46152 pontos (default); >1 para amostragem mais rapida")
    ap.add_argument("--fonte", choices=["dispositivo", "espelho"], default="dispositivo",
                     help="'dispositivo' (default) usa as linhas PT da captura da bateria — o "
                          "estado que o firmware de fato usou; 'espelho' regenera em Python.")
    args = ap.parse_args()

    # A referencia de dupla precisao so' e' comparavel ao dispositivo se for
    # avaliada NO MESMO ponto de operacao. Regenerar a trajetoria em Python
    # introduz descasamento de modelo — em T4 o alvo do degrau vem do sinal de
    # um seno e a grade de 6 ms cai sobre os zeros, onde float32 e float64
    # escolhem alvos opostos. Ver python/pontos_dispositivo.py.
    if args.fonte == "dispositivo":
        import pontos_dispositivo as pd_
        dados = pd_.carregar()
        print("# fonte dos pontos: linhas PT da captura da bateria (estado medido)")
    else:
        dados = trj.gerar_todas()
        print("# fonte dos pontos: espelho de host (python/trajetorias.py)")
    os.makedirs(os.path.dirname(args.saida), exist_ok=True)

    resumo = {}
    t0 = time.time()
    n_total = 0
    n_falha = 0
    with open(args.saida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj", "k", "phi", "theta", "p", "q", "r", "rho", "cond_IGP", "normP_F"])
        for traj_nome, d in dados.items():
            idx = range(0, len(d["t"]), args.stride)
            rhos, conds, normPs = [], [], []
            for k in idx:
                phi, theta, p, q, r = d["phi"][k], d["theta"][k], d["p"][k], d["q"][k], d["r"][k]
                res = analisa_ponto(phi, theta, p, q, r)
                n_total += 1
                if res is None:
                    n_falha += 1
                    continue
                rho, cond_igp, normP_F = res
                rhos.append(rho)
                conds.append(cond_igp)
                normPs.append(normP_F)
                w.writerow([traj_nome, k, phi, theta, p, q, r, rho, cond_igp, normP_F])
            resumo[traj_nome] = (np.array(rhos), np.array(conds), np.array(normPs))
            print(f"# {traj_nome}: {len(rhos)} pontos processados "
                  f"({time.time() - t0:.1f}s decorridos)", flush=True)

    print(f"\n# CSV bruto: {args.saida} ({n_total - n_falha}/{n_total} pontos, "
          f"{n_falha} falhas de DARE)")
    print(f"# tempo total: {time.time() - t0:.1f}s\n")

    print(f"{'trajetoria':<14} {'rho_min':>9} {'rho_max':>9} {'rho_media':>10} | "
          f"{'cond_min':>9} {'cond_max':>9} {'cond_media':>11} | "
          f"{'||P||F_min':>11} {'||P||F_max':>11} {'||P||F_media':>13}")
    print("-" * 118)
    all_rho, all_cond, all_normP = [], [], []
    for traj_nome, (rhos, conds, normPs) in resumo.items():
        all_rho.append(rhos); all_cond.append(conds); all_normP.append(normPs)
        print(f"{traj_nome:<14} {rhos.min():>9.4f} {rhos.max():>9.4f} {rhos.mean():>10.4f} | "
              f"{conds.min():>9.3f} {conds.max():>9.3f} {conds.mean():>11.3f} | "
              f"{normPs.min():>11.4f} {normPs.max():>11.4f} {normPs.mean():>13.4f}")

    all_rho = np.concatenate(all_rho)
    all_cond = np.concatenate(all_cond)
    all_normP = np.concatenate(all_normP)
    print("-" * 118)
    print(f"{'TODAS':<14} {all_rho.min():>9.4f} {all_rho.max():>9.4f} {all_rho.mean():>10.4f} | "
          f"{all_cond.min():>9.3f} {all_cond.max():>9.3f} {all_cond.mean():>11.3f} | "
          f"{all_normP.min():>11.4f} {all_normP.max():>11.4f} {all_normP.mean():>13.4f}")

    # Veredito honesto: a faixa coberta por CADA trajetoria isolada, comparada
    # a faixa coberta por TODAS juntas. Se a razao (faixa_individual/faixa_total)
    # for proxima de 1 para todas as trajetorias, nenhuma delas amplia a
    # cobertura de condicionamento sobre as outras -- sao redundantes nesse
    # eixo, mesmo que diversifiquem o dominio temporal/atitude.
    print("\n# Veredito de cobertura (faixa individual / faixa total, rho):")
    faixa_total = all_rho.max() - all_rho.min()
    for traj_nome, (rhos, _, _) in resumo.items():
        faixa_ind = rhos.max() - rhos.min()
        frac = faixa_ind / faixa_total if faixa_total > 0 else float("nan")
        print(f"#   {traj_nome:<14} faixa rho = [{rhos.min():.4f}, {rhos.max():.4f}] "
              f"({frac*100:.1f}% da faixa total)")


if __name__ == "__main__":
    main()
