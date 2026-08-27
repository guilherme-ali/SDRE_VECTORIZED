import csv
import math
import os
import re
import statistics as st
from collections import defaultdict
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")

def audit_main_battery():
    print("=== 1. AUDITORIA: BATERIA PRINCIPAL (S2) ===")
    bat_file = os.path.join(OUT, "serial_capture_bateria_v5_6traj.txt")
    if not os.path.exists(bat_file):
        print("Arquivo nao encontrado:", bat_file)
        return
    
    t = defaultdict(list)
    it = defaultdict(list)
    res = defaultdict(list)
    outcomes = defaultdict(lambda: defaultdict(int))
    
    with open(bat_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 8:
                continue
            traj, k, m, us, its, r, outc = p[1], int(p[2]), p[3], int(p[4]), int(p[5]), float(p[6]), int(p[7])
            t[m].append(us)
            it[m].append(its)
            res[m].append(r)
            outcomes[m][outc] += 1
            
    methods = ["SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA", "ITERATIVE",
               "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED", "ITERATIVE_FIXED"]
    
    print(f"{'Metodo':<20} | {'Count':<6} | {'Med (ms)':<8} | {'p99.9 (ms)':<10} | {'Mean iter':<9} | {'Std iter':<8} | {'Med Res':<10} | {'Breakdown':<9}")
    print("-" * 95)
    for m in methods:
        cnt = len(t[m])
        if cnt == 0:
            continue
        times_ms = [x / 1000.0 for x in t[m]]
        med_t = st.median(times_ms)
        p999_t = np.percentile(times_ms, 99.9)
        mean_it = st.mean(it[m])
        std_it = st.stdev(it[m]) if cnt > 1 else 0.0
        med_res = st.median(res[m])
        brk = outcomes[m][2]
        print(f"{m:<20} | {cnt:<6} | {med_t:<8.2f} | {p999_t:<10.2f} | {mean_it:<9.2f} | {std_it:<8.4f} | {med_res:<10.2e} | {brk:<9}")

def audit_s3_battery():
    print("\n=== 2. AUDITORIA: BATERIA S3 ===")
    s3_file = os.path.join(OUT, "s3", "serial_capture_bateria_s3.txt")
    if not os.path.exists(s3_file):
        print("Arquivo nao encontrado:", s3_file)
        return
    
    t = defaultdict(list)
    it = defaultdict(list)
    with open(s3_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("RUN,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 8:
                continue
            m, us, its = p[3], int(p[4]), int(p[5])
            t[m].append(us)
            it[m].append(its)
            
    methods = ["SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA", "ITERATIVE",
               "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED", "ITERATIVE_FIXED"]
    
    print(f"{'Metodo':<20} | {'Count':<6} | {'Med (ms)':<8} | {'p99.9 (ms)':<10} | {'Mean iter':<9}")
    print("-" * 65)
    for m in methods:
        cnt = len(t[m])
        if cnt == 0:
            continue
        times_ms = [x / 1000.0 for x in t[m]]
        med_t = st.median(times_ms)
        p999_t = np.percentile(times_ms, 99.9)
        mean_it = st.mean(it[m])
        print(f"{m:<20} | {cnt:<6} | {med_t:<8.2f} | {p999_t:<10.2f} | {mean_it:<9.2f}")

def audit_repeatability():
    print("\n=== 3. AUDITORIA: REPETIBILIDADE / JITTER ===")
    rep_file = os.path.join(OUT, "serial_repeatability_D.txt")
    if not os.path.exists(rep_file):
        print("Arquivo nao encontrado:", rep_file)
        return
    
    cv_by_method = defaultdict(list)
    with open(rep_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("SUMMARY,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 11:
                continue
            m = p[3]
            try:
                cv = float(p[8])
                cv_by_method[m].append(cv)
            except ValueError:
                pass
                
    for m, cvs in sorted(cv_by_method.items()):
        print(f"{m:<20}: N_pts={len(cvs)} | CV medio={st.mean(cvs):.4f}% | min={min(cvs):.4f}% | max={max(cvs):.4f}%")

def audit_closed_loop():
    print("\n=== 4. AUDITORIA: MALHA FECHADA & GANHO CONGELADO ===")
    mf_file = os.path.join(OUT, "malha_fechada_v6_6traj.csv")
    gc_file = os.path.join(OUT, "ganho_congelado_6traj.csv")
    if os.path.exists(mf_file):
        df_mf = pd.read_csv(mf_file)
        print("Malha fechada (resumo por traj e controller):")
        print(df_mf[["traj", "controller", "rms_total_deg", "J_total"]].to_string())
        
        print("\nDesvio de custo J relativo ao SDA_float64 (%):")
        for traj in df_mf["traj"].unique():
            sub = df_mf[df_mf["traj"] == traj]
            j_ref = sub[sub["controller"] == "SDA_float64"]["J_total"].values[0]
            for _, row in sub.iterrows():
                if row["controller"] != "SDA_float64":
                    diff_pct = 100.0 * (row["J_total"] - j_ref) / j_ref
                    print(f"  {traj:<15} | {row['controller']:<18} | J_dev={diff_pct:+.4f}% | RMS={row['rms_total_deg']:.2f} deg")
                    
    if os.path.exists(gc_file):
        df_gc = pd.read_csv(gc_file)
        print("\nGanho congelado:")
        print(df_gc.to_string())

def audit_conditioning():
    print("\n=== 5. AUDITORIA: CONDICIONAMENTO E ENVELOPE ===")
    cober_file = os.path.join(OUT, "cobertura_full_v5_6traj.csv")
    if os.path.exists(cober_file):
        df_c = pd.read_csv(cober_file)
        print(f"Total pontos: {len(df_c)}")
        print(f"cond(I+GP): min={df_c['cond_IGP'].min():.3f}, max={df_c['cond_IGP'].max():.3f}, median={df_c['cond_IGP'].median():.3f}")
        print(f"||P||_F:    min={df_c['normP_F'].min():.4f}, max={df_c['normP_F'].max():.4f}, median={df_c['normP_F'].median():.4f}")

def audit_flight_loop():
    print("\n=== 6. AUDITORIA: FLIGHT LOOP (EXP E) ===")
    fl_file = os.path.join(OUT, "serial_flightloop_E.txt")
    if os.path.exists(fl_file):
        txt = open(fl_file, encoding="utf-8", errors="replace").read()
        blocks = txt.split("STATUS DO SISTEMA")
        print(f"Numero de blocos de status: {len(blocks)-1}")
        
        proc = []
        for b in blocks[1:]:
            m = re.search(r"Tempo_Processamento:\s*(\d+)", b)
            if m:
                proc.append(int(m.group(1)))
        if proc:
            proc_ms = [x / 1000.0 for x in proc]
            print(f"Amostras Tempo_Processamento (1x/s): N={len(proc_ms)}")
            print(f"  Mediana: {st.median(proc_ms):.3f} ms")
            print(f"  Media:   {st.mean(proc_ms):.3f} ms")
            print(f"  Min:     {min(proc_ms):.3f} ms")
            print(f"  Max:     {max(proc_ms):.3f} ms")
            print(f"  p99:     {np.percentile(proc_ms, 99):.3f} ms")
            print(f"  p99.9:   {np.percentile(proc_ms, 99.9):.3f} ms")
            
        last_block = blocks[-1]
        for line in last_block.split("\n"):
            clean_line = line.encode('ascii', errors='replace').decode('ascii')
            if any(k in clean_line for k in ["Overrun_Count", "Tempo_Processamento", "Processamento_", "Tempo_Loop", "Tempo_Medio", "Tempo dos Prints"]):
                print("  [Ultimo bloco]", clean_line.strip())

def audit_gamma():
    print("\n=== 7. AUDITORIA: GAMMA SWEEP (SDA-SS) ===")
    g_file = os.path.join(OUT, "serial_gamma_sweep.txt")
    if not os.path.exists(g_file):
        print("Arquivo nao encontrado:", g_file)
        return
    
    # parse SUMMARY lines
    with open(g_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("SUMMARY"):
                print("  [gamma summary]", line.strip())

def audit_boundary():
    print("\n=== 8. AUDITORIA: BOUNDARY FINE (R_scale) ===")
    b_file = os.path.join(OUT, "serial_boundary_fine_B.txt")
    if not os.path.exists(b_file):
        print("Arquivo nao encontrado:", b_file)
        return
    
    agg = defaultdict(lambda: [0, 0])
    with open(b_file, encoding="utf-8", errors="replace") as f:
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
            
    print("Transicoes da fronteira superior (R_scale > 100):")
    r_scales = sorted(set(k[0] for k in agg.keys()))
    for r in r_scales:
        if r > 100:
            brk_tot = sum(agg[(r, m)][0] for m in ["SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"])
            cnt_tot = sum(agg[(r, m)][1] for m in ["SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED"])
            pct = 100.0 * brk_tot / cnt_tot if cnt_tot > 0 else 0
            print(f"  R_scale = {r:<6.1f} | brk = {brk_tot}/{cnt_tot} ({pct:.2f}%)")

if __name__ == "__main__":
    audit_main_battery()
    audit_s3_battery()
    audit_repeatability()
    audit_closed_loop()
    audit_conditioning()
    audit_flight_loop()
    audit_gamma()
    audit_boundary()
