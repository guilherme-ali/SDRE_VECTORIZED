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
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
ZENODO = os.path.join(REPO, "zenodo_diname2027")

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


def main():
    os.makedirs(ZENODO, exist_ok=True)
    print("Exportando para %s" % ZENODO)

    export_tolerance_sweep(SOURCES["tolerance_sweep"],
                            os.path.join(ZENODO, "tolerance_sweep_runs.csv"))
    export_battery(SOURCES["battery_s2"], os.path.join(ZENODO, "battery_s2_runs.csv"))
    export_battery(SOURCES["battery_s3"], os.path.join(ZENODO, "battery_s3_runs.csv"))

    table2_dst = os.path.join(ZENODO, "table2.csv")
    if os.path.isfile(SOURCES["table2_csv"]):
        with open(SOURCES["table2_csv"], encoding="utf-8") as fi, \
             open(table2_dst, "w", encoding="utf-8") as fo:
            fo.write(fi.read())
        print("  copiado table2.csv")
    else:
        print("  pulado (rodar python/generate_table2.py antes): %s" % SOURCES["table2_csv"])

    export_memory_csv(SOURCES["memory_json"], os.path.join(ZENODO, "memory_map.csv"))

    # ciclo de voo: N execucoes do mesmo binario (unico experimento nao deterministico)
    export_flight(sorted(glob.glob(VOO_GLOB),
                         key=lambda p: int(re.search(r"voo_run(\d+)", p).group(1))),
                  os.path.join(ZENODO, "flight_cycle_runs.csv"),
                  os.path.join(ZENODO, "flight_cycle_histogram.csv"))

    # malha fechada e cobertura: ja sao CSV tabulares na origem, copiados como estao
    for key, name in (("closed_loop", "closed_loop_cost.csv"),
                      ("gain_schedule", "gain_update_schedule.csv"),
                      ("coverage", "operating_point_coverage.csv"),
                      ("reference_residual", "reference_residual.csv"),
                      ("discretisation_fidelity", "discretisation_fidelity.csv")):
        src = SOURCES[key]
        if os.path.isfile(src):
            with open(src, encoding="utf-8") as fi, \
                 open(os.path.join(ZENODO, name), "w", encoding="utf-8") as fo:
                fo.write(fi.read())
            print("  copiado %s" % name)
        else:
            print("  pulado (nao encontrado): %s" % src)

    # as seis series temporais de malha fechada, descobertas por glob
    for src in sorted(glob.glob(SERIES_GLOB)):
        name = "closed_loop_series_" + os.path.basename(src).split("serie_")[1]
        with open(src, encoding="utf-8") as fi, \
             open(os.path.join(ZENODO, name), "w", encoding="utf-8") as fo:
            fo.write(fi.read())
        print("  copiado %s" % name)

    commit = git_commit()
    manifest_path = os.path.join(ZENODO, "MANIFEST.md")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# Manifesto do pacote de dados — DINAME 2027 (v8)\n\n")
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
        f.write("\nCSVs derivados nesta pasta:\n\n")
        for name in sorted(os.listdir(ZENODO)):
            p = os.path.join(ZENODO, name)
            if name.endswith(".csv") and os.path.isfile(p):
                f.write("- `%s` (%d bytes, sha256 `%s`)\n" % (name, os.path.getsize(p), sha256_of(p)))
    print("MANIFEST.md escrito em %s" % manifest_path)


if __name__ == "__main__":
    main()
