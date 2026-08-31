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
import hashlib
import json
import os
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
    "closed_loop_series": os.path.join(OUT, "malha_fechada_serie_T2.csv"),
}


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

    # malha fechada: ja sao CSV tabulares na origem, copiados como estao
    for key, name in (("closed_loop", "closed_loop_cost.csv"),
                      ("closed_loop_series", "closed_loop_series_T2.csv")):
        src = SOURCES[key]
        if os.path.isfile(src):
            with open(src, encoding="utf-8") as fi, \
                 open(os.path.join(ZENODO, name), "w", encoding="utf-8") as fo:
                fo.write(fi.read())
            print("  copiado %s" % name)
        else:
            print("  pulado (nao encontrado): %s" % src)

    commit = git_commit()
    manifest_path = os.path.join(ZENODO, "MANIFEST.md")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# Manifesto do pacote de dados — DINAME 2027 (v8)\n\n")
        f.write("Gerado por `python/exporta_dados_revisores.py` a partir do commit "
                "`%s` deste repositório.\n\n" % commit)
        f.write("Hash SHA-256 dos arquivos de origem (capturas seriais brutas em `outputs/`, "
                "não movidas nem alteradas por este script):\n\n")
        f.write("| Arquivo de origem | SHA-256 |\n|---|---|\n")
        for key, path in SOURCES.items():
            if os.path.isfile(path):
                f.write("| `%s` | `%s` |\n" % (os.path.relpath(path, REPO), sha256_of(path)))
            else:
                f.write("| `%s` | (nao encontrado nesta maquina) |\n" % os.path.relpath(path, REPO))
        f.write("\nCSVs derivados nesta pasta:\n\n")
        for name in ("tolerance_sweep_runs.csv", "battery_s2_runs.csv", "battery_s3_runs.csv",
                     "table2.csv", "memory_map.csv", "closed_loop_cost.csv",
                     "closed_loop_series_T2.csv"):
            p = os.path.join(ZENODO, name)
            if os.path.isfile(p):
                f.write("- `%s` (%d bytes, sha256 `%s`)\n" % (name, os.path.getsize(p), sha256_of(p)))
    print("MANIFEST.md escrito em %s" % manifest_path)


if __name__ == "__main__":
    main()
