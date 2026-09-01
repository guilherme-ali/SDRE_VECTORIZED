# -*- coding: utf-8 -*-
"""
Verifica a PROCEDENCIA de cada captura em outputs/: de qual commit, de qual
build e de qual chip ela veio, e se todas concordam entre si.

Por que isto existe: ate a v8 nenhuma captura serial dizia de qual firmware
tinha saido. Foi assim que uma captura anterior a otimizacao push-through do
ADDA sobreviveu em duas versoes do artigo reportando 12.17 ms onde o codigo do
repositorio mede 9.62 ms. O firmware agora emite, como primeira linha:

    STAMP,<git_rev>,<dirty>,<build_epoch>,<compilado>,<chip>,<rev>,<nucleos>,<mhz>

Este script le esse carimbo, e reprova a campanha quando:
  - falta carimbo (captura anterior a instrumentacao, procedencia desconhecida);
  - os commits divergem entre capturas (dados de builds diferentes no mesmo artigo);
  - a arvore estava suja no build (dirty=1) — o commit nao descreve o binario;
  - o chip nao e' o esperado para aquele experimento (S2 vs S3 trocados);
  - o clock nao e' 240 MHz (invalidaria toda comparacao de tempo).

Uso: python python/verifica_procedencia.py [--exigir-limpo]
"""
import argparse
import subprocess
import re
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")

# experimento -> (arquivo, chip esperado)
CAPTURAS = [
    ("tolerancia",     "serial_tolerance_sweep_frobenius.txt", "ESP32-S2"),
    ("gamma",          "serial_gamma_sweep.txt",               "ESP32-S2"),
    ("sweep_qr",       "serial_sweep_qr_v4.txt",               "ESP32-S2"),
    ("tol_qr",         "serial_tol_qr_sweep_A.txt",            "ESP32-S2"),
    ("fronteiras",     "serial_boundary_fine_B.txt",           "ESP32-S2"),
    ("repetibilidade", "serial_repeatability_D.txt",           "ESP32-S2"),
    ("bateria",        "serial_capture_bateria_v5_6traj.txt",  "ESP32-S2"),
    ("norma",          "serial_norm_benchmark.txt",            "ESP32-S2"),
    ("benchmark_s3",   os.path.join("s3", "serial_capture_bateria_s3.txt"), "ESP32-S3"),
    ("voo",            "serial_flightloop_E.txt",              "ESP32-S2"),
]

CLOCK_ESPERADO_MHZ = 240

# Experimentos cuja arvore e' modificada DE PROPOSITO no momento do build, com a
# razao documentada. Nao e' indulgencia: o carimbo continua marcando "dirty", e a
# razao aparece no relatorio — o que se dispensa e' tratar como anomalia algo que
# e' parte declarada do procedimento.
DIRTY_ESPERADO = {
    "voo": "runner liga DEBUG_MODE=true em src/main.cpp para instrumentar o ciclo, "
           "e restaura ao terminar (ver flight_debug_mode em run_experiments.py)",
}


CARIMBO_RE = re.compile(
    r"STAMP,([^,\s]+),(\d+),(\d+),([^,]+),([^,]+),(\d+),(\d+),(\d+)")


def le_carimbo(path, max_bytes=400000):
    """Devolve o primeiro carimbo BEM FORMADO encontrado no inicio do arquivo.

    Nao exige inicio de linha nem posicao fixa: o host abre a porta com a placa
    ja' transmitindo, entao a captura comeca rotineiramente com lixo de
    sincronizacao colado ao primeiro texto — na primeira campanha carimbada a
    linha saiu como "STAMP,ab678c8-di", truncada. O firmware emite o carimbo
    duas vezes justamente por isso; aqui basta achar uma ocorrencia completa.
    """
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.read(max_bytes)
    m = CARIMBO_RE.search(head)
    if not m:
        return None
    return {
        "git_rev": m.group(1), "dirty": int(m.group(2)),
        "build_epoch": int(m.group(3)), "compilado": m.group(4),
        "chip": m.group(5), "chip_rev": m.group(6),
        "nucleos": m.group(7), "mhz": int(m.group(8)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exigir-limpo", action="store_true",
                     help="reprova tambem quando o build veio de arvore suja (dirty=1)")
    args = ap.parse_args()

    print("%-16s %-18s %-6s %-19s %-10s %-6s %s" %
          ("experimento", "commit", "dirty", "build", "chip", "MHz", "veredito"))
    print("-" * 104)

    problemas = []
    commits = {}
    for key, rel, chip_esp in CAPTURAS:
        path = os.path.join(OUT, rel)
        if not os.path.isfile(path):
            print("%-16s %-18s %-6s %-19s %-10s %-6s %s" %
                  (key, "-", "-", "-", "-", "-", "SEM CAPTURA"))
            problemas.append("%s: sem captura" % key)
            continue
        st = le_carimbo(path)
        if st is None:
            print("%-16s %-18s %-6s %-19s %-10s %-6s %s" %
                  (key, "?", "?", "?", "?", "?", "SEM CARIMBO (procedencia desconhecida)"))
            problemas.append("%s: sem carimbo" % key)
            continue

        build = time.strftime("%d/%m/%Y %H:%M", time.localtime(st["build_epoch"]))
        v = []
        if st["chip"] != chip_esp:
            v.append("CHIP ERRADO (esperado %s)" % chip_esp)
        if st["mhz"] != CLOCK_ESPERADO_MHZ:
            v.append("CLOCK %d MHz" % st["mhz"])
        if st["dirty"] and (args.exigir_limpo or key not in DIRTY_ESPERADO):
            v.append("ARVORE SUJA (commit nao descreve o binario)")
        veredito = "; ".join(v) if v else ("ok (dirty esperado)" if st["dirty"] else "ok")
        if v:
            problemas.append("%s: %s" % (key, veredito))
        # o sufixo -dirty faz o mesmo commit parecer dois; normaliza para comparar
        commits.setdefault(st["git_rev"].replace("-dirty", ""), []).append(key)
        print("%-16s %-18s %-6s %-19s %-10s %-6d %s" %
              (key, st["git_rev"], "sim" if st["dirty"] else "nao",
               build, st["chip"], st["mhz"], veredito))

    print("-" * 104)
    for key, razao in DIRTY_ESPERADO.items():
        if any(k == key for _, k in [(0, key)]):
            print("\nnota: '%s' compila com a arvore modificada de proposito --\n      %s" % (key, razao))

    if len(commits) > 1:
        print("\nCAPTURAS DE COMMITS DIFERENTES:")
        for rev, keys in commits.items():
            print("   %-18s %s" % (rev, ", ".join(keys)))
        # Commits diferentes so' sao aceitaveis se estiverem na MESMA linha de
        # historia e a diferenca nao tocar codigo compartilhado. Um commit que
        # so' acrescenta o carimbo ao firmware de voo, por exemplo, nao pode
        # alterar binario de experimento — os envs de experimento excluem
        # src/main.cpp via build_src_filter. O que nao se aceita e' commits sem
        # parentesco, que descrevem arvores incomparaveis.
        revs = sorted(commits)
        # Sem repositorio git nao da' para responder a pergunta. E' o caso de
        # quem baixa o code.zip do Zenodo: `git archive` nao leva o `.git`.
        # Antes de 2026-09-01 todo merge-base falhava nesse cenario e o script
        # concluia "SEM PARENTESCO", reprovando a auditoria -- transformando a
        # falta da ferramenta numa afirmacao sobre o dado.
        # Nao basta perguntar se ha' um repositorio: a pasta pessoal do autor e'
        # um repositorio git, e `git rev-parse --git-dir` responde 0 de qualquer
        # lugar sob ela, inclusive de dentro do snapshot do Zenodo. O que decide
        # e' se o repositorio corrente conhece os commits carimbados.
        conhece = all(
            subprocess.call(["git", "cat-file", "-e", r + "^{commit}"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL) == 0
            for r in revs
        )
        if not conhece:
            print("\n   [INFO] este repositorio nao conhece %s -- provavel execucao"
                  % ", ".join(revs))
            print("   a partir do snapshot do Zenodo, que nao carrega o historico git.")
            print("   O parentesco nao pode ser verificado aqui; confira no repositorio publico:")
            print("   git merge-base --is-ancestor %s %s" % (revs[0], revs[-1]))
            linear = None
        else:
            linear = True
            for i in range(len(revs) - 1):
                a, b = revs[i], revs[i + 1]
                ok_ab = subprocess.call(["git", "merge-base", "--is-ancestor", a, b],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
                ok_ba = subprocess.call(["git", "merge-base", "--is-ancestor", b, a],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
                if not (ok_ab or ok_ba):
                    linear = False
        if linear:
            print("\n   os commits estao na mesma linha de historia; diferenca entre eles:")
            try:
                d = subprocess.check_output(["git", "diff", "--stat", revs[0], revs[-1]],
                                             text=True, stderr=subprocess.DEVNULL)
                for l in d.strip().splitlines():
                    print("      " + l)
            except Exception:
                pass
            print("   -> revise se essa diferenca poderia afetar os binarios das demais capturas.")
        elif linear is False:
            problemas.append("capturas de commits SEM PARENTESCO: %s" % ", ".join(revs))
    elif commits:
        rev = next(iter(commits))
        print("\nTodas as capturas carimbadas vieram do mesmo build: %s" % rev)

    if problemas:
        print("\n%d PROBLEMA(S) DE PROCEDENCIA:" % len(problemas))
        for p in problemas:
            print("   - " + p)
        return 1
    print("\nProcedencia integra: mesmo commit, chips corretos, clock nominal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
