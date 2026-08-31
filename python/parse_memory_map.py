"""
Extrai métricas de memória do linker map do firmware de voo (xtensa-esp32s2-elf-gcc)
para compor o pacote de dados do artigo DINAME 2027 (v8): IRAM ocupada pelas rotinas
Q13.18, RAM (DRAM) total e % de 320 KB, Flash total e % da partição.

O env de voo (esp32-s2-saola-1, platformio.ini) já grava
`-Wl,-Map=.pio/build/esp32-s2-saola-1/output.map`, mas o SConscript do platform
espressif32 aplica seu próprio `-Wl,-Map=firmware.map` DEPOIS desse flag — o linker
usa o último `-Map` da linha de comando, então o arquivo realmente escrito é
`firmware.map`, não `output.map` (confirmado rodando `pio run -e esp32-s2-saola-1`
e inspecionando `.pio/build/esp32-s2-saola-1/`). Este script procura os dois nomes.

Fórmulas usadas (as mesmas do pedido original, não as que `pio run` imprime no
console — ver --help e o aviso abaixo):
  RAM   = tamanho(.dram0.data) + tamanho(.dram0.bss), % de --ram-total-kb (320 KB).
          Bate exatamente com o "RAM: ... used N bytes" que `pio run` imprime.
  Flash = tamanho(.flash.text) + tamanho(.flash.rodata), % de --flash-partition-kb
          (default 1280 KB = partição app0 padrão da board, 0x140000 bytes).
          Este número NÃO bate com o "Flash: ... used N bytes" do `pio run`, que
          inclui também o conteúdo copiado para IRAM/DRAM no boot (overhead de
          imagem, não código/dado "puro") — a definição aqui é a solicitada
          (.flash.text + .flash.rodata), documentada para não gerar confusão
          numa auditoria futura.
  IRAM Q13.18 = soma dos sub-blocos de código (.iram1.N, sem contar pools de
          literais .iram1.N.literal separadamente) dentro de .iram0.text cujo
          arquivo-objeto contribuinte está em --iram-objects (default
          FixedPointQ.cpp.o — o kernel compartilhado com IRAM_ATTR/FXQ_FAST_ATTR,
          ver lib/AUTOLQR/FixedPointQ.h). O pool de literais é reportado à parte
          (iram_literal_bytes) para transparência, não somado ao número principal.

Uso:
    python python/parse_memory_map.py
    python python/parse_memory_map.py --env esp32-s2-saola-1 --iram-objects FixedPointQ.cpp.o,AutoLQR.cpp.o
    python python/parse_memory_map.py --map caminho/para/outro.map
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECTION_HEADER_RE = re.compile(r"^(\.\S+)\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s*$")
SUBENTRY_INLINE_RE = re.compile(r"^ (\.\S+)\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+(\S.*)$")
SUBENTRY_NAME_ONLY_RE = re.compile(r"^ (\.\S+)\s*$")
SUBENTRY_CONT_RE = re.compile(r"^\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+(\S.*)$")


def find_map_file(env: str, map_override: str | None) -> str:
    if map_override:
        if not os.path.isfile(map_override):
            sys.exit("Arquivo .map nao encontrado: %s" % map_override)
        return map_override
    build_dir = os.path.join(REPO, ".pio", "build", env)
    for name in ("firmware.map", "output.map"):
        p = os.path.join(build_dir, name)
        if os.path.isfile(p):
            return p
    sys.exit(
        "Nenhum .map encontrado em %s (procurado firmware.map e output.map).\n"
        "Rode 'pio run -e %s' primeiro (o build gera o .map como efeito colateral "
        "do link)." % (build_dir, env)
    )


def parse_top_level_sections(lines):
    """Retorna {nome_secao: tamanho_bytes} para as linhas de topo (coluna 0)."""
    out = {}
    for line in lines:
        m = SECTION_HEADER_RE.match(line)
        if m:
            out[m.group(1)] = int(m.group(3), 16)
    return out


def sum_iram_by_object(lines, target_objects):
    """
    Soma bytes de código (não-literal) dentro do bloco .iram0.text cujo objeto
    contribuinte esteja em target_objects (ex.: "FixedPointQ.cpp.o").
    Retorna (code_bytes, literal_bytes, by_symbol) — by_symbol é uma lista de
    (nome_simbolo_ou_secao, bytes) para detalhamento.
    """
    in_block = False
    code_bytes = 0
    literal_bytes = 0
    by_symbol = []
    pending_name = None

    def matches_target(obj_path: str) -> bool:
        return any(obj in obj_path for obj in target_objects)

    for line in lines:
        if not in_block:
            m = SECTION_HEADER_RE.match(line)
            if m and m.group(1) == ".iram0.text":
                in_block = True
            continue

        # saiu do bloco .iram0.text ao encontrar a próxima secao de topo
        if SECTION_HEADER_RE.match(line):
            break

        m_inline = SUBENTRY_INLINE_RE.match(line)
        if m_inline:
            name, _addr, size_hex, obj_path = m_inline.groups()
            pending_name = None
            if matches_target(obj_path):
                size = int(size_hex, 16)
                if name.endswith(".literal"):
                    literal_bytes += size
                else:
                    code_bytes += size
                    by_symbol.append((name, size))
            continue

        m_name = SUBENTRY_NAME_ONLY_RE.match(line)
        if m_name:
            pending_name = m_name.group(1)
            continue

        if pending_name is not None:
            m_cont = SUBENTRY_CONT_RE.match(line)
            if m_cont:
                _addr, size_hex, obj_path = m_cont.groups()
                if matches_target(obj_path):
                    size = int(size_hex, 16)
                    if pending_name.endswith(".literal"):
                        literal_bytes += size
                    else:
                        code_bytes += size
                        by_symbol.append((pending_name, size))
                pending_name = None

    return code_bytes, literal_bytes, by_symbol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="esp32-s2-saola-1", help="env do PlatformIO cujo build inspecionar")
    ap.add_argument("--map", default=None, help="caminho explicito para o .map (ignora --env)")
    ap.add_argument("--iram-objects", default="FixedPointQ.cpp.o",
                     help="lista separada por virgula de nomes de objeto (ex.: FixedPointQ.cpp.o,AutoLQR.cpp.o)")
    ap.add_argument("--ram-total-kb", type=float, default=320.0,
                     help="capacidade total de RAM p/ calculo de %% (default 320 KB, ESP32-S2)")
    ap.add_argument("--flash-partition-kb", type=float, default=1280.0,
                     help="tamanho da particao de flash p/ calculo de %% (default 1280 KB = 0x140000, app0 padrao)")
    ap.add_argument("--out-json", default=None, help="default: outputs/v8/memoria_v8.json")
    ap.add_argument("--out-tex", default=None, help="default: outputs/v8/memoria_v8.tex")
    args = ap.parse_args()

    map_path = find_map_file(args.env, args.map)
    with open(map_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    lines = [l.rstrip("\n") for l in lines]

    sections = parse_top_level_sections(lines)
    for required in (".iram0.text", ".dram0.data", ".dram0.bss", ".flash.text", ".flash.rodata"):
        if required not in sections:
            sys.exit("Secao %s nao encontrada em %s — o .map tem o formato esperado?" % (required, map_path))

    target_objects = [o.strip() for o in args.iram_objects.split(",") if o.strip()]
    iram_code, iram_literal, iram_by_symbol = sum_iram_by_object(lines, target_objects)

    ram_bytes = sections[".dram0.data"] + sections[".dram0.bss"]
    ram_total_bytes = args.ram_total_kb * 1024
    ram_pct = 100.0 * ram_bytes / ram_total_bytes

    flash_bytes = sections[".flash.text"] + sections[".flash.rodata"]
    flash_total_bytes = args.flash_partition_kb * 1024
    flash_pct = 100.0 * flash_bytes / flash_total_bytes

    result = {
        "map_file": os.path.relpath(map_path, REPO),
        "env": args.env,
        "iram_q13_18": {
            "objects": target_objects,
            "code_bytes": iram_code,
            "code_kb": iram_code / 1024.0,
            "literal_bytes": iram_literal,
            "literal_kb": iram_literal / 1024.0,
            "total_bytes": iram_code + iram_literal,
            "total_kb": (iram_code + iram_literal) / 1024.0,
            "by_symbol": [{"symbol": s, "bytes": b} for s, b in iram_by_symbol],
        },
        "ram": {
            "data_bytes": sections[".dram0.data"],
            "bss_bytes": sections[".dram0.bss"],
            "used_bytes": ram_bytes,
            "used_kb": ram_bytes / 1024.0,
            "total_kb": args.ram_total_kb,
            "pct": ram_pct,
        },
        "flash": {
            "text_bytes": sections[".flash.text"],
            "rodata_bytes": sections[".flash.rodata"],
            "used_bytes": flash_bytes,
            "used_kb": flash_bytes / 1024.0,
            "partition_kb": args.flash_partition_kb,
            "pct": flash_pct,
        },
    }

    print("Arquivo .map: %s" % result["map_file"])
    print()
    print("IRAM Q13.18 (objetos: %s):" % ", ".join(target_objects))
    print("  codigo:          %6d bytes (%.2f KB)" % (iram_code, iram_code / 1024.0))
    print("  pool de literais:%6d bytes (%.2f KB)  [reportado a parte, nao somado ao total principal]" %
          (iram_literal, iram_literal / 1024.0))
    for sym, b in iram_by_symbol:
        print("    - %-70s %6d bytes" % (sym, b))
    print()
    print("RAM (.dram0.data + .dram0.bss): %d bytes = %.2f KB (%.1f%% de %.0f KB)" %
          (ram_bytes, ram_bytes / 1024.0, ram_pct, args.ram_total_kb))
    print("Flash (.flash.text + .flash.rodata): %d bytes = %.2f KB (%.1f%% de %.0f KB)" %
          (flash_bytes, flash_bytes / 1024.0, flash_pct, args.flash_partition_kb))
    print()
    print("AVISO: o numero de Flash acima usa a formula solicitada (.flash.text+.flash.rodata),")
    print("que NAO bate com o \"Flash: ... used\" que 'pio run' imprime no console — aquele inclui")
    print("tambem o conteudo copiado para IRAM/DRAM no boot (overhead de imagem). O de RAM bate exatamente.")

    out_json = args.out_json or os.path.join(REPO, "outputs", "v8", "memoria_v8.json")
    out_tex = args.out_tex or os.path.join(REPO, "outputs", "v8", "memoria_v8.tex")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nJSON escrito em %s" % out_json)

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(
            "%% gerado por python/parse_memory_map.py a partir de %s — nao editar a mao\n"
            "%% IRAM ocupada pelo kernel Q13.18 compartilhado (%s)\n"
            "\\newcommand{\\memIramKB}{%.2f}\n"
            "\\newcommand{\\memRamKB}{%.2f}\n"
            "\\newcommand{\\memRamPct}{%.1f}\n"
            "\\newcommand{\\memFlashKB}{%.2f}\n"
            "\\newcommand{\\memFlashPct}{%.1f}\n"
            % (
                result["map_file"], ", ".join(target_objects),
                iram_code / 1024.0,
                ram_bytes / 1024.0, ram_pct,
                flash_bytes / 1024.0, flash_pct,
            )
        )
    print("Trecho LaTeX escrito em %s" % out_tex)


if __name__ == "__main__":
    main()
