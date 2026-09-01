# -*- coding: utf-8 -*-
"""As figuras que estao no artigo saíram dos dados que estao no repositorio?

Furo que isto fecha: um PDF de figura na pasta do artigo era indistinguivel de
outro gerado meses antes. Compilar o .tex com uma figura velha nao produzia erro
nenhum -- e' o mesmo modo de falha das capturas sem carimbo, que deixou 12,17 ms
sobreviver duas versoes do artigo.

Desde 2026-09-01 `python/figuras_artigo_final.py` grava nos metadados de cada PDF:

    Author   commit do repositorio no momento da geracao (+ "-dirty")
    Subject  <captura de origem>=<12 primeiros digitos do SHA-256>; ...

Este script le esses metadados nas figuras da pasta do artigo e recalcula o hash
de cada fonte AGORA. Divergencia quer dizer uma de duas coisas, ambas graves:
a figura e' velha, ou o dado mudou depois dela.

Uso:
    python python/verifica_figuras.py [--figuras-dir <pasta>]
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
REGISTRO = os.path.join(OUT, "v8", "figuras_procedencia.json")


def _figuras_dir_padrao():
    """Mesma fonte de verdade do runner: campanha.json / SDRE_ARTIGO_DIR."""
    d = os.environ.get("SDRE_ARTIGO_DIR")
    if not d:
        try:
            with open(os.path.join(REPO, "campanha.json"), encoding="utf-8") as f:
                d = json.load(f).get("artigo_dir")
        except Exception:
            d = None
    return os.path.join(d, "Figures") if d else None


def _figuras_usadas(artigo_dir):
    """Nomes (sem .pdf) das figuras que o .tex de maior versao inclui.

    Devolve None quando nao da' para saber; nesse caso toda figura sem carimbo e'
    tratada como critica, que e' o lado seguro.
    """
    import glob as _glob
    import re as _re

    try:
        texs = _glob.glob(os.path.join(artigo_dir, "diname2027_v*.tex"))
        if not texs:
            return None
        alvo = max(texs, key=lambda p: [int(x) for x in _re.findall(r"_v(\d+)", p)] or [0])
        txt = open(alvo, encoding="utf-8", errors="replace").read()
        return set(_re.findall(r"includegraphics\[[^\]]*\]\{Figures/([A-Za-z0-9_]+)\}", txt))
    except Exception:
        return None


def sha12(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "ausente"
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figuras-dir", default=_figuras_dir_padrao())
    args = ap.parse_args()

    if not args.figuras_dir or not os.path.isdir(args.figuras_dir):
        raise SystemExit("pasta de figuras nao encontrada: %r "
                         "(use --figuras-dir ou configure artigo_dir em campanha.json)"
                         % args.figuras_dir)
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise SystemExit("PyMuPDF ausente: pip install pymupdf")

    if not os.path.isfile(REGISTRO):
        print("[INFO] %s ausente; conferindo apenas o que estiver nos PDFs."
              % os.path.relpath(REGISTRO, REPO))

    print("%-30s %-16s %s" % ("figura", "commit", "veredito"))
    print("-" * 92)

    problemas = []
    sem_carimbo = []
    for nome in sorted(os.listdir(args.figuras_dir)):
        if not nome.endswith(".pdf"):
            continue
        caminho = os.path.join(args.figuras_dir, nome)
        meta = fitz.open(caminho).metadata or {}
        subject = meta.get("subject") or ""
        if "=" not in subject:
            sem_carimbo.append(nome)
            print("%-30s %-16s %s" % (nome, "?", "SEM PROCEDENCIA (figura anterior a v8)"))
            continue

        divergentes = []
        for item in subject.split(";"):
            item = item.strip()
            if "=" not in item:
                continue
            rel, hash_gravado = item.rsplit("=", 1)
            atual = sha12(os.path.join(REPO, rel.replace("/", os.sep)))
            if atual != hash_gravado:
                divergentes.append("%s (figura=%s, agora=%s)" % (rel, hash_gravado, atual))

        if divergentes:
            veredito = "DADO MUDOU DEPOIS DA FIGURA"
            problemas.append((nome, divergentes))
        else:
            veredito = "ok"
        print("%-30s %-16s %s" % (nome, meta.get("author") or "?", veredito))

    print("-" * 92)

    if sem_carimbo:
        # Uma figura sem carimbo so' e' problema se o artigo a inclui; as demais
        # sao versoes anteriores que ficaram na pasta (o repositorio nao apaga
        # nada) e nao entram em PDF nenhum.
        usadas = _figuras_usadas(os.path.dirname(args.figuras_dir))
        criticas = [n for n in sem_carimbo if usadas is None or n[:-4] in usadas]
        ociosas = [n for n in sem_carimbo if n not in criticas]
        if criticas:
            print("\n%d figura(s) SEM PROCEDENCIA e incluida(s) pelo artigo — regerar com "
                  "'python python/figuras_artigo_final.py --flight-dir outputs/voo':"
                  % len(criticas))
            for n in criticas:
                print("   - " + n)
            problemas.append((", ".join(criticas), ["sem procedencia"]))
        if ociosas:
            print("\n%d figura(s) sem procedencia, nenhuma incluida pelo .tex "
                  "(versoes anteriores mantidas na pasta): %s"
                  % (len(ociosas), ", ".join(ociosas)))

    if problemas:
        print("\n%d FIGURA(S) DESSINCRONIZADA(S) DO DADO:" % len(problemas))
        for nome, divs in problemas:
            print("   %s" % nome)
            for d in divs:
                print("      %s" % d)
        print("\n   Regerar antes de recompilar o artigo.")
        return 1

    print("\nToda figura carimbada corresponde ao dado atual do repositorio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
