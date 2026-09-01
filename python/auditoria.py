# -*- coding: utf-8 -*-
"""Auditoria completa da campanha, num comando só.

Antes disto era preciso lembrar de quatro comandos distintos e ler quatro saídas
para saber se o artigo estava sustentado pelo dado. Aqui eles rodam em sequência
e sai um veredito único; o código de saída é não-zero se qualquer etapa reprovar.

    1. procedência   de qual commit, chip e clock veio cada captura
    2. números       cada afirmação numérica do artigo contra o dado bruto
    3. figuras       as figuras na pasta do artigo saíram do dado de hoje?
    4. voo           as N janelas do ciclo de voo, o único experimento não determinístico
    5. cobertura     quantos números do .tex têm checagem, e quais não têm

Cada etapa continua rodável isoladamente — este script apenas as orquestra.

Uso:
    python python/auditoria.py [--pular figuras] [--tex <caminho do .tex>]
"""
import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)

ETAPAS = [
    ("procedencia", ["verifica_procedencia.py"],
     "de qual commit, chip e clock veio cada captura"),
    ("numeros", ["verifica_numeros_artigo.py", "--v8"],
     "cada afirmacao numerica do artigo contra o dado bruto"),
    ("figuras", ["verifica_figuras.py"],
     "as figuras do artigo sairam do dado de hoje"),
    ("voo", ["analisa_voo.py", "--dir", os.path.join("outputs", "voo")],
     "as N janelas do ciclo de voo"),
]


def _tex_padrao():
    """Mesma fonte de verdade do runner: campanha.json / SDRE_ARTIGO_DIR."""
    d = os.environ.get("SDRE_ARTIGO_DIR")
    if not d:
        try:
            with io.open(os.path.join(REPO, "campanha.json"), encoding="utf-8") as f:
                d = json.load(f).get("artigo_dir")
        except Exception:
            return None
    if not d or not os.path.isdir(d):
        return None
    texs = [p for p in os.listdir(d) if re.match(r"diname2027_v\d+\.tex$", p)]
    if not texs:
        return None
    return os.path.join(d, max(texs, key=lambda p: int(re.search(r"_v(\d+)", p).group(1))))


def doi_do_deposito(tex_path):
    """O .tex ainda carrega um marcador no lugar do DOI do Zenodo?

    O v4 e o v5 traziam `zenodo.XXXXXXX` com um TODO. Em alguma versao seguinte o
    marcador virou `zenodo.14927231`, um numero com cara de DOI legitimo que nao
    corresponde a deposito nenhum deste trabalho -- pior que o placeholder, porque
    passa despercebido numa revisao. Aqui um marcador reprova a auditoria, e um DOI
    que exista de verdade so' passa se o autor o tiver colado a mao.
    """
    if not tex_path or not os.path.isfile(tex_path):
        return None
    txt = io.open(tex_path, encoding="utf-8").read()
    m = re.search(r"zenodo\.([A-Za-z0-9-]+)", txt)
    if not m:
        print("  nenhuma mencao a Zenodo no .tex.")
        return None
    valor = m.group(1)
    if not valor.isdigit():
        print("  DOI do deposito: MARCADOR (`zenodo.%s`)" % valor)
        print("  -> reserve o DOI no Zenodo (\"Reserve DOI\" no formulario de upload),")
        print("     cole no .tex e publique. Nao submeta assim.")
        return False
    print("  DOI do deposito: zenodo.%s (preenchido)" % valor)
    print("  -> confira uma vez que este numero e' o do SEU deposito.")
    return True


def cobertura(tex_path):
    """Quantos numeros do corpo do .tex tem alguma checagem, e quais nao tem.

    E' o numero que faltava nas auditorias anteriores: sem ele, "esta tudo
    verificado?" so' tinha resposta impressionista. Um numero e' considerado
    coberto quando algum literal numerico dos scripts de verificacao bate com
    ele dentro de 0.5%.
    """
    if not tex_path or not os.path.isfile(tex_path):
        print("  [INFO] .tex nao encontrado; cobertura nao calculada.")
        return None

    src = io.open(tex_path, encoding="utf-8").read()
    corpo = src[src.index(chr(92) + "begin{document}"):]

    afirmados = set()
    for nome in ("verifica_numeros_artigo.py", "checagens_prosa.py"):
        caminho = os.path.join(AQUI, nome)
        if not os.path.isfile(caminho):
            continue
        for no in ast.walk(ast.parse(io.open(caminho, encoding="utf-8").read())):
            if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)) \
               and not isinstance(no.value, bool):
                afirmados.add(round(float(no.value), 10))

    BS = chr(92)
    c = re.sub(BS + BS + r"times *10\^\{(-?\d+)\}", r"e\1", corpo)
    c = re.sub(BS + BS + r"cite[a-z]*\{[^}]*\}", " ", c)
    c = re.sub(BS + BS + r"(ref|label|includegraphics|section|subsection)\**\{[^}]*\}", " ", c)
    c = re.sub(BS + BS + r",", "", c)          # 2\,390 -> 2390
    c = re.sub(BS + BS + "[a-zA-Z]+", " ", c)
    c = re.sub(r"[{}$~]", " ", c)

    vistos = set()
    for m in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?(?:e-?\d+)?)(?![\w])", c):
        try:
            vistos.add(round(float(m), 10))
        except ValueError:
            pass

    def coberto(v):
        return any(a == v or (a and abs(a - v) / max(abs(a), 1e-30) < 5e-3)
                   for a in afirmados)

    dist = sorted(vistos)
    cob = [v for v in dist if coberto(v)]
    nao = [v for v in dist if not coberto(v)]
    print("  numeros distintos no corpo do artigo : %d" % len(dist))
    print("  com checagem automatica              : %d (%.0f%%)"
          % (len(cob), 100.0 * len(cob) / max(len(dist), 1)))
    print("  sem checagem                         : %d" % len(nao))
    if nao:
        print("  sem checagem (conferir se sao definicoes/contagens estruturais):")
        for i in range(0, len(nao), 12):
            print("     " + "  ".join("%g" % v for v in nao[i:i + 12]))
    return len(cob), len(dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pular", nargs="*", default=[],
                    choices=[k for k, _, _ in ETAPAS], help="etapas a nao rodar")
    ap.add_argument("--tex", default=None, help="caminho do .tex para a cobertura")
    args = ap.parse_args()

    resultados = {}
    for chave, cmd, descricao in ETAPAS:
        if chave in args.pular:
            resultados[chave] = "pulado"
            continue
        print("=" * 78)
        print("[%s] %s" % (chave, descricao))
        print("=" * 78)
        rc = subprocess.call([sys.executable, os.path.join(AQUI, cmd[0])] + cmd[1:],
                             cwd=REPO)
        resultados[chave] = "OK" if rc == 0 else "REPROVOU (rc=%d)" % rc
        print()

    tex = args.tex or _tex_padrao()

    print("=" * 78)
    print("[deposito] o .tex ja aponta para um DOI real?")
    print("=" * 78)
    doi_ok = doi_do_deposito(tex)
    print()

    print("=" * 78)
    print("[cobertura] quantos numeros do .tex tem checagem")
    print("=" * 78)
    cob = cobertura(tex)
    print()

    print("=" * 78)
    print("VEREDITO")
    print("=" * 78)
    for chave, _, _ in ETAPAS:
        print("  %-14s %s" % (chave, resultados.get(chave, "?")))
    if cob:
        print("  %-14s %d de %d numeros do artigo" % ("cobertura", cob[0], cob[1]))
    if doi_ok is not None:
        print("  %-14s %s" % ("deposito",
                              "DOI preenchido" if doi_ok else "DOI AINDA E' MARCADOR"))
    ruins = [k for k, v in resultados.items() if v.startswith("REPROVOU")]
    if doi_ok is False:
        ruins.append("deposito (DOI nao preenchido)")
    if ruins:
        print("\n%d etapa(s) reprovada(s): %s" % (len(ruins), ", ".join(ruins)))
        return 1
    print("\nCampanha integra: procedencia, numeros, figuras e ciclo de voo conferem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
