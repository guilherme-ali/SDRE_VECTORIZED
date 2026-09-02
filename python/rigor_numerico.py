# -*- coding: utf-8 -*-
"""Confere TODA afirmacao numerica pelo criterio do valor impresso.

Por que existe: a auditoria aprova cada checagem contra a sua propria tolerancia
relativa -- padrao 2%, e ate' 25% em algumas. Para um numero de dois algarismos,
2% ja' permite errar o ultimo digito. Foi assim que o "0,27%" (dado: 0,262%)
sobreviveu ate' 2026-09-01, quando apareceu por acaso na extracao do texto da
Figura 5(b), e nao por varredura.

Tolerancia relativa e' o criterio errado para conferir um numero *impresso*:
apertada demais reprova arredondamento correto (2,9e-13 contra 2,94843e-13),
frouxa demais aprova digito errado. O criterio certo nao e' uma porcentagem --
e' arredondar o medido para os algarismos significativos com que o artigo o
imprime e exigir igualdade.

Este script roda a auditoria inteira, recupera o literal de cada afirmacao no
codigo-fonte (o AST preserva o "4.70" que o float perderia) e reaplica o
criterio. Afirmacoes que sao limite ou aproximacao declarada ficam de fora, em
APROXIMADAS, com o motivo ao lado.

    python python/rigor_numerico.py

Sai com codigo 1 se alguma afirmacao exata nao arredondar para o valor impresso.
"""
import ast
import io
import os
import re
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)
FONTES = ("checagens_prosa.py", "verifica_numeros_artigo.py")

# Afirmacoes que NAO sao o valor exato: limites superiores/inferiores e
# aproximacoes que o texto declara como tais. O criterio de arredondamento nao
# se aplica; o que se confere e' a desigualdade ou a ordem de grandeza.
APROXIMADAS = {
    "prosa: ||AcTs|| maximo em T1/T2/T4/T5 (<0.05)":
        "limite: o texto diz 'stays below 0.05'",
    "prosa: termo desprezado em T1/T2/T4/T5 (<2e-5)":
        "limite: o texto diz 'below 2e-5'",
    "prosa: margem ate o teto (fator 8)":
        "aproximacao: o texto diz 'a factor of eight below the ceiling'",
    "prosa: tau=1e-3 fica 16x acima do pior piso":
        "aproximacao: o texto diz 'a factor of 16 above'",
    "prosa: essa diferenca e' 2% do erro de rastreamento":
        "aproximacao: o texto diz 'two per cent of the tracking error'",
    "prosa: taxa maxima do envelope (3000 graus/s)":
        "limite: o texto diz 'above 3000 deg/s'",
    "prosa: ||P||_F tipica (0.43)":
        "aproximacao: o texto diz 'Frobenius norm near 0.43'",
    "prosa: a cada 20 ciclos, pior acrescimo em T1-T5 (0.05%)":
        "limite: o texto diz 'by at most 0.05%'",
    "prosa: |dJ/J| maximo do doubling em T1-T5 (0.26%)":
        "limite: o texto diz 'by at most 0.26%'",
    "prosa: maior mudanca de residuo na familia (6.8%)":
        "limite: o texto diz 'at most 6.8%'",
    "prosa: teto do formato Q13.18 (8192)":
        "definicao do formato, nao medida",
    "prosa: nenhum ponto acima de 0.15%":
        "limite: o texto diz 'no single point exceeding 0.15%' (pior: 0.1414%)",
    "prosa: SDA-fx nunca atinge 62% do periodo":
        "limite: o texto diz 'never reaches 62% of the control period'",
    "prosa: inclinacao maxima do envelope (80 graus)":
        "nominal: a trajetoria T5 e' projetada para 80 graus de tilt",
}


def literais_do_fonte():
    """label -> texto do literal como escrito no codigo (preserva '4.70')."""
    mapa = {}
    for nome in FONTES:
        caminho = os.path.join(AQUI, nome)
        if not os.path.isfile(caminho):
            continue
        src = io.open(caminho, encoding="utf-8").read()
        arvore = ast.parse(src)
        for no in ast.walk(arvore):
            if not isinstance(no, ast.Call):
                continue
            alvo = getattr(no.func, "id", None)
            if alvo == "check" and len(no.args) >= 2:
                rotulo, valor = no.args[0], no.args[1]
            elif alvo == "check_impresso" and len(no.args) >= 3:
                rotulo, valor = no.args[1], no.args[2]
            else:
                continue
            if not (isinstance(rotulo, ast.Constant)
                    and isinstance(rotulo.value, str)):
                continue
            trecho = ast.get_source_segment(src, valor)
            if isinstance(valor, ast.Subscript):
                trecho = _do_dicionario(arvore, src, valor) or trecho
            if trecho:
                mapa[rotulo.value] = trecho.strip().strip('"').strip("'")
        mapa.update(literais_de_laco(arvore, src))
    return mapa


def _do_dicionario(arvore, src, no):
    """Resolve VOO_ESPERADO["ciclos"] para o literal escrito no dicionario."""
    nome = getattr(no.value, "id", None)
    chave = getattr(no.slice, "value", None)
    if not nome or not isinstance(chave, str):
        return None
    for topo in arvore.body:
        if not isinstance(topo, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == nome for t in topo.targets):
            continue
        if not isinstance(topo.value, ast.Dict):
            continue
        for k, v in zip(topo.value.keys, topo.value.values):
            if isinstance(k, ast.Constant) and k.value == chave:
                return ast.get_source_segment(src, v)
    return None


def _dicts_do_modulo(arvore, src):
    """nome -> {chave: [literais]} para dicionarios de literais no topo."""
    saida = {}
    for topo in ast.walk(arvore):
        if not isinstance(topo, ast.Assign) or not isinstance(topo.value, ast.Dict):
            continue
        for alvo in topo.targets:
            nome = getattr(alvo, "id", None)
            if not nome:
                continue
            entradas = {}
            for k, v in zip(topo.value.keys, topo.value.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                if isinstance(v, ast.Tuple):
                    entradas[k.value] = [ast.get_source_segment(src, e)
                                         for e in v.elts]
                else:
                    entradas[k.value] = [ast.get_source_segment(src, v)]
            if entradas:
                saida[nome] = entradas
    return saida


def literais_de_laco(arvore, src):
    """Rotulos montados dentro de `for chave, valor in DICT.items()`.

    As Tabelas 1 e 2 sao conferidas assim, entao sem isto 94 das afirmacoes --
    as celulas das duas tabelas -- ficariam fora do criterio de arredondamento.
    """
    dicts = _dicts_do_modulo(arvore, src)
    mapa = {}
    for laco in ast.walk(arvore):
        if not isinstance(laco, ast.For):
            continue
        it = laco.iter
        if not (isinstance(it, ast.Call)
                and getattr(it.func, "attr", None) == "items"):
            continue
        nome_dict = getattr(it.func.value, "id", None)
        entradas = dicts.get(nome_dict)
        if not entradas or not isinstance(laco.target, ast.Tuple):
            continue
        alvos = laco.target.elts
        if len(alvos) != 2:
            continue
        chave = getattr(alvos[0], "id", None)
        segundo = alvos[1]
        if isinstance(segundo, ast.Tuple):
            posicao = {getattr(e, "id", None): i for i, e in enumerate(segundo.elts)}
        else:
            posicao = {getattr(segundo, "id", None): 0}
        for no in ast.walk(laco):
            if not (isinstance(no, ast.Call)
                    and getattr(no.func, "id", None) == "check"
                    and len(no.args) >= 2):
                continue
            rot, val = no.args[0], no.args[1]
            if not (isinstance(rot, ast.BinOp) and isinstance(rot.op, ast.Mod)
                    and isinstance(rot.left, ast.Constant)
                    and getattr(rot.right, "id", None) == chave):
                continue
            i = posicao.get(getattr(val, "id", None))
            if i is None:
                continue
            for k, literais in entradas.items():
                if i < len(literais) and literais[i]:
                    mapa[rot.left.value % k] = literais[i].strip()
    return mapa


def significativos(texto):
    """Algarismos significativos impressos: '2.9e-13'->2, '4.70'->3, '0.004'->1."""
    corpo = texto.strip().lstrip("+-").split("e")[0].split("E")[0]
    inteiro, _, frac = corpo.partition(".")
    return len((inteiro + frac).lstrip("0")) or 1


def arredonda_como(texto, medido):
    """`medido` arredondado para os significativos com que `texto` e' impresso.

    Decimal e nao round(): o round() do Python opera na representacao binaria,
    onde 151.35 e' 151.34999... e arredonda para 151.3, enquanto toda convencao
    decimal da' 151.4. Quem escreve o artigo arredonda o decimal que leu.
    """
    import math
    from decimal import Decimal, ROUND_HALF_UP
    if medido == 0:
        return 0.0
    expoente = int(math.floor(math.log10(abs(medido))))
    casas = -(expoente - significativos(texto) + 1)
    passo = Decimal(1).scaleb(-casas)
    return float(Decimal(repr(medido)).quantize(passo, rounding=ROUND_HALF_UP))


def empate(texto, medido):
    """O medido cai exatamente no meio entre dois valores impressos possiveis?

    3.815 ms com duas casas fica entre 3.81 e 3.82, e as duas convencoes usuais
    discordam: meio-para-cima da' 3.82, meio-para-par (o padrao do NumPy, que
    produziu as tabelas) da' 3.81. Num empate as duas leituras sao corretas, e
    reprovar qualquer uma delas seria erro do conferidor, nao do artigo.
    """
    import math
    from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
    if medido == 0:
        return False
    expoente = int(math.floor(math.log10(abs(medido))))
    casas = -(expoente - significativos(texto) + 1)
    passo = Decimal(1).scaleb(-casas)
    d = Decimal(repr(medido))
    return d.quantize(passo, ROUND_HALF_UP) != d.quantize(passo, ROUND_HALF_EVEN)


def roda_auditoria():
    """Roda a auditoria e devolve (label, artigo, dados) de cada afirmacao."""
    saida = subprocess.run(
        [sys.executable, os.path.join(AQUI, "verifica_numeros_artigo.py"), "--v8"],
        cwd=REPO, capture_output=True, text=True, errors="replace").stdout
    linhas = []
    padrao = re.compile(r"^\s+(OK|XX)\s+(.+?)\s+artigo=(\S+)\s+dados=(\S+)\s+\(dif",
                        re.M)
    for m in padrao.finditer(saida):
        try:
            linhas.append((m.group(2).strip(), float(m.group(3)), float(m.group(4))))
        except ValueError:
            pass
    return linhas


def main():
    literais = literais_do_fonte()
    afirmacoes = roda_auditoria()
    if not afirmacoes:
        print("nenhuma afirmacao coletada -- a auditoria rodou?")
        return 1

    exatas, aprox, sem_fonte, falhas, empates = 0, 0, [], [], []
    for label, artigo, dados in afirmacoes:
        if label in APROXIMADAS:
            aprox += 1
            continue
        # o rotulo espelha a forma impressa no artigo ("(285 s)", "(2.9e-13)");
        # o literal do codigo traz um ".0" que o artigo nao imprime e que
        # inflaria a contagem de algarismos significativos.
        # o rotulo pode trazer mais de um numero ("(69.9 us, 3.1x)"): vale o
        # que corresponde a' afirmacao conferida, nao o primeiro que aparece.
        cands = [c.rstrip(".") for c in
                 re.findall(r"[-+]?\d[\d.]*(?:e[-+]?\d+)?", label)]
        texto = None
        if cands:
            def dist(c):
                try:
                    return abs(float(c) - artigo)
                except ValueError:
                    return float("inf")
            melhor = min(cands, key=dist)
            if dist(melhor) <= 1e-6 * max(1.0, abs(artigo)):
                texto = melhor
        if texto is None:
            texto = literais.get(label)
        if texto is None:
            texto = literais.get(label)
        if texto and re.fullmatch(r"-?\d+\.0", texto):
            texto = texto[:-2]          # 285.0 -> 285, 1051.0 -> 1051
        if texto is None:
            sem_fonte.append(label)
            continue
        try:
            obtido = arredonda_como(texto, dados)
        except (ValueError, OverflowError):
            sem_fonte.append(label)
            continue
        exatas += 1
        ref = abs(artigo) if artigo else 1.0
        if abs(obtido - artigo) > 1e-6 * ref:
            if empate(texto, dados):
                empates.append((label, texto, dados))
            else:
                falhas.append((label, texto, dados, obtido))

    print("=" * 92)
    print("RIGOR NUMERICO -- cada valor impresso tem de ser o arredondamento do medido")
    print("=" * 92)
    print("  afirmacoes conferidas por arredondamento : %d" % exatas)
    print("  limites e aproximacoes declaradas        : %d" % aprox)
    if empates:
        print("  empates no ultimo digito                 : %d" % len(empates))
    if sem_fonte:
        print("  sem literal recuperavel                  : %d" % len(sem_fonte))
        for l in sem_fonte[:10]:
            print("      %s" % l)
    print()
    if falhas:
        print("NAO ARREDONDAM PARA O VALOR IMPRESSO (%d):" % len(falhas))
        for label, texto, dados, obtido in falhas:
            print("  XX %-54s artigo=%-10s dados=%-12.6g arredonda para %s"
                  % (label[:54], texto, dados, "%.6g" % obtido))
        print()
        print("Cada um destes e' um digito errado no artigo ou um alvo errado no")
        print("script. Confira qual dos dois antes de corrigir.")
        return 1
    print("Nenhuma divergencia: todo valor impresso e' o arredondamento do medido.")
    if empates:
        print()
        print("Empates no ultimo digito (as duas convencoes discordam; o artigo usa")
        print("meio-para-par, o mesmo do NumPy que gerou as tabelas):")
        for label, texto, dados in empates:
            print("  ~~ %-54s artigo=%-10s dados=%.6g" % (label[:54], texto, dados))
        print()
    for label, motivo in sorted(APROXIMADAS.items()):
        if any(l == label for l, _, _ in afirmacoes):
            print("  [aprox] %-52s %s" % (label[:52], motivo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
