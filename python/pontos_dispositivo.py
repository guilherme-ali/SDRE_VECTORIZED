# -*- coding: utf-8 -*-
"""
Pontos de operacao COMO O DISPOSITIVO OS VIU.

O firmware da bateria emite, para cada ponto, uma linha

    PT,<traj>,<k>,<t>,<phi>,<theta>,<psi>,<p>,<q>,<r>

com o estado que ELE de fato usou para montar A(x) e resolver a DARE. Toda
analise de host que precise do ponto de operacao deve consumir isto, e nao
regenerar a trajetoria em Python.

Por que: a referencia de dupla precisao (e_K, condicionamento, ||P||_F) e'
obtida resolvendo a DARE no ponto de operacao. Se o ponto do host nao for o
ponto do dispositivo, a diferenca entre os dois modelos entra na conta como se
fosse erro do solver. Foi o que acontecia em T4: o alvo do degrau vem do SINAL
de um seno, a grade de 6 ms cai exatamente sobre os zeros dessa funcao, e ali
float32 e float64 escolhem alvos opostos (+40 contra -40 graus). O e_K em T4
media' 4.18e-4 contra 1.5e-6 no ponto real — 280x — e inflava a coluna float da
Tabela 1 de 1.9e-6 para 3.7e-6, metade do numero publicado sendo descasamento
de modelo, nao quantizacao.

O espelho em python/trajetorias.py continua util para gerar trajetorias quando
nao ha captura (simulacao de malha fechada, planejamento), mas deixa de ser a
fonte de verdade do que foi medido.
"""
import collections
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATERIA = os.path.join(REPO, "outputs", "serial_capture_bateria_v5_6traj.txt")

CAMPOS = ("t", "phi", "theta", "psi", "p", "q", "r")


def carregar(path=BATERIA):
    """Retorna {traj: {campo: np.array indexado por k}} a partir das linhas PT.

    Levanta FileNotFoundError se a captura nao existir — deliberadamente, para
    que um consumidor nunca caia em silencio no espelho de host achando que
    esta lendo o dado medido.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "captura da bateria ausente (%s): sem ela nao ha linhas PT, e o "
            "ponto de operacao medido nao pode ser recuperado." % path)

    bruto = collections.defaultdict(dict)
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("PT,"):
            continue
        p = line.rstrip("\n").split(",")
        if len(p) < 10:
            continue
        try:
            bruto[p[1]][int(p[2])] = tuple(float(x) for x in p[3:10])
        except ValueError:
            continue

    out = {}
    for traj, pontos in bruto.items():
        ks = sorted(pontos)
        if ks != list(range(len(ks))):
            raise ValueError("PT de %s tem indices faltando ou fora de ordem "
                             "(%d pontos, k vai de %d a %d)" %
                             (traj, len(ks), ks[0], ks[-1]))
        arr = np.array([pontos[k] for k in ks], dtype=float)
        out[traj] = {c: arr[:, i] for i, c in enumerate(CAMPOS)}
    return out


def disponivel(path=BATERIA):
    return os.path.isfile(path)


if __name__ == "__main__":
    d = carregar()
    print("%-16s %8s  %s" % ("traj", "n", "faixas (tilt max, |taxa| max)"))
    for t, v in sorted(d.items()):
        tilt = np.degrees(np.hypot(v["phi"], v["theta"])).max()
        rate = np.degrees(np.sqrt(v["p"] ** 2 + v["q"] ** 2 + v["r"] ** 2)).max()
        print("%-16s %8d  tilt<=%.1f deg, |w|<=%.0f deg/s" % (t, len(v["t"]), tilt, rate))
