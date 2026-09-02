"""
Emulação em Python (inteiros nativos, precisão arbitrária, com os mesmos
limites explícitos de int32/int64) do kernel Q13.18 de
lib/AUTOLQR/FixedPointQ.{h,cpp} — usada para prever e explicar o erro de
quantização dos seis solvers `_FIXED` fora do alvo, e para gerar a figura
central do artigo (erro de K vs. faixa dinâmica ocupada, com o teto ±8192
marcado). Não substitui a verificação no hardware real: aqui apenas se
reproduz a ARITMÉTICA (mesmas operações, mesmos arredondamentos, mesmos
critérios de overflow); os limites de RAM/stack do ESP32-S2 não têm
equivalente aqui.

Cada função espelha, linha por linha, a correspondente em FixedPointQ.cpp —
ver esse arquivo para o porquê de cada escolha (pivot_floor, saturação em
[0,1;10] no ASDA, a forma V/W separada do ADDA etc.).
"""

import numpy as np

Q_SHIFT_DEFAULT = 18
INT32_MAX = 2**31 - 1
INT32_MIN = -2**31


class Status:
    def __init__(self):
        self.overflow = False
        self.iterations = 0
        self.max_abs_seen = 0
        self.rel_step = 1.0        # espelha Status::rel_step do C++
        self.bit_exact_zero = False


def f2q(x, sh, st):
    v = float(x) * float(1 << sh)
    if v > 2147483647.0:
        st.overflow = True
        return INT32_MAX
    if v < -2147483648.0:
        st.overflow = True
        return INT32_MIN
    return int(round(v))


def q2f(x, sh):
    return float(x) / float(1 << sh)


def qmul(a, b, sh, st):
    r = (a * b) >> sh
    if r > INT32_MAX:
        st.overflow = True
        return INT32_MAX
    if r < INT32_MIN:
        st.overflow = True
        return INT32_MIN
    return r


def qdiv(a, b, sh, st):
    if b == 0:
        st.overflow = True
        return INT32_MAX if a >= 0 else INT32_MIN
    # divisao truncada em direcao a zero, igual ao C++ (int64_t / int64_t) —
    # o // do Python trunca para -infinito, por isso o _trunc_div dedicado.
    num = a << sh
    r = _trunc_div(num, b)
    if r > INT32_MAX:
        st.overflow = True
        return INT32_MAX
    if r < INT32_MIN:
        st.overflow = True
        return INT32_MIN
    return r


def _trunc_div(num, den):
    q = abs(num) // abs(den)
    if (num < 0) != (den < 0):
        q = -q
    return q


def matmul_q(a, b, r1, c1, c2, sh, st):
    """a: lista/array (r1*c1), b: (c1*c2) -> retorna lista (r1*c2)."""
    c = [0] * (r1 * c2)
    for i in range(r1):
        for j in range(c2):
            acc = 0
            for k in range(c1):
                acc += a[i * c1 + k] * b[k * c2 + j]
            r = acc >> sh
            if r > INT32_MAX:
                st.overflow = True
                r = INT32_MAX
            elif r < INT32_MIN:
                st.overflow = True
                r = INT32_MIN
            c[i * c2 + j] = r
            av = abs(r)
            if av > st.max_abs_seen:
                st.max_abs_seen = av
    return c


def transpose_q(a, r, c):
    at = [0] * (r * c)
    for i in range(r):
        for j in range(c):
            at[j * r + i] = a[i * c + j]
    return at


def add_q(a, b):
    return [x + y for x, y in zip(a, b)]


def sub_q(a, b):
    return [x - y for x, y in zip(a, b)]


def invert_q(src, n, sh, st):
    """Gauss-Jordan com pivotamento parcial em Q-format — espelho de
    invert_q() em FixedPointQ.cpp. Retorna (dst, ok)."""
    n2 = 2 * n
    one = f2q(1.0, sh, st)
    pivot_floor = f2q(1e-4, sh, st)
    aug = [0] * (n * n2)
    for i in range(n):
        for j in range(n):
            aug[i * n2 + j] = src[i * n + j]
        for j in range(n):
            aug[i * n2 + n + j] = one if i == j else 0

    for i in range(n):
        maxv = abs(aug[i * n2 + i])
        mr = i
        for k in range(i + 1, n):
            v = abs(aug[k * n2 + i])
            if v > maxv:
                maxv, mr = v, k
        if maxv < pivot_floor:
            return None, False
        if mr != i:
            for j in range(n2):
                aug[i * n2 + j], aug[mr * n2 + j] = aug[mr * n2 + j], aug[i * n2 + j]
        # Reciproco do pivo calculado UMA vez e multiplicado nas n2 colunas --
        # espelha FixedPointQ.cpp:61-63 (o Xtensa LX7 nao tem divisao de 64
        # bits em hardware, entao o C++ troca n2 divisoes por 1 divisao + n2
        # multiplicacoes). MUDA o arredondamento em relacao a qdiv direto, por
        # isso tem de ser replicado aqui para o emulador continuar fiel.
        piv = aug[i * n2 + i]
        inv_piv = qdiv(one, piv, sh, st)
        for j in range(n2):
            aug[i * n2 + j] = qmul(aug[i * n2 + j], inv_piv, sh, st)
        for k in range(n):
            if k != i:
                f = aug[k * n2 + i]
                if f != 0:
                    for j in range(n2):
                        aug[k * n2 + j] -= qmul(f, aug[i * n2 + j], sh, st)

    dst = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            dst[i * n + j] = aug[i * n2 + n + j]
    return dst, True


VARIANT_STANDARD = "standard"
VARIANT_ADAPTIVE_SCALING = "adaptive_scaling"
VARIANT_ALTERNATING_VW = "alternating_vw"


def doubling_loop_q(Ak, Gk, Hk, n, sh, variant, max_iterations, inv_rel_tolerance, st):
    """Espelho de doubling_loop_q() em FixedPointQ.cpp. Ak,Gk,Hk: listas
    planas n*n (inteiros Q-format), modificadas e retornadas (não in-place,
    ao contrário do C++, para evitar armadilhas de aliasing em Python).
    Retorna (Ak, Gk, Hk, cum_s, ok)."""
    nn = n * n
    one = f2q(1.0, sh, st)
    cum_s = 1.0
    st.iterations = max_iterations

    Ak = list(Ak)
    Gk = list(Gk)
    Hk = list(Hk)

    # somas da ultima iteracao executada, para o passo relativo sair uma vez
    ult_diff_sq, ult_h_sq, ult_saturou = 0, 0, False

    for it in range(max_iterations):
        if variant == VARIANT_ADAPTIVE_SCALING:
            sumG = sum(g * g for g in Gk)
            sumH = sum(h * h for h in Hk)
            scale = 1.0 / float(1 << sh)
            normG = (sumG ** 0.5) * scale
            normH = (sumH ** 0.5) * scale
            s_f = 1.0
            if normG > 1e-10 and normH > 1e-10:
                s_f = (normH / normG) ** 0.5
                s_f = min(max(s_f, 0.1), 10.0)
            s_q = f2q(s_f, sh, st)
            Gk = [qmul(g, s_q, sh, st) for g in Gk]
            Hk = [qdiv(h, s_q, sh, st) for h in Hk]
            cum_s *= s_f

        if variant == VARIANT_ALTERNATING_VW:
            T1 = matmul_q(Gk, Hk, n, n, n, sh, st)
            for i in range(n):
                T1[i * n + i] += one
            V, ok = invert_q(T1, n, sh, st)
            if not ok:
                return Ak, Gk, Hk, cum_s, False

            T1b = matmul_q(Hk, Gk, n, n, n, sh, st)
            for i in range(n):
                T1b[i * n + i] += one
            W, ok = invert_q(T1b, n, sh, st)
            if not ok:
                return Ak, Gk, Hk, cum_s, False

            AV = matmul_q(Ak, V, n, n, n, sh, st)
            Akn = matmul_q(AV, Ak, n, n, n, sh, st)

            AT = transpose_q(Ak, n, n)
            GAT = matmul_q(Gk, AT, n, n, n, sh, st)
            AVGAT = matmul_q(AV, GAT, n, n, n, sh, st)
            Gkn = add_q(Gk, AVGAT)

            WH = matmul_q(W, Hk, n, n, n, sh, st)
            WHA = matmul_q(WH, Ak, n, n, n, sh, st)
            ATWHA = matmul_q(AT, WHA, n, n, n, sh, st)
            Hkn = add_q(Hk, ATWHA)
        else:
            T1 = matmul_q(Gk, Hk, n, n, n, sh, st)
            for i in range(n):
                T1[i * n + i] += one
            V, ok = invert_q(T1, n, sh, st)
            if not ok:
                return Ak, Gk, Hk, cum_s, False

            AV = matmul_q(Ak, V, n, n, n, sh, st)
            Akn = matmul_q(AV, Ak, n, n, n, sh, st)

            AT = transpose_q(Ak, n, n)
            GAT = matmul_q(Gk, AT, n, n, n, sh, st)
            AVGAT = matmul_q(AV, GAT, n, n, n, sh, st)
            Gkn = add_q(Gk, AVGAT)

            VA = matmul_q(V, Ak, n, n, n, sh, st)
            HVA = matmul_q(Hk, VA, n, n, n, sh, st)
            ATHVA = matmul_q(AT, HVA, n, n, n, sh, st)
            Hkn = add_q(Hk, ATHVA)

        # Frobenius relativa, igual ao C++ (FixedPointQ.cpp:doubling_loop_q).
        # O CRITERIO nao mudou; a conta sim: ‖ΔH‖²·τ⁻² < ‖H‖² em inteiro, sem
        # conversao, raiz nem divisao. O fator de escala 2^(2s) cancela na razao.
        # ESTE ESPELHO TEM DE SEGUIR O C++ EXATAMENTE: malha_fechada_trajetorias.py
        # e bench_trajetorias.py simulam no host o que a placa executa, e qualquer
        # divergencia aqui vira divergencia entre simulacao e firmware.
        #
        # Python tem inteiro de precisao arbitraria, entao nao ha' o teto de
        # 2^61 do int64 do C++ -- mas o `saturou` e' replicado assim mesmo, para
        # que as duas implementacoes tomem a MESMA decisao na regiao de quebra
        # que o mapa de seguranca varre.
        LIM = 1 << 61
        diff_sq = 0
        h_sq = 0
        bit_exact = True
        saturou = False
        for i in range(nn):
            d = Hkn[i] - Hk[i]
            if d != 0:
                bit_exact = False
            if saturou:
                continue
            d2 = d * d
            h2 = Hk[i] * Hk[i]
            if d2 > LIM - diff_sq or h2 > LIM - h_sq:
                saturou = True
                continue
            diff_sq += d2
            h_sq += h2

        Ak, Gk, Hk = Akn, Gkn, Hkn
        st.bit_exact_zero = bit_exact
        ult_diff_sq, ult_h_sq, ult_saturou = diff_sq, h_sq, saturou

        convergiu = False
        if not saturou:
            inv_tol2 = inv_rel_tolerance * inv_rel_tolerance
            lhs = diff_sq * inv_tol2
            # o C++ recusa por overflow do int64 o que aqui nunca estoura; a
            # condicao equivalente e' o produto passar do que um int64 guarda.
            convergiu = lhs <= 0x7FFFFFFFFFFFFFFF and lhs < h_sq
        if convergiu:
            st.iterations = it + 1
            break

    # passo relativo: instrumentacao, convertido uma vez na parada (igual ao C++)
    st.rel_step = ((ult_diff_sq / ult_h_sq) ** 0.5
                   if (not ult_saturou and ult_h_sq > 0) else 1.0)

    return Ak, Gk, Hk, cum_s, True


# ---------------------------------------------------------------------------
# Conveniência: matriz numpy <-> lista plana Q-format
# ---------------------------------------------------------------------------
def mat_to_q(M, sh, st):
    return [f2q(v, sh, st) for v in np.asarray(M, dtype=float).flatten()]


def q_to_mat(vals, n, m, sh):
    return np.array([q2f(v, sh) for v in vals], dtype=float).reshape(n, m)


if __name__ == "__main__":
    # Auto-teste rápido: qmul/qdiv/invert_q em uma matriz 3x3 bem condicionada,
    # conferindo contra numpy.linalg.inv dentro da resolução do Q13.18.
    st = Status()
    sh = Q_SHIFT_DEFAULT
    M = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    Mq = mat_to_q(M, sh, st)
    dst, ok = invert_q(Mq, 3, sh, st)
    Minv_q = q_to_mat(dst, 3, 3, sh)
    Minv_ref = np.linalg.inv(M)
    erro = np.max(np.abs(Minv_q - Minv_ref))
    print("invert_q ok=%s overflow=%s erro_max=%.3e (resolucao Q13.18=%.3e)" % (
        ok, st.overflow, erro, 2.0 ** -18))
    assert ok and not st.overflow and erro < 1e-3
    print("auto-teste OK")
