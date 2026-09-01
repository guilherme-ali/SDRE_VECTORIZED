"""
Extrai as anotações (highlights/strikeouts + notas) de um PDF comentado via
PyMuPDF, escrevendo trecho destacado + nota de cada uma, na ordem do documento.

Uso: python extrai_anotacoes_pdf.py <caminho_pdf> [saida.txt]

Criado para consolidar os comentários do Reginaldo em
G:\\Meu Drive\\ACADEMICO\\Mestrado\\EVENTOS\\DINAME_2027\\CBA_artigo_comentado.pdf
(ver revisoes_consolidadas.md na mesma pasta).
"""

import io
import sys

import fitz  # PyMuPDF


def extrair_anotacoes(pdf_path, saida_path=None):
    doc = fitz.open(pdf_path)
    out = io.open(saida_path, "w", encoding="utf-8") if saida_path else sys.stdout
    n = 0
    for pno, page in enumerate(doc, 1):
        for a in page.annots() or []:
            n += 1
            info = a.info
            tipo = a.type[1]
            trecho = ""
            if tipo in ("Highlight", "Underline", "StrikeOut", "Squiggly"):
                try:
                    qp = a.vertices
                    quads = [fitz.Quad(qp[i:i + 4]) for i in range(0, len(qp), 4)]
                    trecho = " ".join(
                        page.get_textbox(q.rect).replace("\n", " ") for q in quads
                    )
                except Exception:
                    trecho = "<erro ao extrair quad>"
            out.write("=== #%d p%d [%s]\n" % (n, pno, tipo))
            if trecho.strip():
                out.write("   TRECHO: %s\n" % trecho.strip())
            if info.get("content", "").strip():
                out.write("   NOTA: %s\n" % info["content"].strip())
    if saida_path:
        out.close()
    print("total de anotacoes: %d" % n, file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pdf = sys.argv[1]
    saida = sys.argv[2] if len(sys.argv) > 2 else None
    extrair_anotacoes(pdf, saida)
