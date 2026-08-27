"""Conta palavras por secao de um .tex, para orcamento de paginas."""
import re
import sys

for f in sys.argv[1:]:
    t = open(f, encoding="utf-8").read()
    i = t.find(r"\begin{document}")
    body = t[i:]
    body = re.sub(r"(?m)^%.*$", "", body)
    secs = re.split(r"\\(?:sub)?section\*\{([^}]*)\}", body)
    print("==", f, "| palavras no corpo ~", len(body.split()))
    it = iter(secs[1:])
    for name, content in zip(it, it):
        print("   %-42s %5d" % (name[:42], len(content.split())))
    # abstract
    m = re.search(r"\\abstract\{(.*?)\n\}", t, re.S)
    if m:
        print("   [abstract]                                 %5d" % len(m.group(1).split()))
    print()
