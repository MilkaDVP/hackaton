# -*- coding: utf-8 -*-
"""Показывает ошибки и (опционально) весь текстовый вывод выполненного ноутбука."""
import sys
import nbformat

path = sys.argv[1] if len(sys.argv) > 1 else "solution_exec.ipynb"
show_all = "--all" in sys.argv
only = [a for a in sys.argv[2:] if a.isdigit()]

nb = nbformat.read(path, as_version=4)
n_err = 0
idx = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    idx += 1
    errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errs:
        n_err += 1
        print("=" * 78)
        print(f"ОШИБКА в code-ячейке #{idx} (индекс {i})")
        print("-" * 78)
        print(c.source[:1200])
        print("-" * 78)
        for e in errs:
            print("\n".join(e.get("traceback", []))[-2500:])
        print()
    elif show_all or (only and str(idx) in only):
        texts = []
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                texts.append(o.get("text", ""))
            elif o.get("output_type") == "execute_result":
                texts.append(o.get("data", {}).get("text/plain", ""))
        if texts:
            print(f"--- ячейка #{idx} ---")
            print("".join(texts)[:4000])
            print()

print("=" * 78)
print(f"ВСЕГО ОШИБОК: {n_err} из {idx} code-ячеек")
