# -*- coding: utf-8 -*-
"""Собирает research/risk.ipynb из частей в nbparts/.

Запуск из папки research/build:  python build_notebook.py
Готовый ноутбук кладётся на уровень выше, рядом с results.json.
"""
import importlib
import nbformat as nbf

from nbparts import _h

PARTS = [
    "p01_intro",      # 0-1  загрузка, целевая, G3==0
    "p02_eda",        # 2-3  EDA + утечки
    "p03_features",   # 4    признаки и инженерия
    "p04_models",     # 5    модели и протокол
    "p05_levels",     # 6-7  уровень 1 и уровень 2
    "p06_threshold",  # 8    порог
    "p07_factors",    # 9    три фактора
    "p08_transfer",   # 10   перенос на математику
    "p09_research",   # 11   собственные исследования
    "p10_final",      # 12-13 итог, этика, что не сработало
]

for name in PARTS:
    importlib.import_module(f"nbparts.{name}")

nb = nbf.v4.new_notebook(cells=_h.CELLS)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
nbf.write(nb, "../risk.ipynb")
print(f"risk.ipynb собран: {len(_h.CELLS)} ячеек "
      f"({sum(c.cell_type=='code' for c in _h.CELLS)} кода, "
      f"{sum(c.cell_type=='markdown' for c in _h.CELLS)} markdown)")
