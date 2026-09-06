# -*- coding: utf-8 -*-
"""Общий буфер ячеек: части ноутбука просто вызывают md()/code() на уровне модуля."""
import nbformat as nbf

CELLS = []


def md(s):
    CELLS.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(nbf.v4.new_code_cell(s.strip("\n")))
