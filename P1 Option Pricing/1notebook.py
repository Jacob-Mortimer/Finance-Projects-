# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 19:12:34 2025

@author: Jacon
"""

# py_to_notebook.py
import sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cells = []
    comment_block = []
    code_block = []

    def flush_comment_block():
        nonlocal comment_block
        if comment_block:
            # strip leading '#', support inline '# ' style
            md_lines = [ln.lstrip('#').strip() for ln in comment_block]
            # join, keep blank lines
            md_text = '\n'.join(md_lines)
            cells.append(new_markdown_cell(md_text))
            comment_block = []

    def flush_code_block():
        nonlocal code_block
        if code_block:
            code_text = ''.join(code_block)
            cells.append(new_code_cell(code_text))
            code_block = []

    for ln in lines:
        if ln.strip().startswith('#'):
            # comment line -> markdown
            # flush any queued code first
            if code_block:
                flush_code_block()
            comment_block.append(ln)
        else:
            # code line -> keep code
            if comment_block:
                flush_comment_block()
            code_block.append(ln)

    # flush leftovers
    flush_code_block()
    flush_comment_block()

    nb = new_notebook(cells=cells)
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python py_to_notebook.py input.py output.ipynb")
    else:
        py_to_ipynb(sys.argv[1], sys.argv[2])
        print("Wrote", sys.argv[2])
