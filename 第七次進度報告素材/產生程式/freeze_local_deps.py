#!/usr/bin/env python3
"""以 AST 遞迴解析入口程式的**本地** import（evaluation/ 與 src/ammr_wholebody_mpc/ammr_wholebody_mpc/），列出相依檔（不執行、不啟動 Isaac）。

    python3 evaluation/freeze_local_deps.py <entry.py> [...]     # 印出工作區相對路徑（排序、去重）
"""
import ast
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = [os.path.join(WS, 'evaluation'), os.path.join(WS, 'src', 'ammr_wholebody_mpc')]


def resolve(name):
    parts = name.split('.')
    for r in ROOTS:
        p = os.path.join(r, *parts) + '.py'
        if os.path.isfile(p):
            return p
        p = os.path.join(r, *parts, '__init__.py')
        if os.path.isfile(p):
            return p
    return None


def deps(entry, seen):
    if entry in seen:
        return
    seen.add(entry)
    tree = ast.parse(open(entry, encoding='utf-8').read())
    for n in ast.walk(tree):
        names = []
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names = [n.module] + [f'{n.module}.{a.name}' for a in n.names]
        for nm in names:
            p = resolve(nm)
            if p:
                deps(p, seen)


def main():
    seen = set()
    for e in sys.argv[1:]:
        deps(os.path.abspath(e), seen)
    for p in sorted(os.path.relpath(x, WS) for x in seen):
        print(p)


if __name__ == '__main__':
    main()
