#!/usr/bin/env python3
"""K1（低速介面放寬決策資料）步驟 2a：從 run_mission_roundtrip2.sh 實際啟動的入口檔出發，遞迴追蹤本地匯入
（evaluation/*.py 與 ammr_wholebody_mpc 套件），列出所有出現速度框數值的位置。只讀不改。
輸出：每處「檔案:行、數值、是否為命令列預設（argparse）或模組常數／函式預設」。
    python3 evaluation/k1_box_inventory.py [--out evaluation/results/k1/box_inventory.json]
"""
import argparse
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
PKG = os.path.join(WS, 'src', 'ammr_wholebody_mpc', 'ammr_wholebody_mpc')
RUNNER = os.path.join(HERE, 'run_mission_roundtrip2.sh')
# 速度框相關數值：求解框（逐軸）、執行端合速度界、權威世界合速度、手臂速、引導／交棒上限
VALS = {'0.035255': 'v_box_lin', '0.0352': 'v_box_lin', '0.1999': 'v_box_ang', '0.199900': 'v_box_ang',
        '0.9999': 'v_arm', '0.999900': 'v_arm'}
NAME_RE = re.compile(r'(v[-_]?box|v[-_]lin|v[-_]ang|v[-_]world|v[-_]cap|w[-_]cap|base[-_]lin[-_]max|base[-_]ang[-_]max|arm[-_]rate[-_]max|ready[-_][vw]|v[-_]roll|slow[-_]v|v[-_]max|v[-_]base|w[-_]max|qd[-_]max)', re.I)


def entry_files():
    txt = open(RUNNER, encoding='utf-8').read()
    out = set()
    for ln in txt.splitlines():
        if not re.match(r'\s*(spawn|.*\bpython3 -u)', ln) or ln.lstrip().startswith('#'):
            continue
        for m in re.finditer(r'evaluation/([A-Za-z0-9_]+\.py)', ln):
            out.add(os.path.join(HERE, m.group(1)))
    return sorted(out)


def local_imports(path):
    try:
        tree = ast.parse(open(path, encoding='utf-8').read())
    except (SyntaxError, UnicodeDecodeError):
        return []
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module)
            mods += [f'{n.module}.{a.name}' for a in n.names]
    res = []
    for m in mods:
        if m.startswith('ammr_wholebody_mpc.'):
            f = os.path.join(PKG, m.split('.', 1)[1].replace('.', '/') + '.py')
        else:
            f = os.path.join(HERE, m.replace('.', '/') + '.py')
        if os.path.isfile(f):
            res.append(f)
    return res


def scan(path):
    hits = []
    src = open(path, encoding='utf-8').read().splitlines()
    for i, ln in enumerate(src, 1):
        code = ln.split('#', 1)[0]
        if not code.strip():
            continue
        for v, lab in VALS.items():
            if re.search(r'(?<![\d.])' + re.escape(v) + r'(?!\d)', code):
                kind = ('argparse' if 'add_argument' in code or 'declare_parameter' in code
                        else 'doc' if '=' not in code and re.search(r'[\u4e00-\u9fff]', code) else 'const/default')
                hits.append({'line': i, 'val': v, 'what': lab, 'kind': kind, 'code': code.strip()[:160]})
        if 'add_argument' in code and NAME_RE.search(code) and not any(h['line'] == i for h in hits):
            m = re.search(r"default\s*=\s*([0-9.]+)", code)
            if m:
                hits.append({'line': i, 'val': m.group(1), 'what': 'named_cli', 'kind': 'argparse', 'code': code.strip()[:160]})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'k1', 'box_inventory.json'))
    a = ap.parse_args()
    todo, seen, parent = list(entry_files()), set(), {}
    for f in todo:
        parent[f] = 'runner'
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for g in local_imports(f):
            if g not in seen:
                parent.setdefault(g, f)
                todo.append(g)
    rows = []
    for f in sorted(seen):
        for h in scan(f):
            rows.append({'file': os.path.relpath(f, WS), 'via': os.path.relpath(parent[f], WS) if parent[f] != 'runner' else 'runner', **h})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({'runner': os.path.relpath(RUNNER, WS), 'n_files': len(seen), 'rows': rows}, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'追蹤 {len(seen)} 個本地模組；命中 {len(rows)} 處 → {os.path.relpath(a.out, WS)}')
    for r in rows:
        print(f"{r['file']}:{r['line']}  {r['val']:>9} {r['what']:<10} {r['kind']:<13} {r['code'][:110]}")


if __name__ == '__main__':
    sys.exit(main())
