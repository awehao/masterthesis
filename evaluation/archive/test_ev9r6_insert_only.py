#!/usr/bin/env python3
"""isaac_door_k2_sim_ev9r6 相對 ev9r5 的靜態核對（不匯入、不執行執行端）：
  S1 刪除行只有兩行（門規格載入與建構改經 _DA；--contact-v3-out 關閉時 _DA＝DA ⇒ 等價）；
  S2 新增的執行路徑都在 a.contact_v3_out／_c3_w 的保護內（關閉時不讀新 API、不寫檔）；模組層新增只有兩個匯入與 argparse 一項；
  S3 新退出碼 63 在 ev9r5 未被使用；
  S4 兩檔皆可編譯。

    python3 evaluation/test_ev9r6_insert_only.py
"""
import difflib
import os
import py_compile
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r5.py')).read().splitlines()
B = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r6.py')).read().splitlines()
fails = []
removed, added_blocks = [], []
sm = difflib.SequenceMatcher(a=A, b=B, autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag in ('replace', 'delete'):
        removed += A[i1:i2]
    if tag in ('replace', 'insert'):
        added_blocks.append(B[j1:j2])
exp_removed = ['    dspec = DA.load(a.door_asset)', '    dauth = DA.build_usd(stage, dspec, root=DOOR_ROOT)']
if removed != exp_removed:
    fails.append(f'S1 刪除行 {removed}')
guards = ('a.contact_v3_out', '_c3_w', 'EV9r6', '_DA', 'DA3', 'DCA', '_door_root_prim', '_door_T', '_c3_')
for blk in added_blocks:
    code = [ln for ln in blk if ln.strip() and not ln.lstrip().startswith('#')]
    if code and not any(g in '\n'.join(blk) for g in guards):
        fails.append(f'S2 未受保護的新增：{code[:3]}')
mod_level = [ln for blk in added_blocks for ln in blk if ln and not ln.startswith((' ', '#'))]
exp_mod = {"import door_asset_v3 as DA3", "import door_contact_v3_adapter as DCA", "ap.add_argument('--contact-v3-out', default='',", "                help="}
if not all(any(ln.startswith(e.strip()) for e in exp_mod) for ln in mod_level):
    fails.append(f'S2 模組層新增 {mod_level}')
if re.search(r'return 63\b', '\n'.join(A)):
    fails.append('S3 退出碼 63 在 ev9r5 已使用')
for f in ('isaac_door_k2_sim_ev9r5.py', 'isaac_door_k2_sim_ev9r6.py'):
    try:
        py_compile.compile(os.path.join(HERE, f), doraise=True)
    except py_compile.PyCompileError as e:
        fails.append(f'S4 {f} 編譯失敗 {e}')
print('全部通過' if not fails else f'失敗 {len(fails)}', f'（刪 {len(removed)} 行、新增 {sum(len(b) for b in added_blocks)} 行／{len(added_blocks)} 段）')
for f in fails:
    print('  ', f)
sys.exit(1 if fails else 0)
