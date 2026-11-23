#!/usr/bin/env python3
"""XH2 擷取通路複本的靜態測試（不開 Isaac）：凍結原檔未改；複本差異只在無橫桿支援；N 不寫 handle_center；R／N 二擇一。

    python3 evaluation/test_xh_capture_copies.py
"""
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def diff(a, b):
    out = subprocess.run(['diff', os.path.join(HERE, a), os.path.join(HERE, b)], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.startswith('< ')], [l for l in out.splitlines() if l.startswith('> ')]


# 凍結原檔與 S4／DL0 凍結清單一致
for fz in ('freeze_d1_s4_r5.sha256', 'freeze_dl0_capture.sha256'):
    r = subprocess.run(['sha256sum', '-c', '--quiet', os.path.join('evaluation', 'results', 'vision', fz)], cwd=WS, capture_output=True, text=True)
    bad = [l for l in r.stdout.splitlines() if ('isaac_drawer_room_sim.py' in l or 'wrist_v0_capture.py' in l
                                                 or 'run_wrist_v0.sh' in l or 'drawer_asset.py' in l or 'dl0_paths.py' in l)]
    check(f'frozen_capture_files_unchanged_{fz}', not bad, bad)
rem, add = diff('isaac_drawer_room_sim.py', 'isaac_drawer_room_sim_xh.py')
check('sim_removed_lines_limited', len(rem) == 6 and any('import drawer_asset as DA' in l for l in rem)
      and any('from wrist_v0_capture import run_wrist_v0' in l for l in rem), rem)
src = open(os.path.join(HERE, 'isaac_drawer_room_sim_xh.py')).read()
check('sim_no_bar_only_in_wrist_v0', "not a.wrist_v0" in src and 'return 22' in src and '不建假橫桿' in src)
cap = open(os.path.join(HERE, 'wrist_xh_capture.py')).read()
tree = ast.parse(cap)
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run_wrist_v0')
check('capture_signature_has_kprim', any(a.arg == 'kprim' for a in fn.args.args + fn.args.kwonlyargs))
check('capture_exclusive_hprim_kprim', "if (hprim is None) == (kprim is None):" in cap)
check('capture_n_has_no_handle_center', "{'handle_center_world_at_capture': hp['handle_center']} if 'handle_center' in hp else" in cap
      and "'observation_ref_center_world_at_capture'" in cap)
check('capture_truth_target_kind', "'target_kind': 'R_bar'" in cap and "'target_kind': 'none" in cap)
rr, ra = diff('run_wrist_v0.sh', 'run_wrist_xh.sh')
check('runner_diff_limited', len(rr) == 2 and len(ra) == 4, (rr, ra))
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
