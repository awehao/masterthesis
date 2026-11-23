#!/usr/bin/env python3
"""isaac_door_k2_sim_ev9r7 相對 ev9r6 的靜態核對（不匯入、不執行執行端；Codex r48 §2）：
  D1 程式碼（去掉註解與字串後）只有一處呼叫 get_contact_force_data，位於 _snap.take(...)；
  D2 舊分類 leaf_contacts、EV 取證 extract_contacts、雙把手旁路 step_record 都改取 _snap.get(step_id)；
  D3 被刪的行只來自三段：舊分類讀取＋迴圈本體（搬到 contact_snapshot.old_leaf_contacts）、取證的讀取行、旁路的讀取／寫入行；
  D4 搬走的迴圈與 contact_snapshot.old_leaf_contacts 逐行相同（縮排與變數名對應）；
  D5 兩檔可編譯。

    python3 evaluation/test_ev9r7_dataflow.py
"""
import difflib
import io
import os
import py_compile
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
A_txt = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r6.py')).read()
B_txt = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r7.py')).read()
fails = []


def code_only(txt):
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(txt).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return ' '.join(out)


c = code_only(B_txt)
calls = [m.start() for m in re.finditer(r'get_contact_force_data \(', c)]
if len(calls) != 1:
    fails.append(f'D1 程式碼中呼叫 {len(calls)} 處')
elif '_snap . take' not in c[max(0, calls[0] - 120):calls[0]]:
    fails.append('D1 唯一呼叫不在 _snap.take 內')
for pat, what in ((r'CS \. old_leaf_contacts \( _snap \. get \( step_id \)', '舊分類'), (r'KEV \. extract_contacts \( \* \( _snap \. get \( step_id \) \)', '取證'),
                  (r'DCA \. step_record \( lambda : _snap \. get \( step_id \)', '旁路')):
    if not re.search(pat, c):
        fails.append(f'D2 {what} 未改取快照')
A, B = A_txt.splitlines(), B_txt.splitlines()
removed = []
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=A, b=B, autojunk=False).get_opcodes():
    if tag in ('replace', 'delete'):
        removed += A[i1:i2]
i0 = A.index('    def leaf_contacts(T_WH):')
body = A[i0 + 2:i0 + 15]                                    # try: … return out
allowed = set(body) | {
    "        cd, why = KEV.safe(lambda: KEV.extract_contacts(*(leaf_pair_v.get_contact_force_data(dt=a.physics_dt)), pair_names))",
    "                _c3_r = DCA.step_record(lambda: leaf_pair_v.get_contact_force_data(dt=a.physics_dt), lambda: _qd_rec, _door_T, dspec,",
    "                _c3_w.write(_c3_r)", "                _c3_sum.add(_c3_r)", "            except Exception:                              # 旁路不得中斷模擬",
    "                _c3_w.count_err('record')",
    "        rec['contact_v3'] = {'writer': _c3_w.close(), 'summary': _c3_sum.as_dict(), 'spec_id': _c3_spec_id, 'tol_m': a.bar_class_tol,"}
extra = [ln for ln in removed if ln not in allowed]
if extra:
    fails.append(f'D3 非預期刪除 {extra[:5]}')
# D4：搬走的迴圈 vs contact_snapshot.old_leaf_contacts
cs = open(os.path.join(HERE, 'contact_snapshot.py')).read().splitlines()
k0 = cs.index("    fn, pts, nrm, dist, cnt, start = (_np(v) for v in data)")
moved = [ln.strip() for ln in cs[k0 + 1:k0 + 11]]
orig = [ln.strip() for ln in body[2:12]]
orig = [ln.replace('pair_names[jb]', 'pair_names[jb]').replace('a.bar_class_tol', 'tol') for ln in orig]
if moved != orig:
    fails.append(f'D4 迴圈不一致：{[(x, y) for x, y in zip(orig, moved) if x != y][:3]}')
for f in ('isaac_door_k2_sim_ev9r6.py', 'isaac_door_k2_sim_ev9r7.py'):
    try:
        py_compile.compile(os.path.join(HERE, f), doraise=True)
    except py_compile.PyCompileError as e:
        fails.append(f'D5 {f} {e}')
print('全部通過' if not fails else f'失敗 {len(fails)}', f'（刪 {len(removed)} 行）')
for f in fails:
    print('  ', f)
sys.exit(1 if fails else 0)
