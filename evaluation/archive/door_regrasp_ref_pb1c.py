#!/usr/bin/env python3
"""ref pb1c（Codex 20261028 pb1_6 回覆「先建立實際退離，再張爪」）：ref pb1b 只改 segments['3'] 的時間表與 qf：
  - 時間整段乘 SCALE＝21.3/28（構型路徑不變；退離巡航 ≈ 28 mm/s，使實測可達觸發門檻 25 mm/s）；縮放後逐點核：底盤本體每軸 ≤ 0.035255 m/s、
    偏航 ≤ 0.1999 rad/s、手臂每軸 ≤ 0.9999 rad/s、底盤線加速度 ≤ 0.5 m/s²、手臂角加速度 ≤ 19.984 rad/s²（差分估；超限即拒建）；
  - qf（只供監看預測；實際張爪由權威 trigger_open 決定）：參考退離速度首次 ≥ 25 mm/s 之前維持起點值（閉爪），之後沿用原張開曲線（原 0.9 s 由閉到開）。
其餘段、欄位逐字同 pb1b；登錄自驗須通過。

    python3 evaluation/door_regrasp_ref_pb1c.py   # 寫 results/door/door_regrasp_ref_pb1c.json（不覆寫）
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_regrasp_ref_registry as REG                # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1b.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1c.json')
SCALE = 21.3 / 28.0
V_TRIG = 0.025
LIM = {'vb': 0.035255, 'w': 0.1999, 'qd': 0.9999, 'ab': 0.5, 'qdd': 19.984}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    s3 = p['segments']['3']
    t = np.array(s3['t']) * SCALE
    B, A, S = np.array(s3['base']), np.array(s3['arm']), np.array(s3['s'])
    dt = np.diff(t)
    vb = np.diff(B, axis=0) / dt[:, None]
    th = B[:-1, 2]
    vbx, vby = np.cos(th) * vb[:, 0] + np.sin(th) * vb[:, 1], -np.sin(th) * vb[:, 0] + np.cos(th) * vb[:, 1]
    qd = np.diff(A, axis=0) / dt[:, None]
    ab = np.diff(np.c_[vbx, vby], axis=0) / dt[1:, None]
    qdd = np.diff(qd, axis=0) / dt[1:, None]
    chk = {'vb': float(max(np.abs(vbx).max(), np.abs(vby).max())), 'w': float(np.abs(vb[:, 2]).max()), 'qd': float(np.abs(qd).max()),
           'ab': float(np.abs(ab).max()), 'qdd': float(np.abs(qdd).max())}
    bad = {k: (v, LIM[k]) for k, v in chk.items() if v > LIM[k]}
    if bad:
        raise SystemExit(f'縮放後超限：{bad}')
    vs = np.r_[0.0, np.diff(S) / dt]
    i_trig = int(np.argmax(vs >= V_TRIG))
    qf_old = list(s3['qf'])
    qf = [qf_old[0]] * i_trig + qf_old[:len(qf_old) - i_trig]
    new = dict(s3, t=t.tolist(), qf=qf)
    p['segments']['3'] = new
    p['schema'] = p.get('schema', '') + '+pb1c'
    p['pb1c'] = {'derived_from': {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()},
                 'scale': SCALE, 'limits_check': chk, 'limits': LIM, 'ref_v_max_mps': float(vs.max()), 'i_trig': i_trig, 't_trig_ref_s': float(t[i_trig]),
                 's_trig_ref_m': float(S[i_trig]), 'seg3_T_new': float(t[-1]),
                 'note': 'mode 3 先閉爪退離、實測 ≥ 25 mm/s 才由權威張爪（Codex 20261028）；qf 只供預測'}
    ok, why = REG.validate(json.loads(json.dumps(p, default=float)))
    p['registry_validate'] = {'ok': ok, 'why': why}
    if not ok:
        raise SystemExit(f'登錄自驗不過：{why}')
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(p['pb1c'], ensure_ascii=False, default=float)[:600])


if __name__ == '__main__':
    main()
