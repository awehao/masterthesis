#!/usr/bin/env python3
"""M 組剩餘調參預算（Codex 裁定）：固定 kp 4、mu_post 0.01，t_insert 依序 1.75、2.0、2.5、3.0 s，第一個完整通過即凍結，不挑最好。失敗全部保留。假受控對象。
    python3 evaluation/bottle_s1_m_tinsert.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_groups as GR                           # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402


def main():
    out, frozen = [], None
    for ti in (1.75, 2.0, 2.5, 3.0):
        c = SC.make_lidar(cls=GR.S1MotM, kp=4.0, t_insert=ti)
        c.p1.mu_post = 0.01
        f = CK.fake(c, 48.0)
        r = CK.summary(f'M t_insert {ti}', c, f)
        cb = r['clear_bottle']
        ok = bool(r['success'] and S1.exit_code(r) == 0)
        out.append(dict(t_insert=ti, ok=ok, stop=(r['stop'] or {}).get('why'), meas=cb['meas_min'], pred=cb['pred_min'], next=cb.get('next_min'), t_end=f.t))
        print(f"  ⇒ 通過 {ok}；下一拍 {None if cb.get('next_min') is None else round(cb['next_min'] * 1e3, 2)} mm", flush=True)
        if ok:
            frozen = ti
            break
    print('凍結 t_insert', frozen)
    json.dump(dict(kp=4.0, mu_post=0.01, order=[1.75, 2.0, 2.5, 3.0], frozen=frozen, trials=out), open(os.path.join(HERE, 'results', 'bottle', 's1_m_tinsert.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
