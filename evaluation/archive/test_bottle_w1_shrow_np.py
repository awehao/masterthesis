#!/usr/bin/env python3
"""同一 SHROW 堆疊「僅切換障礙預測」的最小離線核對（假受控對象，非物理）。
1 名目（無移動物）：SHROW 與 SHROW 瞬時組的底盤＋手臂命令逐步比較；有差異時列出首次不同時刻與當時動態軌跡數（追蹤器假動態）。
2 第一案、3 第二案：兩組結果並列（完成或受控停止、Shield、列貼合、命令有限、禁止接觸）；結果照記，不作為通過條件以外的比較。
    python3 evaluation/test_bottle_w1_shrow_np.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
import bottle_w1_shrow_np_ctrl as SN                    # noqa: E402
import test_bottle_w1_dyn as TD                         # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def cmds(f):
    return np.array([np.r_[h['u'], h['s']] for h in f.hist]), np.array([h['t'] for h in f.hist])


def case_nominal():
    c0, f0, r0 = TL.run(SR.W1PlaceController, False)
    c1, f1, r1 = TL.run(SN.W1PlaceController, False)
    U0, t0 = cmds(f0)
    U1, _ = cmds(f1)
    n = min(len(U0), len(U1))
    dd = np.abs(U0[:n] - U1[:n]).max(axis=1)
    k = np.flatnonzero(dd > 0)
    nd = [len([o for o in (cc.get('obs') or []) if not o.get('static')]) for cc in (c1.compact or [])]
    first = None if not len(k) else float(t0[k[0]])
    print(f"   名目：SHROW DONE {r0['mission_done']}／瞬時組 DONE {r1['mission_done']}；命令首次不同 {first}、最大差 {float(dd.max()):.3e}；瞬時組歸零動態軌跡 {c1.n_zeroed} 次、含動態軌跡的求解 {sum(1 for x in nd if x > 0)}")
    chk('1 名目：無動態軌跡時兩組命令逐位元相同（有動態軌跡則差異只能出現在其後）', (not len(k)) or c1.n_zeroed > 0, (first, c1.n_zeroed))
    return dict(first_diff=first, max_diff=float(dd.max()), n_zeroed=c1.n_zeroed, done=[bool(r0['mission_done']), bool(r1['mission_done'])])


def case_dyn(case):
    TD.CA = DC.CASES[case]
    out = {}
    for nm, cls in (('SHROW', SR.W1PlaceController), ('SHROW_NP', SN.W1PlaceController)):
        holder = {}

        class Cap(cls):
            def report(self):
                r = super().report()
                holder['ps'], holder['sh'] = r['paired_strategy'], r['nav_chain']['shield']
                return r
        o = TD.run(Cap)
        u0 = holder['ps']['u0_rows']
        o.update(u0_bind=u0['n_bind'], u0_viol=u0['violation'], shield=holder['sh'], n_zeroed=holder['ps'].get('n_dyn_zeroed'))
        out[nm] = o
        print(f"   {case} {nm}：完成 {o['done']} 阻點 {o['stop'] and o['stop'].get('why')} @ {o['stop'] and round(o['stop']['t'], 2)}；DONE {o['t_done']}；列貼合 {o['u0_bind']} 違反 {o['u0_viol']}；Shield 介入 {o['shield']['active']}；"
              f"底盤中心到障礙表面最小 {o['base_surface_min']:.3f}；歸零 {o['n_zeroed']}")
        chk(f'{case} {nm}：完成或受控停止、命令有限、列無違反', (o['done'] or o['stop'] is not None) and o['finite'] and o['u0_viol'] is None, (o['stop'], o['u0_viol']))
    return out


def main():
    res = dict(nominal=case_nominal())
    res['w1_dyn_pass1'] = case_dyn('w1_dyn_pass1')
    res['w1_dyn_overtake1'] = case_dyn('w1_dyn_overtake1')
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_shrow_np_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
