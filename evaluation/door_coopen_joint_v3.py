#!/usr/bin/env python3
"""連續協同通行：聯合搜尋 v3——放手後收得回來也列為搜尋條件（離線靜態，無 Isaac；單一程序＋溫度暫停）。

v2（door_coopen_joint_v2）找到開門末段朝出口推進的候選（φ60 s0.85 q48：末 10° D 降 0.235 m、轉角 6°，握回 0°），
但 door_coopen_verify_v1／door_coopen_release_plan_v1 顯示該候選放手時 link2 距門扇 6.7 mm，張爪後任何退法在 2 cm 內 link2 撞門扇
（門靜止／k3b 壓力情境／回轉 0.5° 皆失敗）——v2 只要求非手指 ≥ 5 mm，沒有要求放手後收得回來。v2 全掃描於 φ60 s0.85 q56 後停止（部分紀錄 door_coopen_joint_v2_partial.log）。
本版＝v2 的關 1（背面放手狀態＋門靜止握持前進）之後插入：
  關 1b（收臂可行）：door_coopen_release_plan_v1.plan_state，先核門靜止情境，成立者再核 k3b 壓力情境（張爪後 1 °/s × 0.16 s，之後 0.24 °/s）
    與回轉 0.5°；三者皆從張爪到 STOW 全程成立才通過。
其後同 v2：關 2（末 10° 真實測地距離 D 降 ≥ 0.05 m、開門末段與握持前進轉角 ≤ 60°）、關 3（依 D 降取前 3 反向到 0°）。
範圍同 v2（夾法 6 種 × 放手門角 7 個），但關 1b 較貴，故每組關 1 狀態依 min(包絡淨空, 瓶頸) 排序至多取 6 個進關 1b。

    python3 evaluation/door_coopen_joint_v3.py [phi s q_rel]   # 寫 results/door/door_coopen_joint_v3[_單組].json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgehold_v1 as EH                   # noqa: E402
import door_coopen_joint_v2 as J2                      # noqa: E402
import door_coopen_release_plan_v1 as RP               # noqa: E402
import door_coopen_release_set_v1 as RS                # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v3.json')
N_1B = 6


def main():
    global OUT
    grasps, q_rels = J2.GRASPS, J2.Q_RELS
    if len(sys.argv) > 1:
        phi, s_, qr = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
        grasps, q_rels = [(phi, s_, 0)], (qr,)
        OUT = OUT.replace('.json', f'_phi{phi:g}_s{s_:g}_q{qr}.json')
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    fields = {q: RS.field(float(q)) for q in q_rels}
    dmaps = {q: J2.d_map(fields[q]) for q in q_rels}
    survivors, summary = [], []
    for g in grasps:
        for q_rel in q_rels:
            rej = {}
            s1 = J2.stage1(g, q_rel, fields[q_rel], rej)
            s1.sort(key=lambda c: -min(J2.PS.F_at(fields[q_rel], c['base'][:2]), 1.0))
            s1b = []
            for c in s1[:N_1B]:
                ok = True
                rp = {}
                for sc in ('static', 'k3b_stress', 'reverse_0p5'):
                    rows, fail, _, dyaw = RP.plan_state(g, q_rel, c['path'], {'base': c['base'].tolist(), 'qa': c['qa'].tolist()}, c['fwd_m'], sc)
                    rp[sc] = {'ok': fail is None, 'fail': fail, 't_end_s': rows[-1]['t'] if rows else 0.0,
                              'min_margin_mm': (min(r['min_margin_m'] for r in rows) * 1000) if rows else None, 'dyaw_deg': math.degrees(dyaw)}
                    if fail is not None:
                        rej[f'release_{sc}'] = rej.get(f'release_{sc}', 0) + 1
                        ok = False
                        break
                if ok:
                    c['release_plan'] = rp
                    s1b.append(c)
            ok2 = []
            for c in s1b:
                b = J2.blend(g, q_rel, c, *dmaps[q_rel])
                if b is None:
                    rej['blend_infeasible'] = rej.get('blend_infeasible', 0) + 1
                    continue
                if b['D_drop_m'] is None:
                    rej['blend_start_not_on_exit_route'] = rej.get('blend_start_not_on_exit_route', 0) + 1
                    continue
                if b['D_drop_m'] < 0.05:
                    rej['blend_D_drop_lt5cm'] = rej.get('blend_D_drop_lt5cm', 0) + 1
                    continue
                if b['turn_open_fwd_deg'] is None or b['turn_open_fwd_deg'] > 60:
                    rej['blend_turn_gt60'] = rej.get('blend_turn_gt60', 0) + 1
                    continue
                ok2.append((b, c))
            summary.append({'grasp': list(g), 'q_rel_deg': q_rel, 'n_stage1': len(s1), 'n_stage1b': len(s1b), 'n_stage2_ok': len(ok2), 'reject': rej,
                            'best_D_drop_m': max((bb['D_drop_m'] for bb, _ in ok2), default=None)})
            for b, c in ok2:
                survivors.append({'grasp': list(g), 'q_rel_deg': q_rel, 'stage1': c, 'blend': b})
            print(f"φ{g[0]:g} s{g[1]:.2f} q{q_rel}：關1 {len(s1)}、關1b 收臂可行 {len(s1b)}、關2 通過 {len(ok2)}；淘汰 {rej}；{time.time() - t0:.0f} s", flush=True)
    survivors.sort(key=lambda z: -z['blend']['D_drop_m'])
    full = []
    for sv in survivors[:3]:
        g, q_rel = tuple(sv['grasp']), sv['q_rel_deg']
        s10 = sv['blend']['state10']
        seed = {'x': float(s10['base'][0]), 'y': float(s10['base'][1]), 'yaw_deg': math.degrees(s10['base'][2]), 'phi': g[0], 's': g[1], 'roll': g[2],
                'qa': [float(v) for v in s10['qa']], 'jm': s10['jm'], 'clear': s10['clear']}
        orig = TJ.check
        TJ.check = lambda b, q, T, sd, push=False, full=False: (lambda r: r if (r is not None and r['clear']['base'] >= J2.PLAN) else None)(orig(b, q, T, sd, push=push, full=full))
        try:
            layer, qend, why = EH.run(seed, q_rel - J2.N_BLEND, 0, EH.FRONT)
        finally:
            TJ.check = orig
        P = layer[0]['path'][::-1] + list(sv['blend']['fw'][1:])
        rec = {'grasp': list(g), 'q_rel_deg': q_rel, 'reached_q_deg': qend, 'why': why, 'D_drop_last10_m': sv['blend']['D_drop_m'],
               'turn_open_fwd_deg': sv['blend']['turn_open_fwd_deg'], 'fwd_m': sv['stage1']['fwd_m'], 'release_base': sv['stage1']['release_base'],
               'exit_path': sv['stage1']['path'], 'release_plan': sv['stage1']['release_plan'],
               'trajectory': [{'q_deg': float(p['q']), 'base': [float(v) for v in p['base']], 'qa': [float(v) for v in p['qa']],
                               'clear': p['clear'], 'jm': p['jm']} for p in P] if qend == 0 else None}
        full.append(rec)
        print(f"完整持握 φ{g[0]:g} s{g[1]:.2f} q{q_rel}：反向到 {qend}°（{why}）；末 10° D 降 {rec['D_drop_last10_m']:.3f} m、轉角 {rec['turn_open_fwd_deg']}；{time.time() - t0:.0f} s", flush=True)
    json.dump({'schema': 'door_coopen_joint/3', 'grasps': grasps, 'q_rels': q_rels, 'plan_m': J2.PLAN, 'n_blend_deg': J2.N_BLEND, 'n_1b': N_1B,
               'summary': summary, 'n_survivors': len(survivors),
               'survivors_brief': [{'grasp': s['grasp'], 'q_rel_deg': s['q_rel_deg'], 'D_drop_m': s['blend']['D_drop_m'], 'turn_deg': s['blend']['turn_open_fwd_deg'],
                                    'base': s['stage1']['base'].tolist(), 'release_plan': s['stage1']['release_plan']} for s in survivors],
               'full': full, 'not_modeled': ['格點與取樣（非距離下界）', '剛性夾持、靜態', '門餘轉以壓力情境代替（非上界）', '本範圍外的夾法與放手門角']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
