#!/usr/bin/env python3
"""ref pb1f（Codex 20261029_055110 核可單一收臂姿態候選）：pb1 → pb1_finish → pb1e 三步流程**只把 TJ.STOW 換成候選**，其餘程式、規則、門檻原樣重跑。
原因：leader_sc_2 mode 4 收臂移動中 geometry_clearance_self（link1|link5）中止；舊 STOW [0,0,0.035,0,−1.571,0] 的 link1|link5 僅 10.8 mm，對 j3 極敏感
（j3 0.035→0.018 ⇒ 6.9 mm）。候選 STOW_F＝[0, 0, 0.15, 0, −1.10, 0]：link1|link5 35.4 mm、link4|link6 12.3 mm（整體 self 受 link4|link6 限制；
所查姿態範圍〔j3 0.15–0.5、j5 −1.57…+0.3〕未達 20 mm——Codex 已取消該設計目標，原預測淨空判準與誤差扣減不變）。
STOW 在流程中只讀 door_coopen_traj_v1.STOW——本程式**只在自身行程內**原地改寫該陣列，不改 door_coopen_traj_v1.py、不覆寫舊結果（放手後收臂內插、M 段整機 A* 手臂姿態、mode 5 STOW→V→預抓取、包內 'stow'）⇒ 原地改寫該陣列即前後一致；
執行端 Gate5 讀包內 'stow'。mode 7／8（拉關、放手）不讀 STOW、不改。
步驟（輸出皆新檔、不覆寫）：
  P＝door_regrasp_pb1.main（F 改 *_pb1f*；ref_build_v5 在 2π 偏航銜接核停下屬既知行為）→ door_regrasp_pb1_finish（SUF＝'_f'）→ door_regrasp_ref_pb1e.main（SRC／REL／OUT 改 pb1f）。
  另核：seg 3／4／5 逐 10 取樣以 v3 full 後端算 self（標稱、＋dev、−dev；dev＝leader_sc_2 中止時**實測關節**對參考的完整偏差向量）寫入包 pb1f.self_check——只作局部證據，不是穩健性保證。

    python3 -u evaluation/door_regrasp_pb1f.py   # 最終輸出 results/door/door_regrasp_ref_pb1f.json
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_traj_v1 as TJ                       # noqa: E402

STOW_OLD = TJ.STOW.copy()
STOW_F = np.array([0.0, 0.0, 0.15, 0.0, -1.10, 0.0])
DEV = np.array([-0.0725, -0.026, -0.0214, -0.0734, -0.0072, 0.0051])  # leader_sc_2 137.7 s：實測手臂關節（room_run q_arm_meas）− 參考
RD = os.path.join(HERE, 'results', 'door')
OUT = os.path.join(RD, 'door_regrasp_ref_pb1f.json')


def self_check(path):
    import door_k3m_geom_monitor_v3 as GM3
    fn = GM3.backend_v3(self_mode='full')
    p = json.load(open(path))
    res = {}
    for sg in ('3', '4', '5'):
        s = p['segments'][sg]
        B, A, qd, qf = (np.asarray(s[k], float) for k in ('base', 'arm', 'q_door_deg', 'qf'))
        best = {'nominal': (math.inf, None, None), 'plus_dev': (math.inf, None, None), 'minus_dev': (math.inf, None, None)}
        for i in list(range(0, len(B), 10)) + [len(B) - 1]:
            for key, a in (('nominal', A[i]), ('plus_dev', A[i] + DEV), ('minus_dev', A[i] - DEV)):
                d, who = fn(np.r_[B[i], a], math.radians(qd[i]), float(qf[i]))['self']
                if d < best[key][0]:
                    best[key] = (float(d), who, float(s['t'][i]))
        res[sg] = {k: {'d_m': v[0], 'pair': v[1], 't_s': v[2]} for k, v in best.items()}
    return res


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    TJ.STOW[:] = STOW_F
    import door_regrasp_pb1 as P
    P.F = {k: os.path.join(RD, f) for k, f in (('rel', 'door_regrasp_pb1f_release.json'), ('replan', 'door_regrasp_replan_pb1f.json'),
                                                ('retime', 'door_regrasp_retime_pb1f.json'), ('v4', 'door_regrasp_ref_pb1_v4_f.json'),
                                                ('v5', 'door_regrasp_ref_pb1_v5_f_unused.json'))}
    try:
        P.main()
    except SystemExit as e:
        print(f'P（pb1 流程）停於：{e}（預期為 2π 偏航銜接核）', flush=True)
    if not os.path.exists(P.F['v4']):
        raise SystemExit('pb1 流程未產生 v4 包 ⇒ 停止（見上方阻擋）')
    sys.argv = [sys.argv[0], '_f']
    import door_regrasp_pb1_finish as FIN
    FIN.main()
    import door_regrasp_ref_pb1e as E
    E.SRC, E.REL, E.OUT = os.path.join(RD, 'door_regrasp_ref_pb1_f.json'), P.F['rel'], OUT
    E.main()
    pk = json.load(open(OUT))
    pk['pb1f'] = {'stow_old': STOW_OLD.tolist(), 'stow': STOW_F.tolist(), 'dev_vector': DEV.tolist(), 'dev_source': 'door_pullback_extf_leader_sc_2 room_run q_arm_meas @137.7 s − ref pb1e seg4',
                  'self_check': self_check(OUT), 'wall_s': time.time() - t0,
                  'note': '所查姿態範圍未達標稱 self 20 mm（設計目標已取消；保護判準不變）；dev 檢查只作局部證據；新參考姿態不等於關節構型約束，不保證實際構型照走'}
    assert pk['stow'] == STOW_F.tolist(), pk['stow']
    json.dump(pk, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('pb1f self_check', json.dumps(pk['pb1f']['self_check'], ensure_ascii=False), f'牆鐘 {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
