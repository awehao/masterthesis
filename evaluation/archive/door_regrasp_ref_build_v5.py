#!/usr/bin/env python3
"""換抓＋關門參考包 v5（Codex r67）：ref v4 全部不變，另加 segments 7（PASS_CLOSE＝關門時間化 v1 的 P 列）與 8（RELEASE_RETREAT＝R_open／R_slide／R_stow 列）。
每點欄位同 v4（t 自段起點、tcp、base、arm、qf、q_door_deg、src_index、stage）；登錄 close_source {path, sha256}（door_regrasp_close_retime_v1.json）
與 close_plan {path, sha256}。銜接核：mode 7 起點構型＝ref v4 mode 5 終點（≤ 1e-9；關門規劃自該構型起）；登錄自驗（registry.validate）須通過。

    python3 -u evaluation/door_regrasp_ref_build_v5.py   # 寫 results/door/door_regrasp_ref_v5.json（不覆寫）
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_regrasp_ref_registry as REG                # noqa: E402

V4 = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v4.json')
CR = os.path.join(HERE, 'results', 'door', 'door_regrasp_close_retime_v1.json')
CP = os.path.join(HERE, 'results', 'door', 'door_regrasp_close_plan_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v5.json')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    pkg = json.load(open(V4))
    R = json.load(open(CR))['reference_20hz']['rows']
    seg = {7: [i for i, r in enumerate(R) if r[4] in ('P', 'G')], 8: [i for i, r in enumerate(R) if r[4].startswith('R_')]}
    assert seg[7] and seg[8] and seg[8][0] == seg[7][-1] + 1, '段不連續'
    x5 = np.r_[pkg['segments']['5']['base'][-1], pkg['segments']['5']['arm'][-1]]
    d_join = float(np.max(np.abs(np.asarray(R[0][1], float) - x5)))
    if d_join > 1e-9:
        raise SystemExit(f'mode 7 起點與 ref v4 mode 5 終點不符 {d_join}')

    def one(idx):
        tb = R[idx[0]][0]
        X = [np.asarray(R[i][1], float) for i in idx]
        return {'t': [float(R[i][0] - tb) for i in idx], 'tcp': [np.asarray(PI.C.K.fk(list(x), 'link_tcp'), float).reshape(-1).tolist() for x in X],
                'base': [x[:3].tolist() for x in X], 'arm': [x[3:].tolist() for x in X], 'qf': [float(R[i][3]) for i in idx],
                'q_door_deg': [float(R[i][2]) for i in idx], 'src_index': [int(i) for i in idx], 'stage': [R[i][4] for i in idx]}
    pkg['schema'] = 'door_regrasp_ref/5'
    pkg['segments']['7'], pkg['segments']['8'] = one(seg[7]), one(seg[8])
    pkg['close_source'] = {'path': os.path.relpath(CR, HERE), 'sha256': sha(CR)}
    pkg['close_plan'] = {'path': os.path.relpath(CP, HERE), 'sha256': sha(CP)}
    pkg['derived_from'] = {'path': os.path.relpath(V4, HERE), 'sha256': sha(V4)}
    pkg['close_join_max_abs'] = d_join
    ok, why = REG.validate(json.loads(json.dumps(pkg, default=float)))
    pkg['registry_validate'] = {'ok': ok, 'why': why}
    if not ok:
        raise SystemExit(f'登錄自驗不過：{why}')
    json.dump(pkg, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print({k: round(v['t'][-1], 2) for k, v in pkg['segments'].items()}, 'q7', pkg['segments']['7']['q_door_deg'][0], '→', pkg['segments']['7']['q_door_deg'][-1])


if __name__ == '__main__':
    main()
