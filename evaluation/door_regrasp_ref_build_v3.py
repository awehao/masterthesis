#!/usr/bin/env python3
"""v3（Codex r60 §1–§2；v2 保留——v2 的 T_HG 仍為舊抓法、replan 指向失敗的 v5，與實際構型來源 v7 φ＋30° 錯配 3.49 mm／30°）。
來源：door_regrasp_retime_v5（replan v8：φ＋30°、B1 (0.65, −0.05, 270°)、M 段門角包絡）。
登錄（同一成功候選，一路供任務端、Gate6、判定器）：inner_grasp {phi_deg, T_HG＝Rx(φ)·T_HG_base}；outer_grasp_basis {path, sha256}（bench_6 外側抓取基準，獨立不覆寫）；
replan_file／source_file {path, sha256}；**不再有含義模糊的 T_HG 鍵**。起始銜接門角改 0.7°（候選；門角、底盤、偏航、手臂耦合，不代表各容差任意組合皆已驗，
實際聯合銜接檢查保留）。門角修正核對取 −0.7／0／+0.7°。建置後以 door_regrasp_ref_registry.validate 自驗，不過即不寫檔。
以下為 v2 說明：
v2（Codex r54–r57）：來源改 door_regrasp_retime_v4（重規劃 v5：B1 (0.65, −0.10, 225°)、整機 M 段、經過點接近；依預測淨空逐段降速）。
同一份參考登錄新 B1、預抓取分支與目標構型（任務端 RefRuntime3、執行端 Gate5、判定器共用；不平移舊參考）；另登錄 B1 收臂構型（Gate5 目標）。
其餘（切段規則、銜接範圍、門角 ±1° 剛體修正核對）同 v1。以下為 v1 說明：
換抓參考包建置 v1（離線；Codex r49 §4、r50）。來源＝door_coopen_pass_close_inner_retime_v3_static.json 的 20 Hz 參考（同一條已驗路徑，不重規劃）。
切成各模式段（時間各自自 0 起；任務端在該模式**套用確認後**才起算）：
  mode 3 RETREAT_WB ：release_outer 自起點到 TCP 沿起點工具 −z 退開量 s **第一次達到 0.06 m**（v4 的 W1 終點）——含 s(τ)、底盤；
    （r1 誤取 s 最大處：收臂時 TCP 又沿同向移動、s 達 0.226，整段 25 s 被歸入 mode 3；改名保留 *_r1_split_ik_fail.json）
  mode 4 STOW_MOVE  ：release_outer 其餘（W2、W3、收臂）＋M（A* 到 B1）；
  mode 5 APPROACH_INNER：A（關節空間接近）＋I（沿工具 +z 插入 0.06 m）；pregrasp_T＝A 末、grasp_T＝I 末；
  mode 6 CLOSE_INNER：目標＝grasp_T 保持（手指由執行端權威閉合）。
每點存 TCP 4×4（列優先 16）、底盤 (x, y, yaw)、手臂 6 軸、手指開度、門角（參考值）。
起始銜接範圍（登錄；超出 ⇒ 任務端停止並報偏差，不臨時搜尋）：底盤位置 0.020 m、偏航 2.0°（K3-B base_dev）、手臂各軸 0.05 rad（joint_margin）、門角 1.0°。
門角銜接的路徑影響（建置時離線核）：mode 5 參考構型**整體繞鉸鏈軸剛體轉 w·Δq**——底盤位置與偏航一起轉、手臂關節＝參考值，TCP 目標因此＝Rz(−w·Δq)·TCP_ref
  （內側把手隨門剛體轉動）；w 在 A 段由 0（收臂起點，不轉）線性增到 1（預抓取），I 段＝1。
  r2 的「底盤固定 B1、手臂 IK 重解」在預抓取附近 Δq ±1° 都 IK 無解（手臂近限位），改名保留 *_r2_armonly_fail.json。
  以下 r2 說明保留供追溯：以 IK（door_offline_check_v1.ik，**同索引參考關節為種子**，r1 用前一點種子致分支漂移、Δq＝0 也失敗）
  對 Δq ∈ {−1, 0, +1}° 的修正路徑逐點重解（Δq＝0 直接用參考構型），並以 door_coopen_pass_close_inner_v1.check（v3 雙把手、分階段規則）核淨空；
  任一點失敗 ⇒ 該 Δq 不在可接受範圍（結果寫入包，任務端以包內 dq_ok_deg 為準）。

    python3 -u evaluation/door_regrasp_ref_build_v1.py   # 寫 results/door/door_regrasp_ref_v1.json（不覆寫）
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_retime_v5.json')
REPLAN = os.path.join(HERE, 'results', 'door', 'door_regrasp_replan_v8_phi+30.json')
OUTER_GB = 'evaluation/results/door/coopen_core_grasp_basis_r4.json'
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v3.json')
C = PI.C
BRIDGE = {'base_m': 0.020, 'base_deg': 2.0, 'arm_rad': 0.05, 'door_deg': 0.7}
DQ_TEST = (-0.7, 0.0, 0.7)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    raw = open(SRC, 'rb').read()
    src = json.loads(raw)
    R = src['reference_20hz']['rows']
    X = np.array([r[1] for r in R], float)
    st = [r[4] for r in R]
    tt = np.array([r[0] for r in R], float)
    TCP = [np.asarray(C.K.fk([float(v) for v in x], 'link_tcp'), float) for x in X]
    # mode 3：release_outer 內 s 最大處
    i_rel = [i for i, s in enumerate(st) if s == 'release_outer']
    p0, z0 = TCP[0][:3, 3], TCP[0][:3, 2]
    s_all = np.array([float((p0 - TCP[i][:3, 3]) @ z0) for i in i_rel])
    S_W1 = 0.06
    if not np.any(s_all >= S_W1 - 1e-4):
        raise SystemExit('release_outer 未達 W1 終點 0.06 m')
    i3_end = i_rel[int(np.argmax(s_all >= S_W1 - 1e-4))]
    seg = {3: list(range(0, i3_end + 1)), 4: [i for i in range(i3_end + 1, len(R)) if st[i] in ('release_outer', 'M')],
           5: [i for i, s in enumerate(st) if s in ('A', 'I')], 6: [i for i, s in enumerate(st) if s == 'G']}
    assert seg[4][0] == i3_end + 1 and seg[5][0] == seg[4][-1] + 1 and seg[6][0] == seg[5][-1] + 1
    i_pre = max(i for i in seg[5] if st[i] == 'A')
    i_grasp = seg[5][-1]

    def pack(idx, with_s=False):
        tb = tt[idx[0]]
        d = {'t': [float(tt[i] - tb) for i in idx], 'tcp': [TCP[i].reshape(-1).tolist() for i in idx], 'base': [X[i, :3].tolist() for i in idx],
             'arm': [X[i, 3:].tolist() for i in idx], 'qf': [float(R[i][3]) for i in idx], 'q_door_deg': [float(R[i][2]) for i in idx],
             'src_index': [int(i) for i in idx]}
        if with_s:
            d['s'] = [float((p0 - TCP[i][:3, 3]) @ z0) for i in idx]
        return d
    rp = json.load(open(REPLAN))
    import door_regrasp_ref_registry as REG
    sha = lambda pth: hashlib.sha256(open(pth, 'rb').read()).hexdigest()    # noqa: E731
    phi = float(rp['phi_deg'])
    out = {'schema': 'door_regrasp_ref/3',
           'inner_grasp': {'phi_deg': phi, 'T_HG': (REG.Rx4(math.radians(phi)) @ PI.THG).reshape(-1).tolist(), 'basis': 'Rx(phi)·door_coopen_reach_v1.T_HG()'},
           'outer_grasp_basis': {'path': OUTER_GB, 'sha256': sha(os.path.join(os.path.dirname(HERE), OUTER_GB))},
           'replan_file': {'path': os.path.relpath(REPLAN, HERE), 'sha256': sha(REPLAN)}, 'source_file': {'path': os.path.relpath(SRC, HERE), 'sha256': sha(SRC)},
           'replan': {'B1_deg': rp['B1'], 'chosen': rp.get('chosen'), 'N1_seed': (rp.get('N1') or {}).get('seed'),
                                                      'M_method': (rp.get('N2') or {}).get('method')}, 'source': os.path.relpath(SRC, HERE), 'source_sha256': hashlib.sha256(raw).hexdigest(),
           'dt': 0.05, 'q_ref_deg': float(R[0][2]), 'bridge': BRIDGE, 'stow': list(TJ.STOW), 'B1': X[seg[5][0], :3].tolist(),
           'pregrasp_T': TCP[i_pre].reshape(-1).tolist(), 'grasp_T': TCP[i_grasp].reshape(-1).tolist(),
           'mode3_s_end_m': float(s_all[i_rel.index(i3_end)]), 'segments': {str(k): pack(v, with_s=(k == 3)) for k, v in seg.items()}}
    # 門角銜接的路徑影響：mode 5 TCP 目標 Rz(−Δq) 修正後 IK 重解＋v3 分階段淨空
    dq_res = {}
    base5 = X[seg[5][0], :3].copy()
    for dqd in DQ_TEST:
        TJ.cool()
        q_deg = float(R[seg[5][0]][2]) + dqd
        nA = sum(1 for i in seg[5] if st[i] == 'A')
        bad, n_ok, worst = None, 0, {'nonfinger': math.inf, 'self': math.inf, 'base': math.inf, 'jm': math.inf}
        bmax = 0.0
        for j, i in enumerate(seg[5]):
            w = min(1.0, j / max(1, nA - 1)) if st[i] == 'A' else 1.0
            a = -math.radians(w * dqd)
            b = X[i, :3]
            bxy = Rz(a)[:2, :2] @ b[:2]
            x = np.concatenate([[bxy[0], bxy[1], b[2] + a], X[i, 3:]])
            bmax = max(bmax, float(np.linalg.norm(bxy - base5[:2])))
            Tg = Rz(a) @ TCP[i]
            if np.max(np.abs(C.K.fk([float(v) for v in x], 'link_tcp') - Tg)) > 1e-9:
                bad = {'i': int(i), 'stage': st[i], 'why': 'rigid_rotation_inconsistent'}
                break
            ok, why, info = PI.check(x, q_deg, float(R[i][3]), st[i])
            if not ok:
                bad = {'i': int(i), 'stage': st[i], 'why': why}
                break
            n_ok += 1
            for k in ('nonfinger', 'self', 'base'):
                worst[k] = min(worst[k], float(info[k][0]))
            worst['jm'] = min(worst['jm'], float(info['jm']))
        dq_res[str(dqd)] = {'ok': bad is None, 'first_fail': bad, 'n_points_ok': n_ok, 'n_points': len(seg[5]), 'min_clear_m': worst,
                            'base_shift_max_m': bmax}
        print(f'Δq {dqd:+.1f}°：{"通過" if bad is None else "失敗 " + json.dumps(bad, ensure_ascii=False)}；最小 {({k: round(v * 1e3, 2) for k, v in worst.items()})}', flush=True)
    out['door_bridge_check'] = {'method': 'mode 5 參考構型繞鉸鏈軸剛體轉 w·Δq（A 段 w 0→1、I 段 1；底盤一起轉、手臂關節＝參考）；FK 一致性；PI.check（v3、門角 q_ref＋Δq、分階段）',
                                'results': dq_res,
                                'dq_ok_deg': [float(k) for k, v in dq_res.items() if v['ok']]}
    out['wall_s'] = time.time() - t0w
    okv, whyv = REG.validate(json.loads(json.dumps(out, default=float)))
    out['registry_validate'] = {'ok': okv, 'why': whyv}
    if not okv:
        raise SystemExit(f'登錄自驗不過：{whyv}（不寫檔）')
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('q_ref_deg', 'B1', 'mode3_s_end_m', 'bridge')}, ensure_ascii=False),
          {k: len(v['t']) for k, v in out['segments'].items()}, {k: round(v['t'][-1], 2) for k, v in out['segments'].items()}, flush=True)


if __name__ == '__main__':
    main()
