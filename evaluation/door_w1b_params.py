#!/usr/bin/env python3
"""W1b 參數登錄（Codex 20261024_231908 B 節：分類清楚、由規則推得、不手調）。

推得並寫入 results/door/w1b_params_<θ>.json：
  任務參數延伸：
    θ_goal；K3-A 門角上界 θ_goal＋1°（over_goal 1°，不變）；
    OPEN 逾時＝完整斜坡時間（DoorTargetK3 含末段減速，由 door_k3m_offline_replay.reference 計算）＋ 8° 版的絕對餘裕
      （8° 版：逾時 30 s − 斜坡 9.7 s ＝ 20.3 s）。
  保護適用範圍變更（重新登錄）：
    走廊表＝door_wide_corridor_table_90.json（線上引導、走廊監看、離線驗證同一表；記 sha256）；內側帶 20 mm／0.8° 不變；
    執行端活動包絡＝走廊參考在 [0, θ_goal＋1°] 的 SE(2) 範圍（前進、側移各取 min/max；偏航取 max|·|），每側加外側取樣帶
      M＝40 mm／4.5°（door_wide_offline_corridor 外側取樣範圍，涵蓋內帶 20 mm／0.8° ＋ 停止餘量 12.6 mm／2.9°）；
    獨立觀察器門角上限＝θ_goal＋4°（固定角度餘裕，與任務上界 θ_goal＋1° 分開登錄）。
  控制成本項變更：引導 w_vref 1、w_qn 0（GuidanceCfg 預設），協同話題 /door/wide_coord。
核對：每個門角（0.1° 步）參考 ± (內帶＋停止餘量) 的四角點都在包絡內；並列 8° 舊包絡供對照。
r2（Codex 20261024_234630 第 2 項）：偏移改在**參考位姿座標**以 SE(2) 合成（ref(q) ⊕ (d前, d左, d偏航)，與線上走廊偏差定義一致），
  再轉回走廊表錨點座標：包絡＝完整外帶（±40 mm、±40 mm、±4.5° 的 8 個角點）在 [0, θ+1°] 的範圍；
  核對「內帶＋停止餘量」角點與包絡邊界的最小餘裕。輸出 w1b_params_<θ>_r2.json（r1 輸出保留）。停止餘量仍只是情境假設。

    python3 evaluation/door_w1b_params.py 30
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_policy as KP                          # noqa: E402
import door_k3m_corridor as COR                       # noqa: E402
import door_k3m_offline_replay as R                   # noqa: E402
import door_wide_guidance as G                        # noqa: E402

TABLE = os.path.join(HERE, 'results', 'door', 'door_wide_corridor_table_90.json')
OPEN_MARGIN_8 = 30.0 - 9.7
M_LIN, M_YAW = 0.040, 4.5
INNER = (0.020, 0.8)
STOP_RES = (0.0126, 2.9)
OLD_ENVELOPE = (-0.15, 0.02, -0.05, 0.10, 6.0)


def main():
    theta = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    out = os.path.join(HERE, 'results', 'door', f'w1b_params_{int(round(theta))}_r2.json')   # r2
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(R.RUN, 'task.json')))
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    ramp8 = R.reference(T_hold, KP.K3AConfig(theta_goal_deg=8.0))[-1][0]
    ramp = R.reference(T_hold, KP.K3AConfig(theta_goal_deg=theta))[-1][0]
    open_to = round(ramp + OPEN_MARGIN_8, 1)
    tbl = json.load(open(TABLE))
    C = COR.Corridor.from_json(tbl['table'])
    qs = np.arange(0.0, theta + 1.0 + 1e-9, 0.1)
    L = np.array([C.ref_local(q) for q in qs])
    ainv = COR.se2_inv(C.anchor)

    def corners(q, m, my):
        ref = C.ref_pose(q)
        out = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sw in (-1, 1):
                    w = COR.se2_compose(ainv, COR.se2_compose(ref, (sx * m, sy * m, math.radians(sw * my))))
                    out.append((w[0], w[1], math.degrees(w[2])))
        return out
    P = np.array([c for q in qs for c in corners(q, M_LIN, M_YAW)])
    env = (float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max()), float(np.abs(P[:, 2]).max()))
    dl, dy = INNER[0] + STOP_RES[0], INNER[1] + STOP_RES[1]
    Q = np.array([c for q in qs for c in corners(q, dl, dy)])
    worst = {'fwd_lo': float(Q[:, 0].min() - env[0]), 'fwd_hi': float(env[1] - Q[:, 0].max()),
             'lat_lo': float(Q[:, 1].min() - env[2]), 'lat_hi': float(env[3] - Q[:, 1].max()),
             'yaw': float(env[4] - np.abs(Q[:, 2]).max())}
    gc = G.GuidanceCfg()
    doc = {'schema': 'w1b_params/2', 'theta_goal_deg': theta,
           'task_param_extension': {'theta_goal_deg': theta, 'over_goal_deg': 1.0, 'policy_upper_deg': theta + 1.0,
                                    'ramp_s': ramp, 'ramp_8deg_s': ramp8, 'open_margin_s_from_8deg': OPEN_MARGIN_8,
                                    'open_timeout_s': open_to, 'rule': 'OPEN 逾時＝完整斜坡（含末段減速）＋ 8° 版絕對餘裕 20.3 s'},
           'protection_scope_change': {
               'corridor_table': os.path.relpath(TABLE, os.path.dirname(HERE)), 'corridor_table_sha256': hashlib.sha256(open(TABLE, 'rb').read()).hexdigest(),
               'corridor_inner_mm': INNER[0] * 1e3, 'corridor_inner_deg': INNER[1],
               'envelope': {'fwd_lo': env[0], 'fwd_hi': env[1], 'lat_lo': env[2], 'lat_hi': env[3], 'yaw_abs_deg': env[4],
                            'cli': ','.join(f'{v:.4f}' for v in env),
                            'rule': f'完整外帶 ±{M_LIN * 1e3:.0f} mm／±{M_YAW}° 在參考位姿座標以 SE(2) 合成的角點，於 [0, θ+1°] 轉回走廊表錨點座標後的範圍'},
               'envelope_margin_check': {'inner_plus_stop_residual': [dl, dy], 'composition': 'SE2：ref(q) ⊕ 偏移（參考位姿座標）', 'min_margin_m_or_deg': worst,
                                         'ok': all(v >= 0 for v in worst.values())},
               'old_envelope_8deg': OLD_ENVELOPE,
               'observer_door_max_deg': theta + 4.0, 'observer_rule': '固定角度餘裕 θ＋4°（與任務上界 θ＋1° 分開登錄）',
               'stop_residual_source': 'door_k3m_offline_stop（8° 情境 12.6 mm／2.9°；30° 斜坡速率相同，沿用並揭露為假設）'},
           'control_cost_change': {'guidance': vars(gc), 'coord_topic': '/door/wide_coord',
                                   'phases_on': ['OPEN', 'SETTLE'], 'phases_off_weight': 0.0,
                                   'note': '只經求解器既有協同輸入（parse_coord v1）作為軟成本，不新增獨立底盤命令來源；PAUSE 起 w_vref＝0'},
           'pause': 'p4b（主動保持停止、手臂 0.5 s 窗口）＋判定器 p4b2（起跑前凍結）',
           'reference_table_note': '延伸表與 8° 舊表在 0–6° 一致；8° 參考偏航相差約 0.116°（舊表含 8° 目標前減速），不稱完全相同'}
    json.dump(doc, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(doc, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
