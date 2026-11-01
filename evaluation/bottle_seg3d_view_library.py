#!/usr/bin/env python3
"""有界自適應觀察的視點庫（Codex 20261101_230248 (b)；純運動學、無 Isaac）。視點不依實際桌高或佈置編號人工挑：
每個視點都對整個高度包絡（桌面頂 0.45–0.75 m，每 0.05 m）與兩個瓶位核對碰撞，所以執行時不需要讀桌高。
視點（底盤停在同一觀察位姿，手臂從 V0＝Q_LOOK 出發）：
  V0      既有固定俯視（seg3d_collect_design 的 Q_LOOK）。
  V_LOW   缺底用：瞄準點降到 (0.075, 0.485, 0.55)，目標＝包絡內最差情況下兩瓶位瓶身（含瓶底）落在影像內的比例最大（再取相機離瓶較遠者），讓低桌時瓶底入畫。
  V_HIGH  缺頂用：瞄準點 (0.075, 0.485, 0.75)，目標＝包絡內「瓶頂上緣外側視線落在桌面或地面（有深度背景）」的案例數最多，再取俯角最大、相機最高；
          相機高度搜尋從 1.10 m 往下（更正 seg3d_collect_design 從 0.85 m 起搜的限制：那次只證明搜尋範圍內找到 0.85 m，不證明更高不可達）。
約束：IK 由 V0 延續、關節餘裕 ≥ 0.05、自碰 ≥ 8.39 mm、連桿點雲對包絡內每個桌高的桌面／桌腳與兩瓶位瓶身 ≥ 0.05 m；
  V0→候選、V_LOW→V_HIGH 的關節直線路徑（各 21 點）同樣滿足，並列為搜尋條件（停止路徑：手臂停在路徑上任一點，已涵蓋）。
輸出 results/vision/seg3d_view_library.json：各視點關節、相機位姿、俯角、過渡時長（1.875·max|Δq|/v_arm，關節空間直線所以端點 Δq 即沿路徑最大）、
  包絡內各桌高的可見性預估（同 seg3d_collect_design.visibility：只是幾何可能，不計遮擋）。
    python3 evaluation/bottle_seg3d_view_library.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_seg3d_collect_design as CD                # noqa: E402
import wrist_v0_capture as W0                           # noqa: E402

ENVELOPE = tuple(round(z, 2) for z in np.arange(0.45, 0.751, 0.05))
LOOK_LOW, LOOK_HIGH = np.array([0.075, 0.485, 0.55]), np.array([0.075, 0.485, 0.75])


def clear_env(q):
    worst = dict(env=9.9, self=9.9, margin=9.9)
    for tz in ENVELOPE:
        cl = CD.clearance(q, tz)
        worst = dict(env=min(worst['env'], cl['env']), self=min(worst['self'], cl['self']), margin=min(worst['margin'], cl['margin']))
    return worst


def ok_clear(w):
    return w['env'] >= CD.CLEAR and w['self'] >= 0.00839 and w['margin'] >= CD.JOINT_MARGIN


def path_ok(qa, qb, base, n=21):
    worst = dict(env=9.9, self=9.9, margin=9.9)
    for s in np.linspace(0.0, 1.0, n):
        w = clear_env(np.r_[base, qa + (qb - qa) * s])
        worst = {k: min(worst[k], w[k]) for k in worst}
        if not ok_clear(worst):
            return False, worst
    return True, worst


def search(base, q_seed, look_at, T_mount, T_eef_tcp, score, need_path_from=()):
    best = None
    cb, sb = math.cos(base[2]), math.sin(base[2])
    n = 0
    for zr in np.arange(1.10, 0.549, -0.05):
        for xr in np.arange(-0.10, 0.61, 0.10):
            for yr in np.arange(-0.10, 0.51, 0.10):
                cw = np.array([base[0] + cb * xr - sb * yr, base[1] + sb * xr + cb * yr, zr])
                Tc = CD.look_T(cw, look_at)
                T_tcp = Tc @ np.linalg.inv(T_mount) @ T_eef_tcp
                n += 1
                ok, _, qa, pe, re, marg = RS.ik(base, T_tcp, [q_seed], iters=150)
                if not ok or marg < CD.JOINT_MARGIN:
                    continue
                q = np.r_[base, qa]
                if not ok_clear(clear_env(q)):
                    continue
                Tr = CD.cam_T(q, T_mount)
                sc = score(Tr)
                if best is not None and sc <= best['score']:
                    continue
                if not all(path_ok(qf, qa, base)[0] for qf in need_path_from):   # 過渡可行（停止路徑＝停在路徑上任一點）也是候選條件
                    continue
                best = dict(score=sc, q=qa, cam=Tr, target_cam=cw.tolist())
    return best, n


def vis_all(Tc):
    return {f'{tz:.2f}_{xy[0]:.2f}_{xy[1]:.2f}': CD.visibility(Tc, xy, tz) for tz in ENVELOPE for xy in CD.BOTTLES}


def main():
    D = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json')))
    T_mount = W0.urdf_chain_T(CD.URDF, 'link_eef', 'camera_color_optical_frame')
    base = np.array(D['base_obs'])
    q_ready, q0 = np.array(D['q_ready']), np.array(D['q_look'])
    T_eef_tcp = np.linalg.inv(G.C.K.fk(np.r_[base, q_ready], 'link_eef')) @ G.C.K.fk(np.r_[base, q_ready], 'link_tcp')
    pitch = lambda Tc: math.degrees(math.asin(max(-1.0, min(1.0, -Tc[2, 2]))))   # noqa: E731

    def score_low(Tc):
        f = min(CD.visibility(Tc, xy, tz)['body_in_view'] for tz in ENVELOPE for xy in CD.BOTTLES)
        return (round(f, 2), float(np.linalg.norm(Tc[:3, 3] - LOOK_LOW)))

    def score_high(Tc):
        n_ok = sum(CD.visibility(Tc, xy, tz)['top_background'] in ('table_top', 'floor_or_beyond') for tz in ENVELOPE for xy in CD.BOTTLES)
        return (n_ok, round(pitch(Tc), 1), round(float(Tc[2, 3]), 3))
    lib = dict(V0=dict(q=q0, cam=CD.cam_T(np.r_[base, q0], T_mount)))
    out = dict(base=base.tolist(), envelope=ENVELOPE, bottles=CD.BOTTLES, views={})
    for nm, look, sc in (('V_LOW', LOOK_LOW, score_low), ('V_HIGH', LOOK_HIGH, score_high)):
        frm = (q0,) + ((lib['V_LOW']['q'],) if nm == 'V_HIGH' and 'V_LOW' in lib else ())
        cands, n = search(base, q0, look, T_mount, T_eef_tcp, sc, need_path_from=frm)
        if cands is None:
            print(f'{nm}：找不到（試 {n}）')
            continue
        lib[nm] = dict(q=cands['q'], cam=cands['cam'], tried=n, target_cam=cands['target_cam'])
    for nm, v in lib.items():
        out['views'][nm] = dict(q=v['q'].tolist(), cam_pos=v['cam'][:3, 3].tolist(), pitch_deg=pitch(v['cam']), clear=clear_env(np.r_[base, v['q']]), tried=v.get('tried'),
                                target_cam=v.get('target_cam'), vis=vis_all(v['cam']))
    tr = {}
    for a_, b_ in (('V0', 'V_LOW'), ('V0', 'V_HIGH'), ('V_LOW', 'V_HIGH')):
        if a_ in lib and b_ in lib:
            ok, w = path_ok(lib[a_]['q'], lib[b_]['q'], base)
            dq = float(np.abs(lib[b_]['q'] - lib[a_]['q']).max())
            tr[f'{a_}-{b_}'] = dict(ok=ok, worst=w, dq_max=dq, T=math.ceil(1.875 * dq / B.W_ARM['v_arm'] / 0.05) * 0.05)
    out['transitions'] = tr
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'seg3d_view_library.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for nm, v in out['views'].items():
        lo = min(x['body_in_view'] for x in v['vis'].values())
        tb = sorted({x['top_background'] for x in v['vis'].values()})
        hz = [round(x['cam_height_over_table'], 2) for k_, x in v['vis'].items() if k_.endswith('0.00_0.42')]
        print(f"{nm}：相機 {np.round(v['cam_pos'], 3).tolist()} 俯角 {v['pitch_deg']:.1f}°；包絡淨空 {v['clear']['env'] * 1e3:.0f} mm 自碰 {v['clear']['self'] * 1e3:.1f} mm；"
              f"包絡最差瓶身入畫 {lo:.2f}；瓶頂外緣背景 {tb}；相機高於各桌面 {hz}")
    for k_, v in tr.items():
        print(f"過渡 {k_}：可行 {v['ok']} 最差淨空 {v['worst']['env'] * 1e3:.0f} mm、Δq {v['dq_max']:.3f} → {v['T']:.2f} s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
