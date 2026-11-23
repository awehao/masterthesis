#!/usr/bin/env python3
"""視點庫 v2：替代 V_HIGH（Codex 20261101_232935 第 1 項；bottle_seg3d_view_library.py 的版本化延伸，v1 與 seg3d_ad_z075_p2b 的拒絕結果保留）。
v1 的 V_HIGH 只看一條代表射線、把「越過瓶頂後打到地面」一律當有效背景，沒有考慮深度裁切；封存位姿下那條視線到地面的光軸深度約 12.34 m > 10 m，實際無深度。
v2 的瓶頂可確認預估：
  1 瓶頂圓（16 點）全部投影在影像內、離邊界 ≥ 4 px（不截斷）；
  2 瓶頂輪廓上緣（影像中每一欄最上方的瓶點）外 3–6 px 的每一個像素，反投影成射線，與場景求最近交點：
    地面 z＝0、第一桌（桌面＋桌腳，包絡內的桌高）、第二桌（0.60 m）、紅色方塊障礙；交點的**光軸 Z**（相機座標 z）須在 0.05–10.0 m 內才算有效遠背景；
    同時須比瓶頂遠 ≥ 5 cm（同 edge_test 的 EDGE_FAR）；有效比例 ≥ 0.8 才算這個佈置「瓶頂可確認」。
  不延長深度範圍、不把未知改成背景。
選法：瞄準點 xy 固定在兩瓶位中點 (0.075, 0.485)、高度取 0.55–0.95（每 0.10）；對整個高度包絡（0.45–0.75，每 0.05 m）與兩個瓶位，「瓶頂可確認」的案例數最多，再取「瓶身全入畫」案例數、俯角；
  約束同 v1（IK 由 V0 延續、關節餘裕、自碰、包絡碰撞 ≥ 0.05 m、V0→候選與 V_LOW→候選路徑可行；回收經 V0）。
輸出 results/vision/seg3d_view_library2.json（V0、V_LOW 照抄 v1，V_HIGH 換成新候選；transitions 重算），可直接給 isaac_bottle_seg3d_collect2 --library。
    python3 evaluation/bottle_seg3d_view_library2.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_place_ctrl as PCm                         # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_seg3d_collect_design as CD                # noqa: E402
import bottle_seg3d_view_library as VL                  # noqa: E402
import wrist_v0_capture as W0                           # noqa: E402

Z_MIN, Z_MAX, EDGE_FAR, EDGE_FRAC, BORDER = 0.05, 10.0, 0.05, 0.8, 4
OB_C, OB_S = np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50])
T2D = np.array([PCm.PLACE['target'][0] - 0.0, PCm.PLACE['lane_y'], 0.0])


def scene_boxes(tz):
    b = [(c_, s_) for _, c_, s_ in CD.table_boxes(tz)] + [(c_ + T2D, s_) for _, c_, s_ in G.table_boxes()] + [(OB_C, OB_S)]
    return [(np.asarray(c_, float) - np.asarray(s_, float) / 2, np.asarray(c_, float) + np.asarray(s_, float) / 2) for c_, s_ in b]


def ray_hit(o, d, boxes):
    """射線最近正交點參數 t（地面 z＝0 與軸對齊盒）；沒有 ⇒ inf。"""
    best = math.inf
    if d[2] < -1e-9:
        best = -o[2] / d[2]
    for lo, hi in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tmin, tmax = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
        if tmax >= max(tmin, 1e-6) and tmin < best:
            best = tmin if tmin > 1e-6 else best
    return best


def top_check(Tc, xy, tz):
    """回傳 (瓶頂圓不截斷, 上緣外側有效遠背景比例, 樣本數, 瓶身全入畫)。"""
    r, h = G.BOTTLE['r'], G.BOTTLE['h']
    a = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    top = np.array([[xy[0] + r * math.cos(t), xy[1] + r * math.sin(t), tz + h] for t in a])
    body = np.array([[xy[0] + r * math.cos(t), xy[1] + r * math.sin(t), tz + z] for z in np.linspace(0, h, 6) for t in a[::2]])
    Ti = np.linalg.inv(Tc)
    Pc = (Ti[:3, :3] @ top.T).T + Ti[:3, 3]
    if (Pc[:, 2] <= Z_MIN).any():
        return False, 0.0, 0, False
    u = CD.FX * Pc[:, 0] / Pc[:, 2] + CD.W_ / 2
    v = CD.FX * Pc[:, 1] / Pc[:, 2] + CD.H_ / 2
    uncut = bool(((u >= BORDER) & (u < CD.W_ - BORDER) & (v >= BORDER) & (v < CD.H_ - BORDER)).all())
    body_in = bool(CD.project(Tc, body)[0].all())
    # 瓶頂輪廓上緣：每一欄取瓶頂圓投影中 v 最小者（影像上方）；外 3–6 px
    cols = {}
    for ui, vi, zi in zip(u, v, Pc[:, 2]):
        k = int(round(ui))
        if k not in cols or vi < cols[k][0]:
            cols[k] = (vi, zi)
    boxes = scene_boxes(tz)
    R, c = Tc[:3, :3], Tc[:3, 3]
    n_ok, n = 0, 0
    for k, (v0, z_obj) in cols.items():
        for dv in (3, 4, 5, 6):
            vv = v0 - dv
            if vv < 0:
                n += 1
                continue
            dc = np.array([(k - CD.W_ / 2) / CD.FX, (vv - CD.H_ / 2) / CD.FX, 1.0])
            dw = R @ dc
            t = ray_hit(c, dw, boxes)
            n += 1
            if math.isfinite(t):
                zc = t * dc[2]                                      # 光軸 Z（dc 的 z 分量為 1 ⇒ zc＝t）
                if Z_MIN <= zc <= Z_MAX and zc >= z_obj + EDGE_FAR:
                    n_ok += 1
    return uncut, (n_ok / n if n else 0.0), n, body_in


def main():
    L1 = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_view_library.json')))
    D = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json')))
    T_mount = W0.urdf_chain_T(CD.URDF, 'link_eef', 'camera_color_optical_frame')
    base = np.array(L1['base'])
    q_ready = np.array(D['q_ready'])
    q0, q_low = np.array(L1['views']['V0']['q']), np.array(L1['views']['V_LOW']['q'])
    T_eef_tcp = np.linalg.inv(G.C.K.fk(np.r_[base, q_ready], 'link_eef')) @ G.C.K.fk(np.r_[base, q_ready], 'link_tcp')
    pitch = lambda Tc: math.degrees(math.asin(max(-1.0, min(1.0, -Tc[2, 2]))))   # noqa: E731
    cases = [(tz, xy) for tz in VL.ENVELOPE for xy in CD.BOTTLES]

    def evals(Tc):
        out = []
        for tz, xy in cases:
            uncut, frac, n, body_in = top_check(Tc, xy, tz)
            out.append(dict(tz=tz, xy=xy, uncut=uncut, frac=frac, n=n, body_in=body_in, top_ok=bool(uncut and frac >= EDGE_FRAC)))
        return out

    def score(Tc):
        E = evals(Tc)
        return (sum(e['top_ok'] for e in E), sum(e['body_in'] for e in E), round(pitch(Tc), 1))
    # v1 封存 V_HIGH 的檢核（說明為何失敗）
    T_old = CD.cam_T(np.r_[base, np.array(L1['views']['V_HIGH']['q'])], T_mount)
    E_old = evals(T_old)
    best, n_try = None, 0
    for look_z in (0.55, 0.65, 0.75, 0.85, 0.95):                       # 共同瞄準點 xy 固定在兩瓶位中點，只放寬瞄準高度（同一套選法，非依佈置）
        look = np.array([0.075, 0.485, look_z])
        cand, n = VL.search(base, q0, look, T_mount, T_eef_tcp, score, need_path_from=(q0, q_low))
        n_try += n
        if cand is not None and (best is None or cand['score'] > best['score']):
            best = dict(cand, look=look.tolist())
    out = dict(base=base.tolist(), envelope=VL.ENVELOPE, bottles=CD.BOTTLES, depth_range=[Z_MIN, Z_MAX], edge_far=EDGE_FAR, edge_frac=EDGE_FRAC, n_tried=n_try,
               v1_high_check=dict(top_ok=sum(e['top_ok'] for e in E_old), cases=len(E_old), detail=E_old), views={}, transitions={})
    views = dict(V0=q0, V_LOW=q_low)
    if best is not None:
        views['V_HIGH'] = best['q']
    for nm, q in views.items():
        Tc = CD.cam_T(np.r_[base, q], T_mount)
        E = evals(Tc)
        out['views'][nm] = dict(q=q.tolist(), cam_pos=Tc[:3, 3].tolist(), pitch_deg=pitch(Tc), clear=VL.clear_env(np.r_[base, q]), top_ok=sum(e['top_ok'] for e in E),
                                body_in=sum(e['body_in'] for e in E), cases=len(E), detail=E, vis=VL.vis_all(Tc), look=best['look'] if nm == 'V_HIGH' else None)
    for a_, b_ in (('V0', 'V_LOW'), ('V0', 'V_HIGH'), ('V_LOW', 'V_HIGH')):
        if a_ in views and b_ in views:
            ok, w = VL.path_ok(views[a_], views[b_], base)
            dq = float(np.abs(views[b_] - views[a_]).max())
            out['transitions'][f'{a_}-{b_}'] = dict(ok=ok, worst=w, dq_max=dq, T=math.ceil(1.875 * dq / B.W_ARM['v_arm'] / 0.05) * 0.05)
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'seg3d_view_library2.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    o = out['v1_high_check']
    print(f"v1 封存 V_HIGH：瓶頂可確認 {o['top_ok']}／{o['cases']}；0.75/P2 外側有效比例 {[round(e['frac'], 2) for e in o['detail'] if e['tz'] == 0.75 and e['xy'] == (0.15, 0.55) or (e['tz'] == 0.75 and list(e['xy']) == [0.15, 0.55])]}")
    for nm, v in out['views'].items():
        d75 = [e for e in v['detail'] if e['tz'] == 0.75]
        print(f"{nm}：相機 {np.round(v['cam_pos'], 3).tolist()} 俯角 {v['pitch_deg']:.1f}°；瓶頂可確認 {v['top_ok']}／{v['cases']}、瓶身全入畫 {v['body_in']}／{v['cases']}；包絡淨空 {v['clear']['env'] * 1e3:.0f} mm；"
              f"0.75 m：{[(list(e['xy']), e['uncut'], round(e['frac'], 2)) for e in d75]}")
    for k_, v in out['transitions'].items():
        print(f"過渡 {k_}：可行 {v['ok']}、最差淨空 {v['worst']['env'] * 1e3:.0f} mm、{v['T']:.2f} s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
