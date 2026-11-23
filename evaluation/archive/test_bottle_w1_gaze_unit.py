#!/usr/bin/env python3
"""bottle_w1_gaze_ctrl 的函式層單元核對（不跑假世界）：
1 T_MOUNT＝bottle_v1_cam_visibility 手腕外參＝URDF link_eef→camera_color_optical_frame。
2 track：500 組隨機目標與中途換目標，每拍 |ġ| ≤ V_G、|g̈| ≤ A_G；固定目標從靜止出發不越過；最後確切落在目標且 ġ＝0。
3 gaze_target：V_HIGH、底盤在原區域 U3 的 H 段實錄位姿（x −1.40、y −0.71；觀看點 u≈349，在帶內）⇒ 0；R-B1 型位置（y＋0.33、x −2.2）⇒ 解出負偏移，解後水平像素在帶邊（殘差 < 1 px）。
4 preview：前推 N＋10 拍與逐拍 track 相同。
5 移動目標（300 組斜坡目標、中途反向）：g 的二階差分／dt² ≤ A_G（越過照實保留、不拉回造成跳動）。
    python3 evaluation/test_bottle_w1_gaze_unit.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_v1_cam_visibility as VZ                   # noqa: E402
import bottle_w1_gaze_ctrl as GZ                        # noqa: E402
import wrist_v0_capture as W0                           # noqa: E402


def main():
    res = {}
    Tu = W0.urdf_chain_T(os.path.join(HERE, 'models', 'omni_bot_manip.urdf'), 'link_eef', 'camera_color_optical_frame')
    res['mount'] = bool(np.allclose(GZ.T_MOUNT, VZ.CAMS['wrist']['T'], atol=1e-9) and np.allclose(GZ.T_MOUNT, Tu, atol=1e-6))
    rng = np.random.default_rng(0)
    ok, amax, vmax = True, 0.0, 0.0
    for trial in range(500):
        g = v = 0.0
        t1, k_sw = rng.uniform(-0.9, 0.9), int(rng.integers(5, 60))
        t2 = rng.uniform(-0.9, 0.9) if trial % 2 else 0.0
        conv = False
        for k in range(400):
            g2, v2 = GZ.track(g, v, t1 if k < k_sw else t2, 0.05)
            amax, vmax = max(amax, abs(v2 - v) / 0.05), max(vmax, abs(v2))
            g, v = g2, v2
            if k > k_sw and g == t2 and v == 0.0:
                conv = True
                break
        ok &= conv
    over = 0.0
    for tg in np.linspace(-0.9, 0.9, 37):
        g = v = 0.0
        for _ in range(200):
            g, v = GZ.track(g, v, tg, 0.05)
            over = max(over, (g - tg) * np.sign(tg) if tg != 0 else abs(g))
        ok &= (g == tg and v == 0.0)
    res['track'] = dict(converge_exact=bool(ok), v_max=vmax, a_max=amax, overshoot_from_rest=over, pass_=bool(ok and vmax <= GZ.V_G + 1e-12 and amax <= GZ.A_G + 1e-9 and over == 0.0))
    K = G.C.K
    qv = np.array(json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_view_library.json')))['views']['V_HIGH']['q'], float)
    d0, w0, _ = GZ.gaze_target(K, np.array([-1.40, -0.71, 0.0]), qv, 0.0)
    d1, w1, r1 = GZ.gaze_target(K, np.array([-2.2, 0.33, 0.0]), qv, 0.0)
    p = GZ.look_cam(K, np.array([-2.2, 0.33, 0.0]), qv, d1)
    u = GZ.FX * p[0] / p[2] + 320
    res['target'] = dict(u3_h=(d0, w0), rb1=(d1, w1, r1), rb1_u_after=u, pass_=bool(d0 == 0.0 and w0 == 'in_band' and w1 == 'solved' and d1 < 0 and abs(r1) < 1.0 and abs(abs(u - 320) - GZ.BAND_PX) < 1.0))
    P = GZ.preview(0.1, -0.2, -0.6, 30, 0.05)
    g, v, Q = 0.1, -0.2, [0.1]
    for _ in range(30):
        g, v = GZ.track(g, v, -0.6, 0.05)
        Q.append(g)
    res['preview'] = bool(np.array_equal(P, np.array(Q)))
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        g = v = 0.0
        Gs = [0.0]
        tg, rate = rng.uniform(-0.9, 0.9), rng.uniform(-0.8, 0.8)
        for k in range(200):
            g, v = GZ.track(g, v, float(np.clip(tg + rate * 0.05 * k * (1 if k < 100 else -1), -0.9, 0.9)), 0.05)
            Gs.append(g)
        worst = max(worst, float((np.abs(np.diff(np.array(Gs), 2)) / 0.05 ** 2).max()))
    res['moving'] = dict(dd_max=worst, pass_=bool(worst <= GZ.A_G + 1e-6))
    allok = res['mount'] and res['track']['pass_'] and res['target']['pass_'] and res['preview'] and res['moving']['pass_']
    print(json.dumps(res, ensure_ascii=False, default=float))
    print('全過' if allok else '未過')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
