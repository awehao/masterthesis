#!/usr/bin/env python3
"""第八次第一包的瓶位範圍依據（離線，無模擬）：抓取車道固定 y＝0、偏航 0，第一桌桌面頂 0.60 m，
沿用實跑的備妥構型為唯一種子，對瓶位網格呼叫線上規劃工作者同一個 plan_grasp（預抓 0.06、抬升 0.05、相對窗 [−0.35, +0.15]）。
另列車道往桌子靠時，底盤中心對近側桌腳中心 (±0.33, 0.58) 的距離與追蹤器靜態避障半徑 0.52（0.05＋0.47）之差。
    python3 第八次進度報告素材/產生程式/reach_band_lane0.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'ammr_wholebody_mpc'))
import bottle_plan_worker as PW  # noqa: E402

Q_SEED = [0.3890373283602744, -0.19178500500278725, 1.054655868294558, -1.4408633140911007, -1.2030321024228567, 1.222226]   # bottle_f1c_2 sim.log 讀回 q_ready
LEG = (0.58, 0.52)


def main():
    grid = []
    for y in (0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52):
        for x in (-0.30, 0.0, 0.30):
            ok, why, res = PW.plan_grasp(dict(xy=[x, y], z_support=0.60, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], q_seed=Q_SEED, x_nom=0.0))
            grid.append(dict(x=x, y=y, ok=bool(ok), why=why, margin=None if res.get('margin') is None else round(res['margin'], 4)))
    legs = [dict(lane_y=l, d_leg=round(LEG[0] - l, 3), slack=round(LEG[0] - l - LEG[1], 3)) for l in (0.0, 0.02, 0.04, 0.06, 0.08)]
    out = dict(note='車道 y＝0 固定；x 只平移窗口所以結果與 x 無關（已核 −0.30／0／+0.30）；slack＝對近側桌腳靜態避障半徑的剩餘距離', grid=grid, legs=legs,
               y_ok=sorted({g['y'] for g in grid if g['ok']}), y_fail=sorted({g['y'] for g in grid if not g['ok']}))
    p = os.path.join(ROOT, '第八次進度報告素材', '資料', 'reach_band_lane0.json')
    json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(dict(y_ok=out['y_ok'], y_fail=out['y_fail'], legs=legs), ensure_ascii=False))


if __name__ == '__main__':
    main()
