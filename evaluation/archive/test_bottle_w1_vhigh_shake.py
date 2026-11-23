#!/usr/bin/env python3
"""懸持中側向擾動實驗（bottle_w1_vhigh_shake_ctrl）的起跑前離線核對（Codex 20261102_035741 第 5 項；假世界、無滑移模型，只驗動作與安全，不預測滑移）。
沿 test_bottle_w1_segtarget_vhigh 的 0.60 名目閉迴路（候選只在 V_HIGH 段送），控制器換成擾動版。核：
  1 流程：穩定懸持 → 懸持 +4.5 s 起 4 s 擾動 → 懸持 13 s 結束 ⇒ hold_exp_done 受控停止；不送持瓶路徑請求。
  2 擾動實現（底盤執行）：TCP（世界座標，實測）沿閉合軸投影的位移峰峰值、對預定剖面的追蹤誤差、速度與加速度峰；擾動期間底盤實測最大速度；手臂命令變化率（0.05 s 網格）。
  3 停止軌跡：擾動期間每個控制拍（_advance_inner、相位 0）以 compactcarry21.stop_traj 算「現在停／延後一拍停」完整停止軌跡（在途命令、設定點積分、有界減速、收斂尾段，0.04 s 取樣），
    每個構型核：非手指連桿與手指對第一桌（桌面＋桌腳）、夾持中的瓶（瓶座標點雲經 judge.T_rel）對第一桌、自碰（開爪 8.017 mm 已驗證下限；閉爪時另報實測值）、關節餘裕 ≥ 0.05。
    門檻：連桿與瓶對第一桌 ≥ 0.01 m（持瓶淨空拒絕線 5.5 mm 的保守上方）、非有限 0、未停完 0。
    python3 evaluation/test_bottle_w1_vhigh_shake.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], '0.60', 'nom']
sys.path.insert(0, HERE)
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_w1_vhigh_shake_ctrl as SK                 # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import test_bottle_w1 as T                              # noqa: E402

TB = [(c_, s_) for _, c_, s_ in G.table_boxes()]
BPT = G.bottle_points()
ST = dict(n=0, n_cfg=0, worst={}, where={}, nonfinite=0, unfinished=0)
U = []


def chk(q9, T_rel, f_cmd):
    W = G.link_world(q9, f_cmd)
    P_nf = np.vstack([v for k, v in W.items() if k not in G.C.FINGERS])
    P_f = np.vstack([v for k, v in W.items() if k in G.C.FINGERS])
    out = dict(nonfinger_table=min(float(G.box_dist(P_nf, c_, s_).min()) for c_, s_ in TB), finger_table=min(float(G.box_dist(P_f, c_, s_).min()) for c_, s_ in TB),
               self=float(G.C.self_clear(q9, f_cmd)[0]), margin=BB.margin(q9[3:]))
    if T_rel is not None:
        Tb = G.C.K.fk(q9, 'link_tcp') @ T_rel
        Pb = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
        out['bottle_table'] = min(float(G.box_dist(Pb, c_, s_).min()) for c_, s_ in TB)
    return out


class Ctl(SK.W1PlaceController):
    def _advance_inner(self, T_tcp, close_done, q):
        sh = self.shake
        if sh['t0'] is not None and sh['t0'] <= self.t <= sh['t0'] + SK.T_SHAKE and self.judge.stop is None and np.isfinite(q).all():
            ST['n'] += 1
            for e in (0, 1):
                Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
                if len(Q) >= 2000 // C21.SAMPLE_EVERY + 1:
                    ST['unfinished'] += 1
                if not np.isfinite(Q).all():
                    ST['nonfinite'] += 1
                    continue
                for qq in Q:
                    ST['n_cfg'] += 1
                    for k, v in chk(qq, self.judge.T_rel, self.f_cmd).items():
                        key = f'e{e}:{k}'
                        if v < ST['worst'].get(key, 9.9):
                            ST['worst'][key], ST['where'][key] = v, round(float(self.t), 3)
        super()._advance_inner(T_tcp, close_done, q)

    def _solve(self, q):
        u = super()._solve(q)
        sh = self.shake
        if u is not None and sh['t0'] is not None and sh['t0'] - 0.5 <= self.t <= sh['t0'] + SK.T_SHAKE + 0.5:
            U.append((float(self.t), np.asarray(u, float).copy()))
        return u


def main():
    TV.VH.W1PlaceController = Ctl
    c, f, r, st = TV.run()
    why = (r['stop'] or {}).get('why')
    sh = r['shake_exp']
    ax = np.array(sh['axis_base'])
    t0 = sh['t0']
    H = [x for x in f.hist if t0 - 0.5 <= x['t'] <= t0 + SK.T_SHAKE + 0.5]
    tt = np.array([x['t'] for x in H])
    th0 = float(H[0]['q'][2])
    axw = np.array([np.cos(th0) * ax[0] - np.sin(th0) * ax[1], np.sin(th0) * ax[0] + np.cos(th0) * ax[1], 0.0])
    P = np.array([G.C.K.fk(x['q'], 'link_tcp')[:3, 3] for x in H])                       # 世界座標 TCP（底盤執行擾動）
    proj = (P - P[0]) @ axw
    plan = np.array([SK.disp(x - t0) for x in tt])
    vb = float(max(np.hypot(*(H[k]['q'][:2] - H[k - 1]['q'][:2])) / 0.01 for k in range(1, len(H))))
    Vt = np.gradient(proj, tt)
    At = np.gradient(np.convolve(Vt, np.ones(5) / 5, mode='same'), tt)
    du = [float(np.abs(U[i][1][3:] - U[i - 1][1][3:]).max() / (U[i][0] - U[i - 1][0])) for i in range(1, len(U)) if 0 < U[i][0] - U[i - 1][0] <= 0.051]
    lim = dict(nonfinger_table=0.01, finger_table=0.01, bottle_table=0.01, self=0.0, margin=0.05)
    fails = sorted(k for k, v in ST['worst'].items() if v < lim[k.split(':')[1]]) + (['nonfinite'] if ST['nonfinite'] else []) + (['unfinished'] if ST['unfinished'] else [])
    ev = {e['what']: e['t'] for e in r['events']}
    out = dict(outcome=f'停止 {why}', stop=r['stop'], t_carry_request=ev.get('carry_plan_request'), hold=r['hold_exp'], shake_axis_base=sh['axis_base'], t0=t0,
               shake_actual_mm=dict(pp=round(float(proj.max() - proj.min()) * 1e3, 2), max=round(float(proj.max()) * 1e3, 2), min=round(float(proj.min()) * 1e3, 2)),
               shake_plan_pp_mm=round(float(plan.max() - plan.min()) * 1e3, 2), track_err_max_mm=round(float(np.abs(proj - plan).max()) * 1e3, 2),
               base_v_max_in_shake=round(vb, 5), tcp_v_peak=round(float(np.abs(Vt).max()), 3), tcp_a_peak_smooth5=round(float(np.abs(At).max()), 3), du_dt_arm_max=round(max(du), 3) if du else None, A_ARM=C0.A_ARM,
               stop_traj=dict(n_ticks=ST['n'], n_cfg=ST['n_cfg'], sample_dt_s=C21.SAMPLE_EVERY * C0.PHYS, worst={k: round(v, 5) for k, v in ST['worst'].items()}, where=ST['where'],
                              nonfinite=ST['nonfinite'], unfinished=ST['unfinished'], limits=lim, fails=fails),
               finite=T.finite_cmds(f))
    out['pass'] = bool(why == 'hold_exp_done' and out['t_carry_request'] is None and not fails and out['finite'] and out['shake_actual_mm']['pp'] > 0.0)
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'vhigh_shake_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('hold',)}, ensure_ascii=False, default=float))
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
