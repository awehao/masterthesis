#!/usr/bin/env python3
"""0.75 m 量測目標的「真實時序」可達診斷（Codex 20261102_021353 第 2 點、023000：不以共同全窗或 L/T 判定；假世界閉迴路，非物理、不是可執行版本）。
在 test_bottle_w1_segtarget_vhigh 的閉迴路（V_HIGH 序列、原速 v_c 0.10、原啟發式調速、原 t_insert／閉爪／抬升時序）中，**只為診斷**旁路基線的全窗閘：
採用時照常算 window_branch 並記錄其結果（0.75 全窗無解），但不因此停止，讓同一控制器沿實際時序往下走（漸變→預抓→插入→閉爪→抬升→持瓶），看實際卡在哪裡。
另沿閉迴路紀錄的 (底盤 x(t), 當下相位 TCP 目標 T(t)) 逐拍做 IK 延續（由實測手臂起算、單步 ≤ 0.40 rad、餘裕 ≥ 0.05），報可達與否、關節速度。
報：結果／阻點、各相位時刻與底盤 x、插入到抬升期間底盤速度、設定點速率峰、關節餘裕、淨空（PCK）、IK 延續失敗處。
    python3 evaluation/bottle_vhigh_reach075.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], '0.75', 'nom']
sys.path.insert(0, HERE)
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402

LOG = []


class Diag(TV.VH.W1PlaceController):
    def _seg_freeze(self, xy, zs, seq):
        self.seg_z = zs
        T_g = self._Tg(xy)
        T_p = T_g.copy()
        T_p[:3, 3] -= self.task.pre_m * T_g[:3, 2]
        try:
            br = TV.VH.SG.BB.window_branch(T_p, T_g, TV.VH.SG.BRF.lift_pose(T_g, self.task.lift_m), self.win[0], self.win[1])
            wb = dict(ok=True, margin=float(br['margin']))
        except ValueError as e:
            wb = dict(ok=False, error=str(e))
        self.diag_window = wb
        self.task.T_grasp = T_g
        self.vfix = dict(t=self.t, xy=np.array(xy, float), spread=0.0, n=1, margin=float('nan'), seq=seq, z_support=zs)
        self.w_goal = np.array(xy, float)
        self._ev('segtarget_freeze_diag_bypass', window=wb)
        return True

    def step(self, t, q, **kw):
        out = super().step(t, q, **kw)
        if self.vfix is not None and self.st.phase in ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT') and (self.blending or self.st.phase != 'CRUISE') and self.n_tick % 2 == 0:
            T_seq = self._T_seq(np.asarray(q, float), self.t)
            LOG.append((float(t), self.st.phase, np.asarray(q, float).copy(), np.array(T_seq[0]), float(self.v_b), bool(self._rel_task())))
        return out


def main():
    TV.VH.W1PlaceController = Diag
    c, f, r, st = TV.run()
    why = (r['stop'] or {}).get('why')
    ev = [(e['t'], e.get('to') or e['what']) for e in r['events'] if e['what'] in ('phase', 'blend_start', 'segtarget_freeze_diag_bypass', 'close_request', 'stop', 'mission_done')]
    xs = {e['t']: None for e in r['events']}
    H = {round(h['t'], 2): h for h in f.hist}
    ph_x = [(t_, p_, round(float(H[round(t_, 2)]['x']), 4) if round(t_, 2) in H else None) for t_, p_ in ev]
    ins = [h for h in f.hist if h['phase'] in ('INSERT', 'CLOSE', 'LIFT')]
    v_ins = None if not ins else dict(min=round(min(h['v_b'] for h in ins), 4), max=round(max(h['v_b'] for h in ins), 4), x0=round(ins[0]['x'], 4), x1=round(ins[-1]['x'], 4),
                                      dur=round(ins[-1]['t'] - ins[0]['t'], 2))
    mins, sm, ns, rate, marg = PCK.check(f)
    # 沿紀錄的 (x(t), T(t)) 做 IK 延續（世界座標目標；持瓶段以外）
    ik_fail, q_prev, n_ok, vmax = None, None, 0, 0.0
    for t_, ph, q9, T, vb, rel in LOG:
        if rel:
            continue
        seed = q9[3:] if q_prev is None else q_prev
        ok, _, qa, pe, re, m = RS.ik(q9[:3], T, [seed], iters=200)
        if not ok or m < 0.05 or (q_prev is not None and float(np.abs(qa - q_prev).max()) > 0.40):
            ik_fail = dict(t=round(t_, 3), phase=ph, x=round(float(q9[0]), 4), ok=bool(ok), margin=round(float(m), 3), pe_mm=round(pe * 1e3, 2),
                           jump=None if q_prev is None else round(float(np.abs(qa - q_prev).max()), 3))
            break
        if q_prev is not None:
            vmax = max(vmax, float(np.abs(qa - q_prev).max()) / 0.02)
        q_prev, n_ok = qa, n_ok + 1
    out = dict(z1=0.75, outcome='完整抓放' if r['mission_done'] else f'停止 {why}', stop=r['stop'], window_branch=getattr(c, 'diag_window', None), events=ph_x, base_insert_lift=v_ins,
               rate_peak=round(rate, 3), joint_margin=round(marg, 3), self_min_mm=round(sm[0] * 1e3, 2), mins={k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items()},
               ik_along_timeline=dict(n_checked=n_ok, fail=ik_fail, v_joint_max=round(vmax, 3)), seq=r['vhigh_seq']['log'],
               note='只為診斷旁路全窗閘（不是可執行版本）；原速、原調速、原時序；假世界閉迴路')
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'vhigh_reach075_timeline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
