#!/usr/bin/env python3
"""候選轉接的失效關閉與完成通知核對（Codex 20261102_003740；無 ROS、無模擬：假節點直接呼叫訂閱回呼；採用判定用真控制器 candidate_intake）。
  1 t_eval＝NaN ⇒ 拒收且不汙染接收進度（下一筆合格照收）；2 欄位數符合標籤但缺 k_snap_ik ⇒ 拒收（不拋 KeyError）；3 狀態碼 2.5、7 ⇒ 拒收（不截斷）；
  4 未來 t_eval ⇒ 拒收；5 尚無快照（seq 0、欄位 NaN）⇒ 不轉送、不拋例外；6 快照來源戳逾 3 s ⇒ 拒收；
  7 完成後沒有下一張影像：候選列 pending（新鮮）→ 0.1 s 後完成通知（ok、同目標）⇒ 補送並由控制器採用（vfix 成立）；
  8 完成時觀測已逾期：候選列 pending → 1.0 s 後才完成通知 ⇒ 補送但控制器以 stale 拒收（未採用）；9 通知目標不符 ⇒ 不配對；
  10 分支無解（目標 y＋0.5 m）⇒ segtarget_unreachable 受控停止、保留未採用（T_grasp 為 NaN）。
    python3 evaluation/test_bottle_segtarget_link_unit.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_segtarget_link as SL                     # noqa: E402
import bottle_w1_segtarget_ctrl as SG                   # noqa: E402
import test_bottle_w1 as T                              # noqa: E402

BAD = []
FIELDS = ['t_eval', 'src_stamp', 'n_rows', 'eval_ms', 'n_check_run'] + [f'k_{x}' for x in ('ready', 'why', 'cx', 'cy', 'z_support', 'z_top', 'z_grasp', 'n', 'age_s', 'snap_seq',
                                                                                        'snap_t_first', 'snap_t_last', 'snap_n', 'snap_cx', 'snap_cy', 'snap_z_support', 'snap_z_grasp', 'snap_ik',
                                                                                        'snap_local', 'snap_local_min', 'snap_age_s', 'snap_dev_mm', 'snap_consistent')] + ['schema']
LABEL = 'bottle_target_candidates schema=3 frame=odom(world, z up, m) stamps=ros_sim_clock fields=' + ','.join(FIELDS)
CHK_F = ['t_publish', 'kind', 'seq', 't_first', 't_last', 'n', 'cx', 'cy', 'z_support', 'z_grasp', 'ik', 'local', 'local_min', 'schema']
CHK_LABEL = 'bottle_target_candidate_checks schema=3 frame=odom(world, z up, m) fields=' + ','.join(CHK_F)


def chk(name, cond, info=''):
    if not cond:
        BAD.append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


class FakeNode:
    def __init__(self):
        self.cb = {}

    def create_subscription(self, typ, topic, cb, depth):
        self.cb[topic] = cb


class Msg:
    def __init__(self, label, data):
        self.layout = type('L', (), dict(dim=[type('D', (), dict(label=label))()] if label else []))()
        self.data = data


def row(t_eval, seq=1, ik=1, cx=0.0, cy=0.42, z=0.60, age=0.05, ready=1, labels=FIELDS, drop=None, **over):
    d = {f: 0.0 for f in FIELDS}
    d.update(t_eval=t_eval, src_stamp=t_eval - age, k_ready=ready, k_age_s=age, k_snap_seq=seq, k_snap_t_last=t_eval - age, k_snap_cx=cx, k_snap_cy=cy, k_snap_z_support=z,
             k_snap_ik=ik, k_snap_local=ik, k_snap_dev_mm=0.1, k_snap_consistent=1.0, schema=3.0)
    d.update(over)
    fs = [f for f in labels if f != drop]
    if drop:
        fs.append('k_extra')
        d['k_extra'] = 0.0
    lab = 'bottle_target_candidates schema=3 frame=odom(world, z up, m) fields=' + ','.join(fs)
    return Msg(lab, [float(d[f]) for f in fs])


def check(t_pub, seq=1, ik=2, cx=0.0, cy=0.42, z=0.60):
    d = dict(t_publish=t_pub, kind=1.0, seq=seq, t_first=0.0, t_last=0.0, n=3.0, cx=cx, cy=cy, z_support=z, z_grasp=z + 0.1, ik=ik, local=ik, local_min=0.06, schema=3.0)
    return Msg(CHK_LABEL, [float(d[f]) for f in CHK_F])


def fresh_link(sink):
    n = FakeNode()
    L = SL.SegTargetLink(n, None, None, sink=sink)
    L.set_T0(0.0)
    return n, L


def ctrl():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = SG.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    return c


def main():
    got = []
    n, L = fresh_link(lambda t, c: got.append(c))
    cb = n.cb['/bottle/target_candidates']
    cb(row(float('nan')))
    L.tick(10.0)
    cb(row(9.0))
    L.tick(10.0)
    chk('1 t_eval＝NaN 拒收且不汙染接收進度（下一筆照收）', L.rej.get('time_nonfinite') == 1 and len(got) == 1 and L.last_eval == 9.0, (L.rej, len(got), L.last_eval))
    cb(row(9.2, drop='k_snap_ik'))
    L.tick(10.0)
    chk('2 缺 k_snap_ik（欄位數符合標籤）⇒ 拒收、無例外', L.rej.get('missing_field') == 1 and len(got) == 1, L.rej)
    cb(row(9.3, ik=2.5))
    cb(row(9.4, k_snap_local=7.0))
    L.tick(10.0)
    chk('3 狀態碼 2.5、7 ⇒ 拒收（不截斷）', L.rej.get('bad_state_code') == 2 and len(got) == 1, L.rej)
    cb(row(12.0))
    L.tick(10.0)
    chk('4 未來 t_eval ⇒ 拒收', L.rej.get('time_future') == 1 and len(got) == 1, L.rej)
    nan = float('nan')
    cb(row(9.5, seq=0, k_snap_t_last=nan, k_snap_cx=nan, k_snap_cy=nan, k_snap_z_support=nan, k_age_s=nan))
    L.tick(10.0)
    chk('5 尚無快照（seq 0、NaN）⇒ 不轉送、無例外', L.rej.get('no_snapshot_yet') == 1 and len(got) == 1, L.rej)
    cb(row(9.6, k_snap_t_last=5.0))
    L.tick(10.0)
    chk('6 快照來源戳逾 3 s ⇒ 拒收', L.rej.get('snapshot_too_old') == 1 and len(got) == 1, L.rej)
    # 7／8／9 用真控制器
    for nm, dt_chk, cx_chk, want_fix in (('7 完成後沒有下一張影像（0.1 s 後通知）⇒ 補送並採用', 0.1, 0.0, True),
                                          ('8 完成時觀測已逾期（1.0 s 後通知）⇒ 補送但 stale 不採用', 1.0, 0.0, False),
                                          ('9 通知目標不符 ⇒ 不配對、不採用', 0.1, 0.003, False)):
        c = ctrl()
        c.t = 9.0
        n2, L2 = fresh_link(c.candidate_intake)
        n2.cb['/bottle/target_candidates'](row(9.0, ik=1))
        L2.tick(9.0)
        c.t = 9.0 + dt_chk
        n2.cb['/bottle/target_candidate_checks'](check(9.0 + dt_chk - 0.01, cx=cx_chk))
        L2.tick(9.0 + dt_chk)
        print(f"   {nm[:2]}：補送 {L2.n_merged}、控制器拒收 {c.seg_rej}、凍結 {c.vfix is not None}、轉接拒收 {L2.rej}")
        chk(nm, (c.vfix is not None) == want_fix and ((L2.n_merged == 1) if cx_chk == 0.0 else (L2.n_merged == 0 and L2.rej.get('chk_target_mismatch') == 1))
            and (want_fix or cx_chk != 0.0 or c.seg_rej.get('stale', 0) >= 1), (c.vfix, c.seg_rej, L2.rej))
    # 10 分支無解
    c = ctrl()
    c.t = 9.0
    c.candidate_intake(9.0, dict(ready=1.0, age_s=0.05, snap_seq=1.0, snap_t_last=8.95, snap_cx=0.0, snap_cy=0.92, snap_z_support=0.60, snap_ik=2.0, snap_local=2.0,
                                 snap_dev_mm=0.1, snap_consistent=1.0))
    why = (c.judge.stop or {}).get('why')
    chk('10 分支無解 ⇒ segtarget_unreachable 受控停止、保留未採用（T_grasp NaN、vfix None）', why == 'segtarget_unreachable' and c.vfix is None and np.isnan(c.task.T_grasp).all(),
        (c.judge.stop, c.vfix))
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
