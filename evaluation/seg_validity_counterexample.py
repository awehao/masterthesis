#!/usr/bin/env python3
"""抓取目標有效期政策反例（Codex 20261111_142637 第 2 項）：真 candidate_intake 鏈（bottle_w1_obs_ctrl → online2 → segtarget 基線）與真期限檢查
（bottle_w1_segtarget_ctrl._advance_inner，其下內層隔離）。只把抓取計畫工作者（_gp_submit）與計畫接受的凍結（_gp_apply 基線部分：排程平移、第一桌設定）換成輕量替身，其餘採用規則照基線。
時間為任務時刻；候選欄位同 SegTargetLink 轉送格式。
案例：首次合格快照 → 抓取計畫被接受才凍結（線上版；有效起點＝凍結快照的來源擷取時刻）→ 新合格快照刷新 → 重複通知／新序號但舊影格／與快照不一致／未就緒／檢查未過／只有導引 xy（底盤）→ 不刷新
  → 鎖定後新合格快照不刷新 → 停止觀測：在「最後有效來源時刻＋15 s」前不停、之後 segtarget_expired（並列舊規則＝首次凍結＋15 s）。
    python3 evaluation/seg_validity_counterexample.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ros_chain as RC                                           # noqa: E402

OC, = RC.import_with_real_ros('bottle_w1_obs_ctrl')
import bottle_s0_logic as L                                             # noqa: E402
import bottle_w1_segtarget_ctrl as SG                                   # noqa: E402

C = OC.W1PlaceController
MRO = {k.__module__: k for k in C.__mro__}
nxt = [k for k in C.__mro__ if '_advance_inner' in k.__dict__]
below = nxt[nxt.index(MRO['bottle_w1_segtarget_ctrl']) + 1]
below._advance_inner = lambda self, T, c, q: None                       # 隔離：segtarget 以下的內層相位推進


ON = MRO['bottle_w1_online_ctrl']


def light_submit(self, xy, zs, seq):                                    # 替身：抓取計畫工作者（只記 pending）
    self.gp = dict(state='pending', id=1, xy=np.asarray(xy, float).copy(), zs=float(zs), seq=int(seq))


def light_apply(self, g, res, q):                                       # 替身：線上版 _gp_apply 的凍結部分（不做排程平移、第一桌設定）
    xy, zs, seq = g['xy'], g['zs'], g['seq']
    self.seg_z = zs
    self.vfix = dict(t=self.t, xy=np.array(xy, float), spread=0.0, n=1, margin=0.46, seq=seq, z_support=zs)
    self.w_goal = np.array(xy, float)
    self.seg_seq = seq
    self.seg_adopt.append((self.t, seq, list(map(float, xy)), zs))
    g.update(state='accepted', t_acc=float(self.t))
    self._ev('segtarget_freeze', xy=[round(float(v), 5) for v in xy], z_support=round(zs, 5), seq=seq, margin=0.46)


ON._gp_submit = light_submit
ON._gp_apply = light_apply
MRO['bottle_w1_tpa_ctrl']._gp_apply = light_apply                      # 隔離：TPA 起動與 R1a 預抓分支設定（obs 的 _gp_apply 覆寫照真碼，super 直達替身）


def mk():
    o = C.__new__(C)
    o.t = 7.4
    o.judge = L.S0Judge.__new__(L.S0Judge)
    o.judge.stop, o.judge.log = None, []
    o.events = []
    o.seg = dict(SG.SEGREG)
    o.seg_log, o.seg_rej, o.seg_seq, o.seg_z, o.seg_adopt, o.seg_obs = [], {}, 0, None, [], []
    o.vfix, o.w_goal, o.w_locked, o._cand, o.obs_last, o.obs_first = None, None, False, None, None, None
    o.gp, o.tlane, o._gp_defer, o._wrist_src = None, dict(y_lane=0.008), None, None    # 車道已成立（站位完成）⇒ 抓取計畫不延後
    o.blending, o.t_close_req, o.x_blend = True, None, -0.62
    o.guide, o.guide_n, o.guide_block = None, 0, None
    return o


def cand(seq, t_last, xy=(-0.0227, 0.4083), zs=0.58, ready=1.0, ik=2.0, local=2.0, age=0.06, cons=1.0, dev=0.4):
    return dict(ready=ready, age_s=age, snap_seq=float(seq), snap_t_last=t_last, snap_cx=xy[0], snap_cy=xy[1], snap_z_support=zs, snap_ik=ik, snap_local=local,
                snap_dev_mm=dev, snap_consistent=cons)


def main():
    o = mk()
    steps = []

    def feed(t, c, what):
        o.t = t
        sv0 = None if getattr(o, 'seg_valid', None) is None else dict(o.seg_valid)
        o.candidate_intake(t, c)
        sv1 = getattr(o, 'seg_valid', None)
        steps.append(dict(t=t, what=what, refreshed=(sv1 is not None and sv1 != sv0), valid_t_src=None if sv1 is None else sv1['t_src'], seq=o.seg_seq,
                          stop=None if o.judge.stop is None else o.judge.stop['why'], rej=dict(o.seg_rej)))
    feed(7.18, cand(1, 7.05), '首次合格快照（來源 7.05）⇒ 送抓取計畫（線上版此時不凍結）')
    o.t = 7.40
    o._gp_apply(o.gp, dict(margin=0.46), np.array([-2.75, 0.0, 0.0]))   # 計畫被接受 ⇒ 凍結；有效起點＝凍結快照來源時刻 7.05
    steps.append(dict(t=7.40, what='計畫接受 ⇒ 凍結（vfix 7.40）＋有效起點＝快照來源 7.05', refreshed=True, valid_t_src=o.seg_valid['t_src'], seq=o.seg_seq, stop=None))
    feed(10.30, cand(2, 10.20), '新合格快照（序號 2、來源 10.20）⇒ 刷新')
    feed(10.50, cand(2, 10.20), '重複通知（同序號）⇒ 不刷新')
    feed(11.00, cand(3, 9.00), '新序號但舊影格（來源 9.00 早於 10.20）⇒ 不刷新')
    feed(11.50, cand(4, 11.40, cons=0.0, dev=3.5), '與快照不一致（雙相機／同物件核對未過）⇒ 拒收、不刷新')
    feed(12.00, cand(5, 11.90, ready=0.0), '目前觀測未就緒 ⇒ 不刷新')
    feed(12.50, cand(6, 12.40, ik=1.0), '快照檢查未完成 ⇒ 不刷新')
    feed(12.80, cand(7, 12.70, age=0.9), '目前觀測過期（0.9 s > 0.5）⇒ 不刷新')
    o.t = 13.0
    o.guide_intake(13.0, dict(cam='v1', state='selected', xy=[-0.0227, 0.4083], t_last=12.95, reject=False))   # 底盤導引只有 xy
    steps.append(dict(t=13.0, what='底盤導引（只有 xy）⇒ 不經抓取候選、不刷新', refreshed=False, valid_t_src=o.seg_valid['t_src'], seq=o.seg_seq, stop=None))
    o2 = mk()                                                           # 鎖定後：另一個實例
    o = o2
    feed(7.18, cand(1, 7.05), '（鎖定案）首次合格快照')
    o.t = 7.40
    o._gp_apply(o.gp, dict(margin=0.46), np.array([-2.75, 0.0, 0.0]))
    o.w_locked = True
    feed(9.00, cand(2, 8.95), '鎖定後新合格快照 ⇒ 基線不受理、不刷新')
    lock_case = steps[-2:]
    steps = steps[:-2]
    # 停止觀測後依期限停止（主案例實例）
    o = o_main
    exp = []
    for t in (22.39, 22.41, 25.19, 25.21):                            # 舊規則 22.40（凍結 7.40＋15）；新規則 25.20（最後有效來源 10.20＋15）
        o.t = t
        MRO['bottle_w1_segtarget_ctrl']._advance_inner(o, None, False, np.array([-0.5, 0.0, 0.0]))
        exp.append(dict(t=t, stop=None if o.judge.stop is None else dict(o.judge.stop)))
        if o.judge.stop is not None:
            break
    # 第二條期限路徑（bottle_v2_ctrl.vision_fix_expired；A_1 第一阻點）：同一狀態，v2 層真碼、其下隔離
    import inspect
    import re
    import bottle_v2_ctrl as V2
    v2_below = nxt[nxt.index(MRO['bottle_v2_ctrl']) + 1]
    v2_below._advance_inner = lambda self, T, c, q: None
    exp_v2 = []
    o.judge.stop = None
    o.v2 = dict(V2.V2REG)
    for t in (22.39, 22.41, 25.19, 25.21):
        o.t = t
        MRO['bottle_v2_ctrl']._advance_inner(o, None, False, np.array([-0.5, 0.0, 0.0]))
        exp_v2.append(dict(t=t, stop=None if o.judge.stop is None else dict(o.judge.stop)))
        if o.judge.stop is not None:
            break
    # 全鏈掃描：控制器 MRO 內每個類別的原始碼，凡以 vfix 時刻判期限者列出（應只有 segtarget 與 v2，且都走 _seg_valid_t）
    scan = []
    for k in C.__mro__:
        try:
            src_ = inspect.getsource(k)
        except (TypeError, OSError):
            continue
        for ln in src_.split('\n'):
            if 'expire' in ln and re.search(r"self\.t - (self\.vfix\['t'\]|t_valid|self\._seg_valid_t\(\))", ln):
                scan.append(dict(cls=k.__module__, line=ln.strip()[:160], uses_hook=('t_valid' in ln or '_seg_valid_t' in ln)))
    # A_1 實錄時刻重放：凍結 7.9 s（快照來源 7.56）、刷新 10.68／14.20／17.31／20.43 ⇒ 兩條路徑在 22.9 s 都不停、35.43 s 後才逾期
    oa = _mk()
    oa.v2 = dict(V2.V2REG)
    oa.t = 7.67
    oa.candidate_intake(7.67, cand(1, 7.56))
    oa.t = 7.9
    oa._gp_apply(oa.gp, dict(margin=0.46), np.array([-2.72, 0.0, 0.0]))
    for k_, (ta, ts) in enumerate(((10.78, 10.68), (14.25, 14.20), (17.37, 17.31), (20.5, 20.43)), start=2):
        oa.t = ta
        oa.candidate_intake(ta, cand(k_, ts))
    a1 = []
    for t in (22.9, 35.43, 35.45):
        for lay in ('bottle_w1_segtarget_ctrl', 'bottle_v2_ctrl'):
            oa.judge.stop = None
            oa.t = t
            MRO[lay]._advance_inner(oa, None, False, np.array([-0.42, 0.0, 0.0]))
            a1.append(dict(t=t, layer=lay, stop=None if oa.judge.stop is None else oa.judge.stop['why']))
    res = dict(steps=steps, lock_case=lock_case, expiry=exp, expiry_v2=exp_v2, scan=scan, a1_replay=a1, valid_log=getattr(o, 'seg_valid_log', []),
               events=[e for e in o.events if e['what'].startswith('segtarget')], expire_old_rule=round(o.vfix['t'] + 15.0, 3), expire_new=round(o.seg_valid['t_src'] + 15.0, 3))
    for s in steps + lock_case:
        print(s, flush=True)
    print('期限 segtarget', exp, flush=True)
    print('期限 v2', exp_v2, flush=True)
    print('掃描', scan, flush=True)
    print('A_1 重放', a1, flush=True)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str)


if __name__ == '__main__':
    o_main = None
    _mk = mk

    def mk():                                                           # 記住第一個實例（主案例）
        global o_main
        x = _mk()
        if o_main is None:
            o_main = x
        return x
    main()
