#!/usr/bin/env python3
"""有效期政策「走實際繼承鏈」回歸（Codex 20261111_150220）。範圍＝dynsr43 實際使用的控制器類別 bottle_w1_obs_ctrl.W1PlaceController 的繼承鏈（不是整個 repository）。
做法：從最上層 o._advance_inner 依實際 MRO 順序往下呼叫；**含期限判定的層（以原始碼掃描找出：bottle_w1_segtarget_ctrl、bottle_v2_ctrl）跑真碼**，
  其餘定義 _advance_inner 的層改為透明轉呼叫（掃描證明其原始碼不含期限判定），最底層相位引擎（v2 之下）截斷。候選採用走真 candidate_intake 鏈；
  只把抓取計畫工作者與計畫接受的排程平移換成輕量替身（同 seg_validity_counterexample）。
核對：
  A_1 狀態（凍結 7.9 s、快照來源 7.56；刷新 10.68／14.20／17.31／20.43）：22.9 s 不被舊時鐘中止；停止新觀測後，第一個「t − 20.43 > 15」的控制拍才停止（推算邊界 35.43 s）。
  舊版起算（同一鏈去掉觀察層覆寫 ⇒ bottle_w1_tpa_cj_ctrl.W1PlaceController）：仍自首次凍結起算，22.9 s 後第一拍停止。
  缺有效來源戳：受控拒絕 segtarget_valid_src_missing，不退回凍結時刻。
  nom_3 紀錄（凍結 7.5、來源 7.22；刷新 10.35／13.47／16.56；閉爪 21.7）：閉爪前每個控制拍新舊判定一致（皆不停）。
    python3 evaluation/seg_validity_chain_regression.py OUT.json
"""
import inspect
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ros_chain as RC                                           # noqa: E402

OC, CJ = RC.import_with_real_ros('bottle_w1_obs_ctrl', 'bottle_w1_tpa_cj_ctrl')
import bottle_s0_logic as L                                             # noqa: E402
import bottle_v2_ctrl as V2                                             # noqa: E402
import bottle_w1_segtarget_ctrl as SG                                   # noqa: E402

NEW, OLD = OC.W1PlaceController, CJ.W1PlaceController
EXP_RE = re.compile(r'expire|_expired')


def scan(cls):
    out = []
    for k in cls.__mro__:
        if '_advance_inner' not in k.__dict__:
            continue
        src = inspect.getsource(k.__dict__['_advance_inner'])
        out.append(dict(cls=k.__module__, has_expiry=bool(EXP_RE.search(src))))
    return out


SC_NEW, SC_OLD = scan(NEW), scan(OLD)
EXPIRY_LAYERS = {r['cls'] for r in SC_NEW + SC_OLD if r['has_expiry']}
assert EXPIRY_LAYERS == {'bottle_w1_segtarget_ctrl', 'bottle_v2_ctrl'}, EXPIRY_LAYERS
ORIG = {}
for cls in (NEW, OLD):
    mro = [k for k in cls.__mro__ if '_advance_inner' in k.__dict__]
    i_v2 = [k.__module__ for k in mro].index('bottle_v2_ctrl')
    for j, k in enumerate(mro):
        if k in ORIG:
            continue
        ORIG[k] = k.__dict__['_advance_inner']
        if k.__module__ in EXPIRY_LAYERS:
            continue                                                    # 真碼
        if j > i_v2:
            k._advance_inner = lambda self, T, c, q: None               # v2 之下：相位引擎截斷（不含期限判定）
        else:
            k._advance_inner = (lambda kk: (lambda self, T, c, q: super(kk, self)._advance_inner(T, c, q)))(k)   # 透明轉呼叫（不含期限判定）
ON = [k for k in NEW.__mro__ if k.__module__ == 'bottle_w1_online_ctrl'][0]


def light_submit(self, xy, zs, seq):
    self.gp = dict(state='pending', id=1, xy=np.asarray(xy, float).copy(), zs=float(zs), seq=int(seq))


def light_apply(self, g, res, q):
    xy, zs, seq = g['xy'], g['zs'], g['seq']
    self.seg_z = zs
    self.vfix = dict(t=self.t, xy=np.array(xy, float), spread=0.0, n=1, margin=0.46, seq=seq, z_support=zs)
    self.w_goal = np.array(xy, float)
    self.seg_seq = seq
    self.seg_adopt.append((self.t, seq, list(map(float, xy)), zs))
    g.update(state='accepted', t_acc=float(self.t))
    self._ev('segtarget_freeze', seq=seq)


ON._gp_submit = light_submit
ON._gp_apply = light_apply
[k for k in NEW.__mro__ if k.__module__ == 'bottle_w1_tpa_ctrl'][0]._gp_apply = light_apply   # TPA 起動／R1a 分支設定隔離（與期限無關）


def mk(cls):
    o = cls.__new__(cls)
    o.t = 0.0
    o.judge = L.S0Judge.__new__(L.S0Judge)
    o.judge.stop, o.judge.log = None, []
    o.events = []
    o.seg, o.v2 = dict(SG.SEGREG), dict(V2.V2REG)
    o.seg_log, o.seg_rej, o.seg_seq, o.seg_z, o.seg_adopt, o.seg_obs = [], {}, 0, None, [], []
    o.vfix, o.w_goal, o.w_locked, o._cand, o.obs_last, o.obs_first = None, None, False, None, None, None
    o.blending, o.t_close_req, o.x_blend = True, None, -0.62
    o.gp, o.tlane, o._gp_defer, o._wrist_src = None, dict(y_lane=0.008), None, None
    o.vbuf, o.vrej = [], {}
    return o


def cand(seq, t_last, xy=(-0.0227, 0.4083), zs=0.58):
    return dict(ready=1.0, age_s=0.06, snap_seq=float(seq), snap_t_last=t_last, snap_cx=xy[0], snap_cy=xy[1], snap_z_support=zs, snap_ik=2.0, snap_local=2.0,
                snap_dev_mm=0.4, snap_consistent=1.0)


def build(cls, t_obs, t_src0, t_freeze, refresh):
    o = mk(cls)
    o.t = t_obs
    o.candidate_intake(t_obs, cand(1, t_src0))
    o.t = t_freeze
    o._gp_apply(o.gp, dict(margin=0.46), np.array([-2.7, 0.0, 0.0]))
    for k, (ta, ts) in enumerate(refresh, start=2):
        o.t = ta
        o.candidate_intake(ta, cand(k, ts))
    return o


def first_stop(o, t0, t1):
    """每個控制拍（0.05 s）從最上層 _advance_inner 呼叫；回傳第一個停止的時刻與原因。"""
    for t in np.round(np.arange(t0, t1 + 1e-9, 0.05), 2):
        o.t = float(t)
        o._advance_inner(None, False, np.array([-0.42, 0.0, 0.0]))
        if o.judge.stop is not None:
            return dict(t=float(t), why=o.judge.stop['why'], age=round(o.judge.stop.get('age', float('nan')), 3), policy=o.judge.stop.get('policy'))
    return None


def main():
    res = dict(scope='dynsr43 控制器 bottle_w1_obs_ctrl.W1PlaceController 的實際繼承鏈', scan_new=SC_NEW, scan_old=SC_OLD, expiry_layers=sorted(EXPIRY_LAYERS))
    A1 = dict(t_obs=7.67, t_src0=7.56, t_freeze=7.9, refresh=((10.78, 10.68), (14.25, 14.20), (17.37, 17.31), (20.5, 20.43)))
    o = build(NEW, **A1)
    res['a1_new'] = dict(stop_22_9=first_stop(o, 20.55, 22.95), first_stop_after=None)
    res['a1_new']['first_stop_after'] = first_stop(o, 23.0, 36.0)
    o = build(OLD, **A1)
    res['a1_old_chain'] = first_stop(o, 20.55, 36.0)
    om = mk(NEW)
    om.t = 7.67
    om.candidate_intake(7.67, dict(cand(1, 7.56), snap_t_last=float('nan')))
    om.t = 7.9
    om._gp_apply(om.gp, dict(margin=0.46), np.array([-2.7, 0.0, 0.0]))
    res['missing_src'] = dict(stop=om.judge.stop, seg_valid=getattr(om, 'seg_valid', None))
    N3 = dict(t_obs=7.29, t_src0=7.22, t_freeze=7.5, refresh=((10.42, 10.35), (13.54, 13.47), (16.63, 16.56)))
    on, oo = build(NEW, **N3), build(OLD, **N3)
    same, n = True, 0
    for t in np.round(np.arange(16.65, 21.7 + 1e-9, 0.05), 2):
        for x in (on, oo):
            x.t = float(t)
            x._advance_inner(None, False, np.array([-0.3, 0.0, 0.0]))
        n += 1
        same &= (on.judge.stop is None) == (oo.judge.stop is None)
    res['nom3_before_close'] = dict(ticks=n, identical=bool(same), new_stop=on.judge.stop, old_stop=oo.judge.stop)
    for k, v in res.items():
        if k not in ('scan_new', 'scan_old'):
            print(k, v, flush=True)
    print('scan_new', [(r['cls'], r['has_expiry']) for r in SC_NEW], flush=True)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str)


if __name__ == '__main__':
    main()
