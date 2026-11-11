#!/usr/bin/env python3
"""右側整合前置核對（Howard B.5；Codex 20261111_163931）。離線、不跑 Isaac。區域座標；右側＝整場景鏡射後的名目（瓶 (−0.0227, −0.4081)、第一桌在車道右側）。
一、控制端（真方法：bottle_w1_side_ctrl 與其繼承鏈；只以假工作者收請求）：
  a 區域先驗注入：觀看點、第一桌、桌腳、放置側。
  b 選側：右側導引 ⇒ right；不明（|d| < 0.15）⇒ 不選、不送站位；之後反側量測 ⇒ 受控停止 grasp_side_changed。
  c 站位請求：帶 side、鏡射桌腳、依側方位種子；未選側不送。
  d 抓取姿態 _grasp_T（工具 z 朝 −y）、收近點 _tcp_c（0, −0.2, 0.6）；未選側 ⇒ NaN／預設。
  e 候選同側閘：節點側別不同、快照序號 < 起算序號 ⇒ 拒收；同側 ⇒ 進基線採用鏈。
  f 預測操作構型核對：車道成立後物件機身 y 與側別同號。
二、幾何（工作者同一函式）：
  g 站位 bottle_standoff.evaluate（右／左對照）；h 抓取計畫 plan_grasp（沿所選車道）；
  i 收近與放置過渡：以右側抓持構型（抓取分支 q_mid）抬升後的側持姿態 T_side、TCP→瓶 T_rel，plan_return（tcp_c 依側）核自碰、持瓶↔連桿、車體頂、關節餘裕；
  j 觀察可見性：右側區域觀看點、預測底盤位姿，view_q1＋visible。
    python3 evaluation/right_side_integration_check.py OUT.json
"""
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ros_chain as RC                                           # noqa: E402

SC, = RC.import_with_real_ros('bottle_w1_side_ctrl')
import bottle_plan_worker as PW                                         # noqa: E402
import bottle_r0_approach as R0                                         # noqa: E402
import bottle_s0b_ctrl as B                                             # noqa: E402
import bottle_s1_geom as G                                              # noqa: E402
import bottle_side as BSD                                               # noqa: E402
import bottle_standoff as SO                                            # noqa: E402
import bottle_tlane as TLN                                              # noqa: E402
import bottle_view_gen as VG                                            # noqa: E402
import bottle_w1_online_ctrl as ON                                      # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2m                              # noqa: E402
import bottle_s0_logic as L                                             # noqa: E402

C = SC.W1PlaceController
OBJ_R = np.array([-0.0227, -0.4081])
MIRROR = np.array([1.0, -1.0, 1.0])


def region_prior(side):
    my = -1.0 if side == 'right' else 1.0
    m = np.array([1.0, my, 1.0])
    top = [(np.asarray(c, float) * m)[:2].tolist() for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_ == 'table_top'][0]
    legs = [((np.asarray(c, float) * m).tolist(), np.asarray(sz, float).tolist()) for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]
    return dict(side_prior=side, mirror_y=side == 'right', table1_top=top, table1_size=[G.S1_TABLE['size_x'], G.S1_TABLE['size_y']], table1_legs=legs,
                look=BSD.look_point(side).tolist())


def mk(side='right'):
    o = C.__new__(C)
    o.t = 2.0
    o.judge = L.S0Judge.__new__(L.S0Judge)
    o.judge.stop, o.judge.log = None, []
    o.events = []
    o.grasp_side, o.side_info, o.side_log, o.side_chk = None, None, [], None
    o.place_side, o.side_prior, o.region_prior = None, None, None
    o.node_side, o.node_side_seq0, o._legs_prior, o.side_rej = None, None, None, {}
    o.guide, o.guide_n, o.guide_block, o.obs_trig, o.so, o.tlane, o._wrist_src, o._gp_defer = None, 0, None, None, None, None, None, None
    o.seg_rej, o.seg_log = {}, []
    my = -1.0 if side == 'right' else 1.0
    o.place = dict(c=(1.8, my * -1.19), target=(1.8, my * -1.38), lane_y=my * -1.8, top=0.60, size=(0.8, 0.5), theta=0.0, entry_back=0.6)   # 區域放置先驗（PLACE 世界減區域偏移）
    sent = []
    o.pc = types.SimpleNamespace(submit=lambda kind, inp: (sent.append((kind, inp)), len(sent))[1], poll=lambda rid: None, alive=lambda: True)
    C.__init__.__globals__  # noqa
    sub = o.pc.submit                                                   # 與建構子相同的請求包裝（建構子其餘狀態不需要）

    def submit(kind, inp):
        if kind in ('grasp', 'standoff'):
            if o.grasp_side is None:
                o._ev('submit_refused_side_unknown', kind=kind)
                return None
            inp = dict(inp, side=o.grasp_side)
            if kind == 'standoff':
                if o._legs_prior is not None:
                    inp['legs'] = [list(map(list, lg)) for lg in o._legs_prior]
                seeds = [list(s_) for s_ in inp.get('q_seeds', [])]
                if len(seeds) >= 2:
                    seeds[1][0] = math.atan2(BSD.sgn(o.grasp_side) * 0.4275, 0.35)
                    inp['q_seeds'] = seeds
        return sub(kind, inp)
    o.pc.submit = submit
    o._sent = sent
    o.region_prior = region_prior(side)
    return o


def part_control():
    out = {}
    o = mk('right')
    o.apply_region_prior()
    out['a_region_prior'] = dict(look=o.look.tolist(), table1_xy=o.table1_xy, n_legs=len(o._legs_prior), legs_y_sign=sorted({int(np.sign(lg[0][1])) for lg in o._legs_prior}),
                                 place_side=o.place_side, side_prior=o.side_prior)
    # b 選側
    o.t = 2.3
    s_amb = o._update_side(np.array([-0.02, 0.10]), 'test_ambiguous')
    s_r = o._update_side(OBJ_R, 'guide:w1')
    out['b_select'] = dict(ambiguous=s_amb, right=s_r, grasp_side=o.grasp_side, info=o.side_info)
    # c 站位請求（真 _standoff 的 submit 包裝；直接呼叫包裝）
    q_seeds = [R0.Q_TUCK.tolist(), np.r_[math.atan2(0.4275, 0.35), R0.Q_TUCK[1:]].tolist()] + [s_.tolist() for s_ in R0.RNG_SEEDS]
    legs_left = [(c.tolist(), sz.tolist()) for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]
    o.pc.submit('standoff', dict(xy=OBJ_R.tolist(), q_seeds=q_seeds, legs=legs_left, y_tube=list(B.Y_TUBE)))
    k_, inp_ = o._sent[-1]
    out['c_standoff_request'] = dict(kind=k_, side=inp_.get('side'), legs_y_sign=sorted({int(np.sign(lg[0][1])) for lg in inp_['legs']}), seed1_q1=round(inp_['q_seeds'][1][0], 4))
    o2 = mk('right')
    o2.apply_region_prior()
    r2 = o2.pc.submit('standoff', dict(xy=OBJ_R.tolist(), q_seeds=q_seeds, legs=legs_left))
    out['c_unselected_refused'] = dict(returned=r2, events=[e['what'] for e in o2.events])
    # d 抓取姿態、收近點
    Tg = o._grasp_T(-0.0227, -0.4081, 0.68)
    out['d_grasp_T'] = dict(tool_z=np.round(Tg[:3, 2], 4).tolist(), tool_x=np.round(Tg[:3, 0], 4).tolist(), tcp=np.round(Tg[:3, 3], 4).tolist())
    out['d_tcp_c'] = [float(v) for v in o._tcp_c()]
    out['d_unselected'] = dict(grasp_T_nan=bool(np.isnan(o2._grasp_T(0.0, -0.4, 0.68)).all()), tcp_c=[float(v) for v in o2._tcp_c()])
    # e 候選同側閘（基線採用鏈以記錄器替代，只看是否被放行）
    passed = []
    base_ci = SC.OC.W1PlaceController.candidate_intake
    SC.OC.W1PlaceController.candidate_intake = lambda self, t, c: passed.append(int(c['snap_seq']))
    c_ = dict(ready=1.0, age_s=0.05, snap_seq=3.0, snap_t_last=7.2, snap_cx=-0.0227, snap_cy=-0.4081, snap_z_support=0.58, snap_ik=2.0, snap_local=2.0, snap_dev_mm=0.3, snap_consistent=1.0)
    o.node_side, o.node_side_seq0 = 1, 1
    o.candidate_intake(7.3, c_)
    o.node_side, o.node_side_seq0 = 2, 4
    o.candidate_intake(7.35, c_)
    o.node_side, o.node_side_seq0 = 2, 2
    o.candidate_intake(7.4, c_)
    SC.OC.W1PlaceController.candidate_intake = base_ci
    out['e_candidate_gate'] = dict(passed_seqs=passed, rejected=dict(o.side_rej))
    # f 預測操作構型
    r_ = TLN.lane_from_target(OBJ_R, -0.35, 0.40, 'right')
    out['f_lane'] = dict(x_entry=round(r_[0], 4), y_lane=round(r_[1], 4), y_body=round(float(OBJ_R[1] - r_[1]), 4))
    # b' 反側量測 ⇒ 受控停止
    o.t = 8.0
    o._update_side(np.array([-0.0227, 0.40]), 'snapshot:9')
    out['b_side_changed'] = o.judge.stop
    return out


def part_geometry():
    out = {}
    q_seeds = [R0.Q_TUCK.tolist()] + [s_.tolist() for s_ in R0.RNG_SEEDS]
    for side in ('right', 'left'):
        my = -1.0 if side == 'right' else 1.0
        xy = np.array([OBJ_R[0], my * abs(OBJ_R[1])])
        legs = [((np.asarray(c, float) * np.array([1.0, my, 1.0])).tolist(), np.asarray(sz, float).tolist()) for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]
        seeds = [q_seeds[0], np.r_[math.atan2(my * 0.4275, 0.35), R0.Q_TUCK[1:]].tolist()] + q_seeds[1:]
        ok, why, res = SO.evaluate(dict(xy=xy.tolist(), band=0.10, pre_m=0.06, lift_m=0.05, q_seeds=seeds, legs=legs, y_tube=list(B.Y_TUBE), side=side))
        rows = [dict(y_reach=r['y_reach'], y_lane=r['y_lane'], ok=r['ok'], why=r['why'], margin=r['margin_min'], leg_clear=r['base_leg_clear']) for r in res['rows']]
        g = dict(standoff=dict(ok=ok, why=why, chosen=None if not ok else {k: res['chosen'][k] for k in ('y_reach', 'y_lane', 'margin_min', 'base_leg_clear')}, rows=rows))
        yr = res['chosen']['y_reach'] if ok else 0.4275
        yl = float(TLN.lane_from_target(xy, -0.35, yr, side)[1])
        okg, whyg, rg = PW.plan_grasp(dict(xy=xy.tolist(), z_support=0.58, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl,
                                           q_seeds=q_seeds, q_now=R0.Q_TUCK.tolist(), q_seed=None, side=side))
        g['grasp_plan'] = dict(ok=okg, why=whyg, y_lane=round(yl, 4), margin=None if not okg else round(float(rg.get('margin', float('nan'))), 4),
                               x_mid=None if not okg else rg.get('x_mid'), q_mid=None if not okg else np.round(rg['q_mid'], 4).tolist())
        if okg:                                                          # i 收近與放置過渡（從右側抓持構型）
            q_mid = np.asarray(rg['q_mid'], float)
            x_mid = float(rg['x_mid'])
            T_tcp_b = R0.K.fk(np.r_[0.0, 0.0, 0.0, q_mid], 'link_tcp')
            T_side = T_tcp_b.copy()
            T_side[2, 3] += 0.05                                         # 抬升 lift_m（底盤座標）
            T_gw = BSD.grasp_target(xy[0], xy[1], 0.58 + 0.10, side)
            T_bot = np.eye(4)
            T_bot[:3, 3] = [xy[0], xy[1], 0.58 + G.BOTTLE['h'] / 2]
            T_rel = np.linalg.inv(T_gw) @ T_bot
            tcp_c = (np.array([0.0, 0.2, 0.6]) * np.array([1.0, my, 1.0])).tolist()
            okr, whyr, rr = PW.plan_return(dict(T_side=T_side.tolist(), T_rel=T_rel.tolist(), q_arm=q_mid.tolist(), top2=0.60, z_pad=C2m.Z_PAD, h_clear=ON.H_CLEAR,
                                                tcp_c=tcp_c, ns=ON.NS_RET, scale=1.05, step=0.05, floor=dict(ON.DUR_FLOOR), limits=ON.RET_LIM))
            g['return_plan'] = dict(ok=okr, why=whyr, tcp_c=tcp_c, T_side_xyz=np.round(T_side[:3, 3], 4).tolist(), base_x_grasp=round(x_mid, 4),
                                    seg={k: dict(T=round(v['T_need'], 3), worst={kk: round(float(vv), 4) for kk, vv in v['worst'].items()}) for k, v in (rr.get('seg') or {}).items()},
                                    dur=rr.get('dur'), z_back=None if not rr else round(float(rr.get('z_back', float('nan'))), 4))
        vis = []                                                         # j 觀察可見性
        look = BSD.look_point(side)
        for bp in ([-3.86, -my * 0.60, my * 0.60], [-2.9, 0.0, 0.0], [-1.5, 0.0, 0.0]):   # 起點在車道另一側（左：區域 y −0.8；右：+0.8），斜向車道
            q_obs, info = VG.view_q1(R0.K, np.asarray(bp, float), R0.Q_VHIGH, look, -1.5, 1.5)
            v = None if q_obs is None else VG.visible(R0.K, np.asarray(bp, float), np.asarray(q_obs, float), VG.bottle_pts(float(look[0]), float(look[1]), 0.60), None)
            vis.append(dict(base=bp, q1=None if q_obs is None else round(float(q_obs[0]), 4), ok=None if v is None else bool(v['ok']),
                            elev_deg=None if v is None else round(float(v.get('elev_deg', float('nan'))), 2)))
        g['view'] = vis
        out[side] = g
    return out


def main():
    res = dict(control=part_control(), geometry=part_geometry())
    print(json.dumps(res['control'], ensure_ascii=False, default=str)[:3000], flush=True)
    for side in ('right', 'left'):
        g = res['geometry'][side]
        print(side, 'standoff', g['standoff']['ok'], g['standoff']['chosen'], 'grasp', g['grasp_plan']['ok'], g['grasp_plan']['why'], 'margin', g['grasp_plan']['margin'],
              'return', g.get('return_plan', {}).get('ok'), g.get('return_plan', {}).get('why'), flush=True)
        print('   return seg', json.dumps(g.get('return_plan', {}).get('seg'), ensure_ascii=False)[:600], flush=True)
        print('   view', g['view'], flush=True)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str)


if __name__ == '__main__':
    main()
