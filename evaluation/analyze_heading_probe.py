#!/usr/bin/env python3
"""Metrics for one bigarena static heading trial.

Reads the bag written by run_bigarena_isaac.sh and reports what the heading
objective actually did. Everything is timed on the message header where one
exists, so the numbers are in simulation time, not bag receipt time.

Definitions used here, stated because they have been confused before:
  * 淨距 in the site-selection record is CENTRE to obstacle surface. Nothing
    is subtracted for the robot. Divide-out is the reader's job.
  * 前進 / 倒退 / 側移主導 are NOT a partition. A robot moving forward while
    strafing is in both 前進 and 側移主導. Each fraction is the share of
    moving time satisfying its own predicate; they do not sum to 1.
"""
import argparse, json, math, os, sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry, Path

MOVE_EPS = 0.05      # m/s, below this the sample is "not moving"

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def hstamp(m):
    s = m.header.stamp
    return s.sec + s.nanosec * 1e-9

def read(bag):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    truth, cmd, head, plans, goal = [], [], [], [], []
    movers = {}
    while rd.has_next():
        topic, data, t = rd.read_next()
        tr = t * 1e-9
        if topic == '/model/omni_bot/pose':
            m = deserialize_message(data, PoseStamped)
            q = m.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y ** 2 + q.z ** 2))
            truth.append((hstamp(m), m.pose.position.x, m.pose.position.y, yaw))
        elif topic == '/cmd_vel':
            m = deserialize_message(data, Twist)
            # Twist has no header; bag receipt time is the only clock available
            cmd.append((tr, m.linear.x, m.linear.y, m.angular.z))
        elif topic == '/gmpc/heading':
            m = deserialize_message(data, Vector3Stamped)
            head.append((hstamp(m), m.vector.x, m.vector.y, m.vector.z))
        elif topic == '/plan':
            m = deserialize_message(data, Path)
            plans.append((hstamp(m), len(m.poses)))
        elif topic == '/goal_pose':
            m = deserialize_message(data, PoseStamped)
            goal.append((hstamp(m), m.pose.position.x, m.pose.position.y))
        elif topic.startswith('/model/dyn_obs_'):
            m = deserialize_message(data, PoseStamped)
            movers.setdefault(topic, []).append(
                (hstamp(m), m.pose.position.x, m.pose.position.y))
    return (np.array(truth), np.array(cmd), np.array(head),
            np.array(plans), np.array(goal), movers)

def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    a = ap.parse_args()
    bag = os.path.join(a.run_dir, 'bag')
    truth, cmd, head, plans, goal, movers = read(bag)
    rec = {}
    js = os.path.join(a.run_dir, 'isaac_run.json')
    sim = json.load(open(js)) if os.path.exists(js) else {}

    print(f'== {os.path.basename(a.run_dir)} ==')
    _rd = sim.get('run', {}).get('rendering_dt')
    _pd = sim.get('run', {}).get('physics_dt')
    _sk = sim.get('run', {}).get('time_skew_max_s')
    if _rd is None or _pd is None:
        print('  !! 時間基準待核正：本趟未記錄 physics_dt / rendering_dt，'
              '所有「秒」與「每秒」數值不可引用')
    elif abs(float(_rd) - float(_pd)) > 1e-12:
        print(f'  !! 時間基準失真：rendering_dt={_rd} != physics_dt={_pd}')
    else:
        print(f'  時間基準：physics_dt={_pd}, rendering_dt={_rd}, '
              f'迴圈/物理時鐘最大偏差 {(_sk or 0)*1e6:.2f} µs')
    print(f'  訊息數  真值位姿 {len(truth)}  /cmd_vel {len(cmd)}  '
          f'/gmpc/heading {len(head)}  /plan {len(plans)}  /goal_pose {len(goal)}')
    if len(truth) < 10:
        print('  !! 真值位姿樣本不足，無法分析'); return 1

    # `ros2 topic pub` leaves header.stamp at zero, so /goal_pose's own stamp
    # is not a usable task start. The simulator records the sim time at which
    # it received the goal; that is the authority. Fall back only if absent.
    _run = sim.get('run', {})
    _gt = _run.get('goal_sim_t')
    # Motion start is taken from the command stream, and goal_sim_t is reported
    # alongside it rather than instead of it.
    #
    # NOTE: earlier runs showed goal_sim_t sitting 0.84-1.96 s away from the
    # first command, and that was attributed here to the simulator servicing
    # its ROS callbacks inside the physics loop. That attribution is NOT
    # supported: those runs were published by `ros2 topic pub`, whose
    # header.stamp is zero, so the baseline being compared against was itself
    # invalid. With publish_goal.py stamping in simulation time, the gap
    # between publish and the simulator's callback measured 0.030 s on seed 1.
    # That shows the old comparison was meaningless -- it does not establish
    # what the old gap actually was. Settling that would need a per-message
    # re-check of those runs, which has not been done.
    _first = None
    for _x in (_run.get('log') or []):
        if max(abs(v) for v in _x['cmd']) > 1e-6:
            _first = float(_x['t']); break
    if _first is not None:
        t0, src = _first, '首次非零 /cmd_vel（真值 log）'
    elif _gt is not None:
        t0, src = float(_gt), 'isaac_run.json 的 goal_sim_t'
    else:
        t0, src = float(truth[0, 0]), 'bag 起點（無可用時戳）'
    print(f'  任務起點模擬時刻 {t0:.2f} s（來源：{src}）')
    if _gt is not None and _first is not None:
        print(f'    模擬器記錄的 goal_sim_t = {_gt:.2f} s，'
              f'與首次命令相差 {_gt - _first:+.2f} s'
              '（成因未定；舊趟的零時間戳使當時的比較基準無效）')
        rec['goal_cb_lag_s'] = _gt - _first

    # ---- 結束狀態（來自模擬器，不是我重算的）----
    run = sim.get('run', {})
    for k in ('stop_reason', 'sim_time', 'goal_sim_t', 'goal_msgs_received',
              'rtf_measured', 'wall_time', 'task_limit'):
        if k in run: print(f'  模擬器回報 {k} = {run[k]}')
    for tag in ('at_trigger', 'after_stop'):
        d = run.get(tag)
        if isinstance(d, dict):
            print(f'    {tag}: t={d.get("sim_t")} ({d.get("x"):.3f},{d.get("y"):.3f}) '
                  f'yaw={math.degrees(d.get("yaw", 0.0)):.2f}° '
                  f'距目標 {d.get("dist_goal"):.3f} m')
    if run.get('at_trigger'):
        st = run['at_trigger'].get('sim_t')
        if st is not None:
            print(f'  任務時間（首次命令 -> 停止觸發）= {st - t0:.2f} s 模擬時間')
            rec['task_time_s'] = st - t0
    rec['stop_reason'] = run.get('stop_reason')
    L = run.get('log') or []
    if L:
        import math as _m
        print(f'  模型穩定性（{len(L)} 筆真值取樣）：手臂保持誤差 max '
              f'{max(r["arm_err"] for r in L)*1000:.3f} mrad，'
              f'|roll| max {max(abs(r["roll"]) for r in L)*57.2958:.4f}°，'
              f'|pitch| max {max(abs(r["pitch"]) for r in L)*57.2958:.4f}°')
        # Final-approach sensitivity: once the robot is inside a short remaining
        # distance, the robot->goal chord swings hard for a small lateral
        # offset. That is where a heading objective can start chasing its tail,
        # so it is reported rather than left to be inferred from the endpoint.
        near = [r for r in L if r['dist_goal'] <= 0.6]
        if len(near) > 2:
            yy = [r['yaw'] for r in near]
            cum = sum(abs((b - a_ + _m.pi) % (2*_m.pi) - _m.pi)
                      for a_, b in zip(yy[:-1], yy[1:]))
            print(f'    最後 0.6 m 內（{len(near)} 筆）累計絕對轉角 '
                  f'{_m.degrees(cum):.2f}°，yaw {_m.degrees(yy[0]):.2f}° -> '
                  f'{_m.degrees(yy[-1]):.2f}°')
            rec['near_goal_cum_turn_deg'] = _m.degrees(cum)

    # ---- 移動體是否真的靜止 ----
    _traf = run.get('traj')
    print('  移動體位移（未下命令不等於不動；traffic 開啟時本欄預期為非零）：')
    worst = 0.0
    for t, h in sorted(movers.items()):
        h = np.array(h)
        d = float(np.max(np.hypot(h[:, 1] - h[0, 1], h[:, 2] - h[0, 2])))
        worst = max(worst, d)
        if d > 0.005:
            print(f'    {t} 最大位移 {d*1000:.1f} mm  << 非靜止')
    print(f'    全部 {len(movers)} 個移動體的最大位移 = {worst*1000:.1f} mm')
    rec['mover_max_disp_mm'] = worst * 1000

    # ---- 只看目標發布之後 ----
    tr = truth[truth[:, 0] >= t0]
    if len(tr) < 10:
        print('  !! 目標後真值樣本不足'); return 1
    dt = np.diff(tr[:, 0])
    ok = dt > 1e-6

    # 真值機體速度：世界速度旋回車體座標
    vx_w = np.diff(tr[:, 1]) / np.where(ok, dt, np.nan)
    vy_w = np.diff(tr[:, 2]) / np.where(ok, dt, np.nan)
    yaw_m = tr[:-1, 3]
    vx_b = vx_w * np.cos(yaw_m) + vy_w * np.sin(yaw_m)
    vy_b = -vx_w * np.sin(yaw_m) + vy_w * np.cos(yaw_m)
    spd = np.hypot(vx_b, vy_b)
    dyaw = np.array([wrap(b - a_) for a_, b in zip(tr[:-1, 3], tr[1:, 3])])
    wz_t = dyaw / np.where(ok, dt, np.nan)

    m = ok & np.isfinite(spd) & (spd > MOVE_EPS)
    tm = float(np.sum(dt[m]))
    print(f'\n  ---- 運動組成（真值機體座標，移動樣本 {int(m.sum())}，'
          f'合計 {tm:.2f} s；門檻 {MOVE_EPS} m/s）----')
    if tm > 0:
        f_fwd = float(np.sum(dt[m & (vx_b > MOVE_EPS)])) / tm
        f_bwd = float(np.sum(dt[m & (vx_b < -MOVE_EPS)])) / tm
        f_lat = float(np.sum(dt[m & (np.abs(vy_b) > np.abs(vx_b))])) / tm
        print(f'    前進   (vx_b > +{MOVE_EPS})        {f_fwd*100:5.1f} %')
        print(f'    倒退   (vx_b < -{MOVE_EPS})        {f_bwd*100:5.1f} %')
        print(f'    側移主導 (|vy_b| > |vx_b|)      {f_lat*100:5.1f} %')
        both = float(np.sum(dt[m & (vx_b > MOVE_EPS) &
                               (np.abs(vy_b) > np.abs(vx_b))])) / tm
        print(f'    前進且側移主導（三者重疊，非互斥）{both*100:5.1f} %')
        rec.update(frac_forward=f_fwd, frac_backward=f_bwd,
                   frac_lateral_dom=f_lat, frac_fwd_and_lat=both)

    # ---- 車頭與行進方向的夾角 -------------------------------------------
    # 定義在看結果之前固定：
    #  * 主指標**不分穩態／過渡段**。用朝向誤差本身去切出「穩態」再報朝向誤差
    #    會構成循環；而且長時間穩定側移會被整段叫成「過渡」，不符實際行為。
    #  * 有效移動樣本 = 真值平移速度 >= MOVE_EPS。低速樣本不計夾角。
    #  * 「持續對齊區間」= 夾角連續至少 SUSTAIN_S 秒低於 SUSTAIN_DEG。
    #    報它占有效移動時間的比例，**不稱為穩態**。
    SUSTAIN_S, SUSTAIN_DEG = 1.0, 15.0
    ang = np.full(len(dt), np.nan)
    ok_ang = ok & np.isfinite(spd) & (spd > MOVE_EPS)
    for i in np.nonzero(ok_ang)[0]:
        ang[i] = abs(wrap(math.atan2(vy_w[i], vx_w[i]) - tr[i, 3]))
    av = ang[np.isfinite(ang)]
    print(f'\n  ---- 車頭與行進方向夾角（有效移動樣本 {len(av)}，'
          f'門檻 {MOVE_EPS} m/s；不分穩態／過渡段）----')
    if len(av):
        w_ = dt[np.isfinite(ang)]
        tot = float(np.sum(w_))
        print(f'    中位 {math.degrees(np.percentile(av,50)):6.2f}°  '
              f'p90 {math.degrees(np.percentile(av,90)):6.2f}°  '
              f'max {math.degrees(np.max(av)):6.2f}°')
        f10 = float(np.sum(w_[av < math.radians(10)])) / tot
        f15 = float(np.sum(w_[av < math.radians(15)])) / tot
        print(f'    < 10° 佔 {f10*100:5.1f} %   < 15° 佔 {f15*100:5.1f} %'
              '（依時間加權）')
        # 持續對齊區間
        run_s, best, segs = 0.0, 0.0, []
        for i in np.nonzero(np.isfinite(ang))[0]:
            if ang[i] < math.radians(SUSTAIN_DEG):
                run_s += dt[i]
            else:
                if run_s >= SUSTAIN_S: segs.append(run_s)
                run_s = 0.0
        if run_s >= SUSTAIN_S: segs.append(run_s)
        sus = float(sum(segs))
        print(f'    持續對齊區間（連續 >= {SUSTAIN_S} s 低於 {SUSTAIN_DEG}°）：'
              f'{len(segs)} 段，合計 {sus:.2f} s = {100*sus/tot:.1f} % 有效移動時間')
        rec.update(align_med_deg=math.degrees(np.percentile(av, 50)),
                   align_p90_deg=math.degrees(np.percentile(av, 90)),
                   align_frac_lt10=f10, align_frac_lt15=f15,
                   sustained_align_s=sus, sustained_align_frac=sus / tot,
                   sustained_align_n=len(segs))
    # 低速樣本另外報，不丟掉
    lo = ok & np.isfinite(spd) & (spd <= MOVE_EPS)
    lo_t = float(np.sum(dt[lo]))
    lo_turn = float(np.sum(np.abs(dyaw[lo]))) if lo.any() else 0.0
    print(f'    低速（<= {MOVE_EPS} m/s）時間 {lo_t:.2f} s，'
          f'其間累計絕對轉角 {math.degrees(lo_turn):.2f}°'
          f'（原地轉向）')
    rec.update(lowspeed_s=lo_t, lowspeed_turn_deg=math.degrees(lo_turn))

    # ---- 轉動 ----
    cum = float(np.sum(np.abs(dyaw[ok])))
    net = wrap(float(tr[-1, 3] - tr[0, 3]))
    print(f'\n  ---- 轉動（真值）----')
    print(f'    起始 yaw {math.degrees(tr[0,3]):7.2f}°  結束 yaw {math.degrees(tr[-1,3]):7.2f}°')
    print(f'    淨轉角 {math.degrees(net):7.2f}°   累計絕對轉角 {math.degrees(cum):8.2f}°')
    print(f'    比值 累計/|淨| = {cum/max(abs(net),1e-6):6.2f}  （1.0 = 全程單向轉）')
    w = wz_t[ok & np.isfinite(wz_t)]
    print(f'    角速度 |wz| p50 {pct(np.abs(w),50):.4f}  p95 {pct(np.abs(w),95):.4f}  '
          f'max {float(np.max(np.abs(w))) if len(w) else float("nan"):.4f} rad/s')
    tw = tr[:-1, 0][ok & np.isfinite(wz_t)]
    if len(w) > 2:
        aw = np.diff(w) / np.maximum(np.diff(tw), 1e-6)
        print(f'    角加速度 |dwz/dt| p95 {pct(np.abs(aw),95):.3f}  '
              f'max {float(np.max(np.abs(aw))):.3f} rad/s²')
        rec['ang_acc_p95'] = pct(np.abs(aw), 95)
        # 擺動：角速度變號次數（先過門檻，避免雜訊計數）
        sg = np.sign(np.where(np.abs(w) > 0.02, w, 0.0))
        sg = sg[sg != 0]
        flips = int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0
        print(f'    角速度變號次數（|wz|>0.02 才計）= {flips}')
        rec['wz_sign_flips'] = flips
    rec.update(cum_abs_turn_deg=math.degrees(cum), net_turn_deg=math.degrees(net),
               wz_p95=pct(np.abs(w), 95),
               wz_max=float(np.max(np.abs(w))) if len(w) else None)

    # ---- 朝向參考：原始 vs 限速後 ----
    if len(head):
        hd = head[head[:, 0] >= t0]
        if len(hd) > 2:
            raw, ref, act = hd[:, 1], hd[:, 2], hd[:, 3]
            fin = np.isfinite(raw) & np.isfinite(ref)
            print(f'\n  ---- 朝向參考（{len(hd)} 筆，有效 {int(fin.sum())}）----')
            e = np.array([abs(wrap(r - a_)) for r, a_ in zip(ref[fin], act[fin])])
            print(f'    |參考 - 實際 yaw|  p50 {math.degrees(pct(e,50)):6.2f}°  '
                  f'p95 {math.degrees(pct(e,95)):6.2f}°  max {math.degrees(float(np.max(e))):6.2f}°')
            jr = np.abs([wrap(b - a_) for a_, b in zip(raw[fin][:-1], raw[fin][1:])])
            jf = np.abs([wrap(b - a_) for a_, b in zip(ref[fin][:-1], ref[fin][1:])])
            print(f'    每步跳動 原始 p95 {math.degrees(pct(jr,95)):6.2f}° max {math.degrees(np.max(jr)):6.2f}°')
            print(f'    每步跳動 限速後 p95 {math.degrees(pct(jf,95)):6.2f}° max {math.degrees(np.max(jf)):6.2f}°')
            print(f'    限速削掉的最大單步跳動 = {math.degrees(np.max(jr)-np.max(jf)):6.2f}°')
            rec.update(ref_err_p95_deg=math.degrees(pct(e, 95)),
                       raw_jump_max_deg=math.degrees(float(np.max(jr))),
                       ref_jump_max_deg=math.degrees(float(np.max(jf))))
            # 重規劃當下原始朝向是否跳動
            if len(plans):
                pl = plans[plans[:, 0] >= t0]
                js_ = []
                for tp in pl[:, 0]:
                    i = int(np.searchsorted(hd[:, 0], tp))
                    if 1 <= i < len(hd) and np.isfinite(hd[i,1]) and np.isfinite(hd[i-1,1]):
                        js_.append(abs(wrap(hd[i, 1] - hd[i - 1, 1])))
                if js_:
                    print(f'    重規劃 {len(pl)} 次；重規劃當下原始朝向跳動 '
                          f'p95 {math.degrees(pct(js_,95)):6.2f}° max {math.degrees(max(js_)):6.2f}°')
                    rec['replan_n'] = len(pl)
                    rec['replan_raw_jump_max_deg'] = math.degrees(max(js_))
    else:
        print('\n  !! /gmpc/heading 無訊息 —— heading 未啟用或未錄到')

    # ---- 命令端角速度限值檢查（限速在參考上，不代表 wz 受限）----
    if len(cmd):
        cw = np.abs(cmd[:, 3])
        print(f'\n  ---- 命令 /cmd_vel ----')
        print(f'    |wz_cmd| p95 {pct(cw,95):.4f}  max {float(np.max(cw)):.4f} rad/s')
        print(f'    |vx_cmd| max {float(np.max(np.abs(cmd[:,1]))):.4f}  '
              f'|vy_cmd| max {float(np.max(np.abs(cmd[:,2]))):.4f} m/s')
        # 全向底盤不可只看縱向：側向加速度同樣是平滑度的一部分
        ct = cmd[:, 0]; cdt = np.diff(ct)
        good = cdt > 1e-6
        for lbl, col in (('dvx/dt', 1), ('dvy/dt', 2)):
            dv = np.abs(np.diff(cmd[:, col])[good] / cdt[good])
            print(f'    |{lbl}| p95 {np.percentile(dv,95):7.3f}  '
                  f'max {np.max(dv):7.3f} m/s²  '
                  f'（限值 a{"x" if col==1 else "y"}_max 6.25）')
            rec[f'{"ax" if col==1 else "ay"}_p95'] = float(np.percentile(dv, 95))
            rec[f'{"ax" if col==1 else "ay"}_max'] = float(np.max(dv))
        rec['wz_cmd_max'] = float(np.max(cw))
        # gmpc_node runs with wheel_coupling=True (its declare_parameter
        # default, not overridden in gmpc_params.yaml), so the QP already
        # carries the four-wheel constraint |v_perp + L*wz| <= r*w_max on top
        # of the per-axis boxes. This check therefore CONFIRMS that constraint
        # rather than testing for its absence: the demand should saturate at
        # w_max and never meaningfully exceed it. A slew limit on the heading
        # REFERENCE would say nothing about this either way, which is why it is
        # measured on the command stream.
        R, LL, WMAX = 0.05, 0.245, 5.55        # m, m, rad/s (gmpc_params.yaml)
        need = (np.maximum(np.abs(cmd[:, 1]), np.abs(cmd[:, 2]))
                + LL * np.abs(cmd[:, 3])) / R
        # Tolerance, not a strict >: a command that saturates wz at exactly
        # wz_max with zero translation needs exactly WMAX, and reporting that
        # as a violation would be a floating-point artefact, not a finding.
        TOL = 1e-6 * WMAX
        over = need > WMAX + TOL
        print(f'    輪速需求 max {float(np.max(need)):.4f} rad/s '
              f'（硬體上限 {WMAX}）；超出樣本 {int(over.sum())}/{len(need)}'
              f' = {100.0*over.mean():.1f} %')
        i = int(np.argmax(need))
        print(f'      需求最高一筆 vx={cmd[i,1]:+.4f} vy={cmd[i,2]:+.4f} '
              f'wz={cmd[i,3]:+.4f} -> {need[i]:.4f} rad/s '
              f'（餘裕 {WMAX - need[i]:+.4f}）')
        # The dangerous combination is high wz WHILE translating: wz_max is only
        # valid at zero translation, so this is where a decoupled box constraint
        # could authorise something the wheels cannot do.
        moving = np.maximum(np.abs(cmd[:, 1]), np.abs(cmd[:, 2])) > 0.05
        if moving.any():
            j = int(np.argmax(need * moving))
            print(f'      平移中(>0.05 m/s)需求最高 vx={cmd[j,1]:+.4f} '
                  f'vy={cmd[j,2]:+.4f} wz={cmd[j,3]:+.4f} -> {need[j]:.4f} rad/s '
                  f'（餘裕 {WMAX - need[j]:+.4f}）')
            rec['wheel_need_max_while_moving'] = float(need[j])
        if over.any():
            print(f'      註：超出量 {float(np.max(need))-WMAX:.2e} rad/s，'
                  '屬 QP 求解器容差，非約束失效')
        rec['wheel_need_max'] = float(np.max(need))
        rec['wheel_over_frac'] = float(over.mean())

    out = os.path.join(a.run_dir, 'heading_metrics.json')
    json.dump(rec, open(out, 'w'), indent=2)
    print(f'\n  -> {out}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
