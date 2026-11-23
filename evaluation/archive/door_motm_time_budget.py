#!/usr/bin/env python3
"""MotM 第一包（5）完整時間帳 r2：彙整已存量測（不重跑），加一個運動時間估計，輸出相依圖與時間窗。
來源：full_2 實跑紀錄（停車版時間軸、規劃 wall 與模擬暫停）、door_motm_timing_r1／fast_r1（規劃與控制拍）、door_wbopt_g1_compare_r1（G1）、
door_g2_compare_r1（G2）、door_motm_concurrency_r1（整趟控制拍與同時負載）、door_motm_reuse_check_r1（重用核對）、door_motm_grasp_fake_r1（抓持時間）、door_motm_window_r1（窗口）。
唯一新算的量：遠距擷取時的底盤／手臂到「窗口起點、預抓構型」的運動時間，用停車版同一界限模型 R2.bounded_T（停到停），
  移動版是以 v 通過窗口起點，不是停在那裡——所以只是量級估計，照實標示。
    python3 evaluation/door_motm_time_budget.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
D = os.path.join(HERE, '..', '第八次進度報告素材', '資料')
J = lambda n: json.load(open(os.path.join(D, n)))                                  # noqa: E731
R = json.load(open(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'result.json')))
L = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'log.npz'))['rows']
tim, timf, g1, g2, conc, reuse = J('door_motm_timing_r1.json'), J('door_motm_timing_fast_r1.json'), J('door_wbopt_g1_compare_r1.json'), J('door_g2_compare_r1.json'), J('door_motm_concurrency_r1.json'), J('door_motm_reuse_check_r1.json')
gr = J('door_motm_grasp_fake_r1/nominal_0.05.json') if os.path.exists(os.path.join(D, 'door_motm_grasp_fake_r1', 'nominal_0.05.json')) else None
g1c = {c['tag']: c for c in g1['cases']}

# ---- 運動時間估計（遠距擷取時狀態 → 窗口起點＋預抓構型）
import door_online_plan as OP
A0 = L[L[:, 1] == 0]
t_far = float(R['A0']['online']['far_estimate']['t_obs'])
row = A0[np.argmin(np.abs(A0[:, 0] - t_far))] if A0.size else L[0]
b_far, q_far = row[5:8], row[8:14]
E4 = L[L[:, 1] == 4][-1]; q_b = np.r_[E4[5:8], E4[8:14]]
b_win = np.r_[q_b[:2] + np.array([0.03, 0.0]), q_b[2]]
Zn = np.load(os.path.join(os.environ.get('SP', '/tmp/claude-1000/-home-howardchen-masterthesis/2a47f71f-1a91-4ff0-9dbf-1892d8540042/scratchpad'), 'mg3', 'nominal_0.05_ready.npz'))
T_app, use = OP.R2.bounded_T(b_far, q_far, b_win, q_b[3:])
READY_after_win = gr['report']['ready_t_rel'] if gr else 2.05
t_ready_est = t_far + T_app + READY_after_win

op = R['A0']['online']['plan']
out = dict(
    parking_full2_timeline=dict(
        t_far_capture_s=t_far, far_detect_ms=R['A0']['online']['far_estimate']['detect_ms'],
        standoff_reaim_plan_wall_s=op['wall_s'], split_wall_s=op['wall_split_s'], sim_clock_advance_during_plan_s=R['A0']['online']['sim_clock_advance_during_plan_s'],
        planned_motion_s=op['plan_motion_s'], near_capture_t_s=R['A0']['near']['t_obs'], near_detect_ms=R['A0']['near']['detect_ms'], replan_ms=R['A0']['replan']['wall_ms'],
        open_plan_wall_s=R['plan']['wall_s'], open_plan_sim_clock_advance_s=R['plan']['sim_clock_advance_s'],
        note='停車版：兩次規劃共 21.6＋67.4 s 都在模擬暫停中算（模擬時鐘前進 0），等待時間被藏掉；重瞄候選 18.4 s 是「停在停位再轉相機看」這一步的計算'),
    motm_compute_s=dict(
        open_plan=dict(orig_fd_full2=tim['A_cold']['solve_s'], fast_clear_full2=timf['A_cold']['solve_s'], g1_analytic_full2=g1c['full2_cold']['analytic']['wall_s'],
                       g1_analytic_motm_pre=g1c['motm_pre']['analytic']['wall_s'], g1_analytic_motm_pre_loaded=[w['wall_s'] for w in conc['bg_loaded']['planner']],
                       g1_warm_resolve=[g1c[k]['analytic']['wall_s'] for k in ('motm_warm_combined', 'motm_warm_v095', 'motm_warm_yawdist')],
                       reuse_verify=sorted({c['verify_s'] for c in reuse['cases']})[::max(1, len(reuse['cases']) // 3)]),
        perception=dict(detect_offline_alone=conc['bg_alone']['percep'], detect_offline_loaded=[conc['bg_loaded'][k] for k in ('percep_1', 'percep_2')],
                        note='每路上限 30 Hz、1 核：實際約 15 Hz（p50 約 50 ms、p95 約 135 ms）；雙相機 30 Hz 需要每路約 2 核'),
        control_tick_ms=dict(open_r7=g2['off']['tick_ms'], open_r8_g2=g2['gravity+ik_fc+sweep_cache']['tick_ms'], alone={k: conc['alone'][k]['tick_ms'] for k in ('open', 'grasp', 'exit')},
                             loaded={k: conc['loaded'][k]['tick_ms'] for k in ('open', 'grasp', 'exit')}, period_ms=50,
                             breakdown_note='開門 r8：求解 ~10 ms、閉鏈 ~25 ms、停止掃掠 ~70 ms；抓持：掃掠 83%；退出：掃掠 94%（cProfile）。掃掠每個樣本一次 clear_door_fast（約 1.8 ms，自碰撞約一半）')),
    motm_dependency_P1=[
        dict(node='far_estimate', needs=[], compute='偵測 46–89 ms／格（實跑），離線 p50 約 50 ms', when='t_far'),
        dict(node='pass_line_and_window', needs=['far_estimate'], compute='停車版對應：停位 1.0 s＋整段核對 2.2 s（實跑 wall）；重瞄 18.4 s 這一步在移動版沒有對應（不停車轉相機）；移動版的通過線／視向選擇本身尚未實作、未量', when='t_far 之後，邊走邊算'),
        dict(node='predicted_READY_state', needs=['pass_line_and_window'], compute='接近路徑＋抓持排程（插入 1.0＋閉爪 0.5＋確認 0.3＋保持 0.3）＋管線 0.08 s 可預測', when='同上'),
        dict(node='open_preplan', needs=['predicted_READY_state', 'far_estimate'], compute=f"G1 冷啟 {g1c['motm_pre']['analytic']['wall_s']} s 單跑／{min(w['wall_s'] for w in conc['bg_loaded']['planner'])}–{max(w['wall_s'] for w in conc['bg_loaded']['planner'])} s 負載", when='背景，必須在核對開始前完成'),
        dict(node='near_estimate_stream', needs=[], compute='雙相機持續（實際約 15 Hz／路／核）', when='接近全程'),
        dict(node='reuse_verify', needs=['open_preplan', 'near_estimate_stream'], compute='完整非線性重核約 0.7 s', when='必須在閉爪開始（READY−0.8 s）前完成 ⇒ 最晚 READY−1.5 s 開始，只能用「預測的」READY 狀態；READY 時實際狀態與預測的差需另有便宜核對（未設計）'),
        dict(node='fallback_warm_resolve', needs=['reuse_verify fail'], compute=f"G1 熱重解 {min(g1c[k]['analytic']['wall_s'] for k in ('motm_warm_combined', 'motm_warm_v095', 'motm_warm_yawdist'))}–{max(g1c[k]['analytic']['wall_s'] for k in ('motm_warm_combined', 'motm_warm_v095', 'motm_warm_yawdist'))} s（重用不過的三個實際案）", when='READY 後沒有時間（v105 顯示 READY 後須立即交接）；要在閉爪前完成就得在進窗口前約 2.6 s 開始 ⇒ 實際上只能「放棄這次抓取、繞過重來」或改 P2'),
        dict(node='handover_open_exec', needs=['READY', 'reuse_verify ok'], compute='開門執行層 r7 移動承接；每拍 r8 p50 104 ms（>50 ms 307/322 拍）', when='READY 後第一拍'),
    ],
    windows=dict(
        approach_motion_est_s=round(float(T_app), 2), approach_wheel_use=round(float(use), 3), approach_note='停車版界限模型 R2.bounded_T（遠距擷取時狀態 → 窗口起點、預抓構型，停到停）；移動版以 v 通過、時間不同，只是量級',
        ready_after_window_start_s=READY_after_win, t_ready_est_s=round(t_ready_est, 2), time_far_to_ready_s=round(t_ready_est - t_far, 2),
        preplan_fits=bool(t_ready_est - t_far - 1.5 > max(w['wall_s'] for w in conc['bg_loaded']['planner'])),
        preplan_slack_s=round(t_ready_est - t_far - 1.5 - max(w['wall_s'] for w in conc['bg_loaded']['planner']), 2),
        window_margin_at_ready=dict(v005_m=0.12, v005_s=2.4, v008_m='0.05–0.06', v008_s='0.6–0.75'),
        note='preplan_fits：遠距擷取到 READY−1.5 s（核對最晚開始）之間放不放得下負載下最長一次冷啟規劃；不含通過線／視向選擇自身的時間（未實作）'),
    verdict_inputs=dict(
        control_ticks_realtime=False, why_ticks='三個控制器 p50 62–109 ms 都超過 50 ms 週期；G2 等值加速後仍是，主因停止掃掠的淨空計算',
        planning_overlap='G1 後 MotM 預先規劃（含 v0，23 迭代）單跑 {:.1f} s、負載 {:.1f} s；遠距擷取→READY 估計 {:.1f} s，扣掉核對最晚開始的 1.5 s 只剩 {:.1f} s ⇒ {}（餘裕 {:+.1f} s）；停車版靜止起點冷啟 {:.1f} s 則放得下；熱重解放不進 READY 後'.format(
            g1c['motm_pre']['analytic']['wall_s'], max(w['wall_s'] for w in conc['bg_loaded']['planner']), t_ready_est - t_far, t_ready_est - t_far - 1.5,
            '放不下' if not (t_ready_est - t_far - 1.5 > max(w['wall_s'] for w in conc['bg_loaded']['planner'])) else '放得下',
            t_ready_est - t_far - 1.5 - max(w['wall_s'] for w in conc['bg_loaded']['planner']), g1c['full2_cold']['analytic']['wall_s']),
        perception_rate='雙相機 30 Hz 在單核上達不到（約 15 Hz／路）'),
)
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out['windows'], ensure_ascii=False)); print(json.dumps(out['verdict_inputs'], ensure_ascii=False))
