"""R2：殼—橫桿的精確距離下界，是否真的接進屏障（離線，不開模擬器）。

驗收條件逐項：
  1. 輸出當前位姿的下界、上界與耗時
  2. 與既有保持位姿的參考區間 7.4160–7.4415 mm 相容
  3. 下界**真的進入**安全濾波器的距離項（不是另寫一份紀錄）
  4. 速度修正**全部保留**（omega x rho、velocity_error_margin、相對速度項）
  5. 逾時或資料無效時**不沿用上一筆**，退回既有 d − rho
"""
from __future__ import annotations
import os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import (                        # noqa
    VOBS_STATIC, forced_pair_rows, pair_distance_bound, parse_obstacles)
from ammr_wholebody_mpc.arm_link_geometry import (                        # noqa
    link_collision_tris, sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics    # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (                   # noqa
    DetectionPoint, SafetyConfig, STATUS_OK, STATUS_STALE, _rows_from_points)
from coman_zero_cmd_feasible import hold_state                             # noqa

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
LINK, OBST = 'uflite_gripper_link', 'handle_bar'
REF = (7.4160, 7.4415)          # mm，§26 的離線參考區間


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:54s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    st = hold_state()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    q = np.zeros(len(K.dof_names))
    q[0], q[1], q[2] = st['park']
    q[3:9] = st['q_arm']
    xml = open(URDF, encoding='utf-8').read()
    sp = subprocess.run([sys.executable, os.path.join(HERE,
                         'coman_obstacle_specs.py'), str(st['unit'][0]),
                         str(st['unit'][1])], capture_output=True, text=True,
                        check=True)
    obs = parse_obstacles([x for x in sp.stdout.split() if x.strip()])
    T_db = np.eye(4)
    T_db[0, 3] = st['unit'][0]
    T_db[1, 3] = st['drawer_y0'] - st['opening']
    for o in obs:
        if o.model == 'drawer_body':
            o.T_world_link = T_db
    names = [o.name for o in obs]
    bar = obs[names.index(OBST)]
    tl = link_collision_tris(xml, links=[LINK])[LINK]

    # ---- 1 輸出下界、上界、耗時（完整路徑：FK + 變換 + 分支界限）----
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        T = K.fk(q, LINK)
        tris = (tl.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(-1, 3, 3)
        r = pair_distance_bound(tris, bar, tol=5.0e-4, max_tris=60000,
                                budget_s=0.020)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000.0
    print(f'  下界 {r["lb"]*1000:.4f} mm、上界 {r["ub"]*1000:.4f} mm、'
          f'gap {r["gap"]*1000:.4f} mm、{r["rounds"]} 輪、{r["n_tris"]} 三角形')
    print(f'  完整路徑耗時 p50 {np.median(ts):.2f} ms、max {ts.max():.2f} ms'
          f'（控制週期 50 ms）')
    check('1 下界與上界皆為有限值', np.isfinite(r['lb']) and np.isfinite(r['ub']))
    check('1 達到設定容差', bool(r['tol_met']), f'  reason={r["reason"] or "—"}')

    # ---- 2 與參考區間相容 ----
    lb_mm, ub_mm = r['lb'] * 1000.0, r['ub'] * 1000.0
    check('2 下界 <= 參考區間下端（是有效下界）', lb_mm <= REF[0] + 1e-9,
          f'  {lb_mm:.4f} <= {REF[0]}')
    check('2 上界 >= 參考區間下端（未低估真實距離）', ub_mm >= REF[0] - 1e-9)
    check('2 上界與參考上端相差 < 0.01 mm', abs(ub_mm - REF[1]) < 0.01,
          f'  |{ub_mm:.4f} − {REF[1]}| = {abs(ub_mm-REF[1]):.4f}')
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)[LINK]
    W = (S.points @ T[:3, :3].T) + T[:3, 3]
    from ammr_wholebody_mpc.arm_link_geometry import obstacle_distances
    d_s, _, _ = obstacle_distances(W, [bar])
    old = float(d_s.min()) - S.rho
    check('2 比既有 d − rho 緊得多', r['lb'] > old,
          f'  {lb_mm:.3f} mm vs {old*1000:.3f} mm')

    # ---- 3 下界真的進入濾波器的距離項 ----
    cfg = SafetyConfig()
    cfg.phase = None
    rows = forced_pair_rows(W, S.points, bar, names.index(OBST), li=0,
                            rho=S.rho, status=STATUS_OK, age=0.0,
                            occ_of=lambda p, v: 0.0, max_range=3.0,
                            vobs_of=None, d_lb=r['lb'])
    check('3 必要配對列帶出下界與有效旗標',
          bool(rows) and abs(rows[0][20] - r['lb']) < 1e-12
          and rows[0][21] == 1.0)
    row = rows[0]
    mk = lambda dlb: DetectionPoint(
        LINK, np.array(row[0:3]), np.array(row[3:6]), float(row[6]),
        int(row[7]), float(row[8]), row[9] >= 0.5,
        offset=np.array(row[11:14]), rho=float(row[14]), obs=OBST,
        v_obs=np.zeros(3), v_obs_state=VOBS_STATIC, d_lb=dlb)
    A_n, b_n, _, _ = _rows_from_points(K, q, [mk(None)], cfg, np.zeros(9))
    A_t, b_t, _, _ = _rows_from_points(K, q, [mk(r['lb'])], cfg, np.zeros(9))
    d_stop0 = cfg.d0 + cfg.eps
    check('3 無下界時 rhs 用 d − rho',
          abs(float(b_n[0]) - cfg.alpha * (old - d_stop0)) < 1e-12)
    check('3 有下界時 rhs 改用下界（**確實生效**）',
          abs(float(b_t[0]) - cfg.alpha * (r['lb'] - d_stop0)) < 1e-12,
          f'  {float(b_t[0]):+.5f} vs {float(b_n[0]):+.5f}')
    check('3 下界未被重複扣 rho',
          abs(float(b_t[0]) - cfg.alpha * (r['lb'] - S.rho - d_stop0)) > 1e-6)
    check('3 餘量因此放寬但**仍為負**（門檻未改）', float(b_t[0]) < 0.0)

    # ---- 4 速度修正全部保留 ----
    v_in = np.zeros(9)
    v_in[1] = 0.05
    # **關節速度**才會讓連桿有角速度；底盤純平移的 omega 為零，
    # omega x rho 那一項本來就不會生效（先前這裡設錯，量不到該修正）。
    v_in[3] = 0.10
    A1, b1, _, _ = _rows_from_points(K, q, [mk(r['lb'])], cfg, v_in)
    check('4 有接近速度時 rhs 變小（v·tau 與 v²/2a 仍計入）',
          float(b1[0]) < float(b_t[0]))
    cfg_w = SafetyConfig(use_omega_rho_margin=False)
    A2, b2, _, _ = _rows_from_points(K, q, [mk(r['lb'])], cfg_w, v_in)
    check('4 omega x rho 速度修正仍在（關掉會使 rhs 變大）',
          float(b2[0]) > float(b1[0]))
    cfg_m = SafetyConfig(velocity_error_margin=0.01)
    A3, b3, _, _ = _rows_from_points(K, q, [mk(r['lb'])], cfg_m, v_in)
    check('4 velocity_error_margin 仍套用', float(b3[0]) < float(b1[0]) - 0.009)
    p_rel = mk(r['lb'])
    p_rel.v_obs = np.array([0.0, -0.05, 0.0])
    p_rel.v_obs_state = 2
    A4, b4, _, _ = _rows_from_points(K, q, [p_rel], cfg, v_in)
    check('4 相對速度項仍生效（迎面更嚴）', float(b4[0]) < float(b1[0]))

    # ---- 5 失效不沿用上一筆 ----
    for label, dlb in (('下界為 nan', float('nan')),
                       ('下界為 None（計算失敗）', None)):
        A5, b5, _, _ = _rows_from_points(K, q, [mk(dlb)], cfg, np.zeros(9))
        check(f'5 {label} ⇒ 退回 d − rho，不冒充有效值',
              abs(float(b5[0]) - cfg.alpha * (old - d_stop0)) < 1e-12)
    bad_r = pair_distance_bound(np.zeros((0, 3, 3)), bar)
    check('5 空三角形陣列回傳 nan 並附原因',
          not np.isfinite(bad_r['lb']) and bool(bad_r['reason']))
    nan_r = pair_distance_bound(np.full((2, 3, 3), np.nan), bar)
    check('5 含非有限值回傳 nan 並附原因',
          not np.isfinite(nan_r['lb']) and bool(nan_r['reason']))
    bar2 = parse_obstacles([f'{OBST}:drawer_body:cylinder:0.005,0.2:0,0,0'])[0]
    check('5 障礙物位姿未知回傳 nan',
          not np.isfinite(pair_distance_bound(tris, bar2)['lb']))
    tiny = pair_distance_bound(tris, bar, tol=1e-9, max_tris=4000)
    check('5 預算截斷後 lb 仍是**有效下界**（只是較鬆）',
          np.isfinite(tiny['lb']) and tiny['lb'] <= r['ub'] + 1e-12
          and not tiny['tol_met'], f'  lb {tiny["lb"]*1000:.3f} mm')
    st_pt = mk(r['lb'])
    st_pt.status = STATUS_STALE
    st_pt.age = 0.1
    A6, b6, _, _ = _rows_from_points(K, q, [st_pt], cfg, np.zeros(9))
    check('5 stale 時以下界為基底再收縮（不因過期取得更大距離）',
          float(b6[0]) < float(b_t[0]))

    # ---- 6 一般列也要套到下界（否則它會壓過較緊的下界）----
    # 重現節點的一般帶狀列：最近障礙物是橫桿，該列若仍用 d − rho，
    # 就算另外補了帶下界的配對列也沒有用 —— 較嚴的那一列會決定結果。
    dmin = float(d_s.min())
    sel = np.nonzero(d_s <= dmin + S.rho)[0]
    sel = sel[np.argsort(d_s[sel])]          # 節點是依距離排序後取用
    band_no_lb, band_lb = [], []
    for k in sel[:3]:
        nh = np.array([0.0, 1.0, 0.0])
        band_no_lb.append(DetectionPoint(
            LINK, W[k], nh, float(d_s[k]), STATUS_OK, 0.0, False,
            offset=S.points[k], rho=S.rho, obs=OBST))
        band_lb.append(DetectionPoint(
            LINK, W[k], nh, float(d_s[k]), STATUS_OK, 0.0, False,
            offset=S.points[k], rho=S.rho, obs=OBST, d_lb=r['lb']))
    _, b_band0, _, _ = _rows_from_points(K, q, band_no_lb, cfg, np.zeros(9))
    _, b_band1, _, _ = _rows_from_points(K, q, band_lb, cfg, np.zeros(9))
    check('6 一般列未套下界時，最嚴的列仍是 d − rho',
          abs(float(min(b_band0)) - cfg.alpha * (old - d_stop0)) < 1e-9)
    check('6 一般列套上下界後，最嚴的列改由下界決定',
          abs(float(min(b_band1)) - cfg.alpha * (r['lb'] - d_stop0)) < 1e-9,
          f'  {float(min(b_band1)):+.5f} vs {float(min(b_band0)):+.5f}')
    check('6 混合時（一般列無下界、配對列有）較嚴者仍是無下界那條',
          float(min(list(b_band0) + list(b_t))) == float(min(b_band0)))

    print('精確距離下界離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
