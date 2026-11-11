#!/usr/bin/env python3
"""(b) 觀察觸發改為「導引就緒或觀察期限」，不再無條件在 X_U0 左擺（Howard 11/11；Codex 20261111_050821 (b)；版本化子類，基線＝bottle_w1_head_ctrl）。
導引候選（bottle_guide_link → guide_intake；雙相機節點的 /bottle/target_guide：xy、來源、擷取時刻、狀態）：
  1 車道（每拍，不綁 TUCK）：尚未決定目標專屬車道且有量測來源（導引新鮮；或手腕抓取候選先到——此時以實測支撐高度確認，抓取請求延後到車道成立）⇒ 先向計畫工作者要 (d) 站位選擇（bottle_standoff：區域桌高範圍皆可行 → 關節餘裕最大 → 桌腳間隙最大），
    結果到了再以選定 y_reach 交同一 lane_planner（bottle_tlane.decide；代價圖核對、太晚／擋住照舊拒絕）提早決定（導引初選；三個取樣桌高通過，不代表整段區間可行，
    抓取高度由手腕量到支撐後 plan_grasp 確認）；
    之後抓取計畫請求沿用該車道（tlane 子類只在未決定時才再決定）。
  2 抬高觀看：手臂在收攏（TUCK）時，(導引新鮮) 或 (以區域觀看點生成的抬高視角在 U 段結束時已能看到支撐——幾何必要條件，提早抬高、不必等底盤先找到)
    或 (實測 x ≥ 觀察期限 X_DEAD＝現行 X_U0，最後界線) 先到者觸發；觸發後若導引才到 ⇒ 視向改對準量測位置（重新瞄準）；q1 限 ±Q1_LIM（離線核過）；
    觀看點＝導引 xy（z＝區域觀看高度）或無導引時的區域觀看點（區域知識）；視角＝bottle_view_gen.view_q1（V_HIGH 腕部樣板、只解 q1），
    以「U 段結束時的預測底盤位姿」（路徑上 s_ref＋v_b·T_u，偏航取航向參考）生成；之後的 U→H、視向追蹤（look 改為同一觀看點）、TPA 承接照基線。
  3 拒絕：導引 reject（兩來源不一致逾時）⇒ 受控停止 dual_inconsistent；導引狀態為不一致／歧義／物件遺失 ⇒ 快取導引失效（不建車道、不重新瞄準）。
  4 視角生成失敗：不退回固定左看樣板；期限前保持收攏、下一次判斷再試，期限已到仍失敗 ⇒ 受控停止 obs_view_infeasible。
  5 站位結果綁定導引版本（xy 容差 2 mm、新鮮度）：晚到或導引已變 ⇒ 丟棄重送；全不可行 ⇒ standoff_none_feasible、工作者不可用 ⇒ planner_unavailable／dead
    （不默默退回名目站位）；控制端以實際橫向接回後的底盤時間與手臂轉移時間選擇（_pick_standoff）。
不改：抓取目標仍只來自手腕 known_type 快照（雙相機節點已交叉核對）；segtarget 限速更新、凍結、期限；GMPC、安全規則。
右側：車道規則（bottle_tlane.lane_from_target）仍只支援物件在左側；(d) 處理。
"""
import math

import numpy as np

import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_tlane as TLNm
import bottle_view_gen as VG
import bottle_w1_head_ctrl as HC
import bottle_w1_gaze_ctrl as GZ
import bottle_w1_online_ctrl as ON
import bottle_w1_tpa_ctrl as TPAm

GUIDE_MAX_AGE = 1.0
Q1_LIM = 1.5                                                    # 視角 q1 範圍：view_domain_check（收攏→視角路徑自碰撞 ≥ 5 mm、關節裕度 ≥ 0.05）
USEFUL_EVERY = 5                                                # 每 5 拍（0.25 s）判一次「抬高視角是否已可看到支撐」
Z_SUP_REGION = 0.60                                             # 區域支撐高度中點（區域知識；桌高域 [0.55, 0.65]），只供可用性判斷，不當目標
OK_STATES = ('selected', 'kept', 'selected_unchecked', 'kept_unchecked')
BLOCK_STATES = ('inconsistent', 'reject', 'ambiguous_two_objects', 'adopted_object_lost')
SO_TOL = 0.002                                                  # 站位結果所綁導引 xy 的容差（同 segtarget 一致性 2 mm）


class W1PlaceController(HC.W1PlaceController):
    group = HC.W1PlaceController.group + '_OBS'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.guide, self.obs_trig, self.guide_n, self.guide_lane = None, None, 0, None
        self.so = None                                                      # (d) 站位選擇請求（計畫工作者 kind＝standoff）
        self.guide_block, self.obs_fail = None, []
        self._wrist_src, self._gp_defer = None, None
        self.x_dead = float(self.x_u0)
        self.q_view_tmpl = np.array(self.q_vhigh, float).copy()

    # ---- 導引
    def guide_intake(self, t_recv, g):
        self.guide_n += 1
        if g.get('reject') and self.judge.stop is None:
            self._ev('dual_reject', state=g.get('state'), grasp_state=g.get('grasp_state'))
            self.judge._stop(self.t, 'dual_inconsistent', state=g.get('state'), grasp_state=g.get('grasp_state'))
            return
        if g.get('state') in BLOCK_STATES or g.get('grasp_state') in BLOCK_STATES:
            if self.guide is not None:
                self._ev('guide_invalidated', state=g.get('state'), grasp_state=g.get('grasp_state'))
            self.guide = None                                                  # 不一致／歧義／物件遺失：快取導引失效（不再用來建車道、重新瞄準）
            self.guide_block = dict(t=round(float(t_recv), 3), state=g.get('state'), grasp_state=g.get('grasp_state'))
            return
        if g.get('cam') and g.get('state') in OK_STATES and g.get('xy') is not None:
            self.guide = dict(g, t_recv=float(t_recv))

    # ---- 抓取目標有效期：由新合格快照的來源擷取時刻重新驗證（Codex 20261111_142637 第 2 項；有效期政策變更，非「原判準不變」）----
    def candidate_intake(self, t_recv, c):
        """基線採用規則不變（就緒、快照檢查 ok、目前觀測新鮮、與快照一致、序號遞增、凍結後相對首次凍結 xy／高度 ≤ 5 mm）。
        本筆被採用（凍結或凍結後更新）時，以其快照來源擷取時刻 snap_t_last 推進「最後有效驗證」；只進不退（舊影格、重複通知不推進）。
        首次凍結與幾何基準（vfix）保留不重設；鎖定後基線已不再受理（不新增刷新途徑）；導引（底盤 xy）不經此路徑，不能刷新。"""
        src = getattr(self, '_snap_src', None)
        if src is None:
            src = self._snap_src = {}
        try:
            k_, ts_ = int(c.get('snap_seq')), float(c.get('snap_t_last'))
            if k_ > 0 and np.isfinite(ts_):
                src.setdefault(k_, ts_)                                    # 快照不可變：每個序號一個來源擷取時刻（凍結時取用）
        except (TypeError, ValueError):
            pass
        seq0, n0 = self.seg_seq, len(self.seg_adopt)
        super().candidate_intake(t_recv, c)
        if self.judge.stop is not None or len(self.seg_adopt) <= n0 or self.seg_seq <= seq0 or self.vfix is None:
            return
        ts = float(c.get('snap_t_last', float('nan')))
        sv = getattr(self, 'seg_valid', None)
        if not np.isfinite(ts) or ts > float(self.t) + 1e-9 or (sv is not None and ts <= sv['t_src'] + 1e-9):
            return
        first = sv is None
        self.seg_valid = dict(t_src=ts, seq=int(self.seg_seq), t_acc=round(float(self.t), 3))
        log = getattr(self, 'seg_valid_log', None)
        if log is None:
            log = self.seg_valid_log = []
        log.append(dict(self.seg_valid, t_src=round(ts, 3)))
        self._ev('segtarget_valid_init' if first else 'segtarget_revalidated', t_src=round(ts, 3), seq=int(self.seg_seq),
                 expire_new=round(ts + float(self.seg['expire_s']), 3), expire_old_rule=round(float(self.vfix['t']) + float(self.seg['expire_s']), 3))

    def _gp_apply(self, g, res, q):
        """線上版在抓取計畫被接受時才凍結（vfix）：此時以凍結快照的來源擷取時刻初始化有效起點。
        缺有效來源擷取戳 ⇒ 受控拒絕 segtarget_valid_src_missing（Codex 150220：不退回凍結時刻冒充觀測時間）。"""
        super()._gp_apply(g, res, q)
        if self.vfix is None or getattr(self, 'seg_valid', None) is not None or self.judge.stop is not None:
            return
        ts = (getattr(self, '_snap_src', None) or {}).get(int(self.vfix['seq']))
        if ts is None or not np.isfinite(ts) or ts > float(self.t) + 1e-9:
            self.judge._stop(self.t, 'segtarget_valid_src_missing', seq=int(self.vfix['seq']), t_src=ts)
            return
        self.seg_valid = dict(t_src=float(ts), seq=int(self.vfix['seq']), t_acc=round(float(self.t), 3), init='freeze_snapshot')
        log = getattr(self, 'seg_valid_log', None)
        if log is None:
            log = self.seg_valid_log = []
        log.append(dict(self.seg_valid, t_src=round(self.seg_valid['t_src'], 3)))
        self._ev('segtarget_valid_init', t_src=round(self.seg_valid['t_src'], 3), seq=int(self.vfix['seq']), init=self.seg_valid['init'],
                 expire_new=round(self.seg_valid['t_src'] + float(self.seg['expire_s']), 3), expire_old_rule=round(float(self.vfix['t']) + float(self.seg['expire_s']), 3))

    def _seg_valid_t(self):
        sv = getattr(self, 'seg_valid', None)
        return self.vfix['t'] if sv is None else sv['t_src']

    def _seg_expire_info(self):
        sv = getattr(self, 'seg_valid', None)
        return dict(policy='last_valid_src_capture', t_src=None if sv is None else round(sv['t_src'], 3), seq=None if sv is None else sv['seq'],
                    first_freeze_t=round(float(self.vfix['t']), 3), expire_old_rule=round(float(self.vfix['t']) + float(self.seg['expire_s']), 3),
                    n_valid=len(getattr(self, 'seg_valid_log', []) or []))

    def _guide_fresh(self):
        g = self.guide
        return g is not None and float(self.t) - float(g['t_last']) <= GUIDE_MAX_AGE

    def _lane_src(self):
        """車道／站位的量測來源：導引新鮮 ⇒ 導引（xy；支撐未知 ⇒ 區域取樣高度）；否則手腕抓取候選先到（延後中）⇒ 手腕（xy＋實測支撐高度，抓取確認）。
        都沒有 ⇒ None。不退回名目模板（Codex 134437）。"""
        if self._guide_fresh():
            g = self.guide
            return dict(xy=list(g['xy']), t=float(g['t_last']), cam=g['cam'], z_list=None)
        w = self._wrist_src
        if w is not None and float(self.t) - float(w['t']) <= ON.OBS_FRESH_S:
            return dict(xy=list(w['xy']), t=float(w['t']), cam='w1_grasp', z_list=[float(w['z'])])
        return None

    def _d_stop(self):
        v = max(float(getattr(self, 'v_b', 0.0)), ON.V_STOP_FLOOR)            # 同 bottle_w1_tlane_ctrl 的煞停距離
        return v * (C0.D_STEPS * C0.PHYS + 0.05) + v * v / (2.0 * float(B.A_BASE[0])) + 0.02

    def _pick_standoff(self, rows, g):
        """控制端選擇：可行候選逐一以 bottle_tlane.rejoin 算實際橫向接回後的路徑，估底盤到 x_pre 的時間（名目剖面）與手臂到該分支的轉移時間，
        取 max(兩者) 最短（0.1 s 內視為同）→ 最小關節餘裕最大 → 桌腳間隙最大。回 (選定列或 None, 每列成本)。g＝量測來源（_lane_src）。"""
        sig_join = float(self.sig_robot) + self._d_stop() + 0.02
        q_now = self._arm_now()
        win_lo = float(self.win[0]) - float(self.x_nom)
        out = []
        for r in rows:
            if not r.get('ok'):
                continue
            x_entry, y_lane = TLNm.lane_from_target(g['xy'], win_lo, r['y_reach'])
            try:
                new = TLNm.PlannedPathLane(TLNm.rejoin(self.path, sig_join, x_entry, y_lane), x_entry, y_lane)
            except Exception as e:                                             # noqa: BLE001
                out.append(dict(r, cost=None, why_cost=f'rejoin:{e!r}'))
                continue
            ds = float(new.sig0 - self.path.sig0)
            x_pre_c = float(g['xy'][0]) + win_lo + 0.05
            t_base = self._standoff_t_base(new, float(self.sig_robot) + ds, x_pre_c)
            t_arm = 0.0 if r.get('q_pre_lo') is None else 1.875 * float(np.abs(np.asarray(r['q_pre_lo'], float) - q_now).max()) / B.W_ARM['v_arm']
            out.append(dict(r, cost=round(max(t_base, t_arm), 3), t_base=round(t_base, 3), t_arm=round(t_arm, 3), dsig0=round(ds, 4)))
        feas = [r for r in out if r.get('cost') is not None]
        if not feas:
            return None, out
        best = min(feas, key=lambda r: (round(r['cost'], 1), -r['margin_min'], -(r['base_leg_clear'] or 0.0)))
        return best, out

    def _standoff_t_base(self, path_new, sig, x_pre_c):
        """站位候選的底盤時間成本（hook；預設＝既有梯形剖面；A0 子類改用同源剖面）。"""
        return TPAm.base_time(sig, x_pre_c, x_pre_c - 0.6, float(self.v_nav), float(self.v_c), float(self.v_b))

    def _standoff(self, g):
        """(d) 導引新鮮且尚未決定車道 ⇒ 送站位選擇請求（綁定導引版本＝xy）；結果到了若導引已變（> SO_TOL）或過期 ⇒ 丟棄、下一拍以新導引重送。
        全不可行 ⇒ 受控停止 standoff_none_feasible；工作者不可用 ⇒ planner_unavailable／planner_dead（不默默退回名目站位）。回 y_reach 或 None（未到）。
        g＝量測來源（_lane_src：導引或手腕；手腕帶實測支撐高度 z_list）。"""
        if self.so is None or self.so['state'] == 'stale':
            q_now = self._arm_now()
            qb = q_now.copy()
            qb[0] = math.atan2(0.4275, 0.35)                                   # 方位種子：車道未定，以名目站位與窗起點的相對方位近似（只是 IK 種子，不是姿態先驗）
            import bottle_s1_geom as G
            legs = [(c.tolist(), sz.tolist()) for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]   # 已知家具（區域先驗桌腳）
            inp = dict(xy=list(g['xy']), band=0.10, pre_m=0.06, lift_m=0.05, q_seeds=[np.asarray(z_, float).tolist() for z_ in [q_now, qb] + list(self._rng_seeds)],
                       legs=legs, y_tube=list(B.Y_TUBE))
            if g.get('z_list'):
                inp['z_list'] = list(g['z_list'])                              # 手腕先到：以實測支撐高度確認（不是區域取樣）
            rid = self.pc.submit('standoff', inp)
            n_req = 1 if self.so is None else self.so.get('n_req', 1) + 1
            self.so = dict(id=rid, t_req=round(float(self.t), 3), state='pending', xy=list(g['xy']), t_src=float(g['t']), src=g['cam'], z_list=g.get('z_list'), n_req=n_req)
            self._ev('standoff_request', id=rid, n=n_req)
            if rid is None:
                self.so['state'] = 'not_sent'
                self.judge._stop(self.t, 'planner_unavailable', kind='standoff', alive=self.pc.alive())
            return None
        if self.so['state'] != 'pending':
            return self.so.get('y_reach')
        m = self.pc.poll(self.so['id'])
        if m is None:
            if not self.pc.alive():
                self.judge._stop(self.t, 'planner_dead', kind='standoff')
            return None
        res = m.get('result') or {}
        self.so.update(t_res=round(float(self.t), 3), ok=bool(m['ok']), error=m.get('error'), rows=res.get('rows'), compute_wall=round(m['wall']['t1'] - m['wall']['t0'], 3))
        if m.get('t_wall_submit') is not None and m.get('t_wall_recv') is not None:          # 工作者排隊（送出→開始）、回傳（完成→控制端取回）牆鐘
            self.so.update(queue_wall=round(m['wall']['t0'] - m['t_wall_submit'], 3), return_wall=round(m['t_wall_recv'] - m['wall']['t1'], 3))
        hist = getattr(self, 'so_hist', None)
        if hist is None:
            hist = self.so_hist = []
        hist.append({k: self.so.get(k) for k in ('id', 'n_req', 't_req', 't_res', 'src', 'z_list', 'ok', 'compute_wall', 'queue_wall', 'return_wall')})
        if g is None or float(np.hypot(*(np.asarray(g['xy'], float) - np.asarray(self.so['xy'], float)))) > SO_TOL or (g.get('z_list') is not None and self.so.get('z_list') is None):
            self.so['state'] = 'stale'                                         # 結果晚到、導引已更新或過期：丟棄，下一拍以新導引重送
            self._ev('standoff_stale', id=self.so['id'])
            return None
        best, costs = self._pick_standoff(res.get('rows') or [], g) if m['ok'] else (None, [])
        self.so['costs'] = [{k: v for k, v in c.items() if k != 'q_pre_lo'} for c in costs]
        if best is None:
            self.so.update(state='none_feasible', y_reach=None)
            self._ev('standoff_result', ok=False, wall=self.so['compute_wall'])
            self.judge._stop(self.t, 'standoff_none_feasible', error=m.get('error'))
            return None
        self.so.update(state='done', y_reach=float(best['y_reach']), chosen={k: v for k, v in best.items() if k != 'q_pre_lo'})
        self._ev('standoff_result', ok=True, y_reach=self.so['y_reach'], cost=best['cost'], wall=self.so['compute_wall'])
        return self.so['y_reach']

    def _lane_decide(self):
        """每拍（不綁 TUCK）：車道尚未成立 ⇒ 依量測來源走站位選擇，結果到了才以選定 y_reach 建車道。"""
        if self.tlane is not None or self.lane_planner is None or self.judge.stop is not None:
            return
        g = self._lane_src()
        if g is None:
            return
        yr = self._standoff(g)                                                 # 在途請求的來源若已換（導引過期、手腕先到）：結果回來時版本核對丟棄並以新來源重送
        if yr is None:
            return                                                             # 站位結果未到、已丟棄重送或已受控停止
        d_stop = self._d_stop()
        r = self.lane_planner(dict(t=float(g['t']), xy=list(g['xy'])), float(self.t), float(self.sig_robot), d_stop, y_reach=yr)
        rec = {k: v_ for k, v_ in r.items() if k != 'path_new'}
        rec.update(t=round(float(self.t), 3), seq=-1, sig_robot=round(float(self.sig_robot), 4), d_stop=round(d_stop, 4), source=g['cam'])
        self.tlane_log.append(rec)
        self.guide_lane = rec
        if not r['ok']:
            self._ev('tlane_reject', why=r['why'], source=g['cam'])
            if g['cam'] == 'w1_grasp':
                self.judge._stop(self.t, 'target_lane_rejected', why=r['why'], source='standoff')   # 手腕確認後仍不通過：同基線受控停止
            return
        ds = float(r['dsig0'])
        self.path = r['path_new']
        self.s_ref = float(self.s_ref) + ds
        self.sig_robot = float(self.sig_robot) + ds
        self.tlane = rec
        self._ev('tlane_accept', x_entry=r['x_entry'], y_lane=r['y_lane'], sig_join=r['sig_join'], dsig0=ds, cost_max=r.get('cost_max'), source=g['cam'], y_reach=yr)

    _lane_from_guide = _lane_decide                                            # 舊名（離線反例腳本沿用）

    def _gp_submit(self, xy, zs, seq):
        """手腕抓取候選先到而車道未成立：不讓父類建名目車道（會繞過站位選擇）——記為延後，待站位確認、車道成立後再送抓取計畫。"""
        if self.tlane is None:
            self._wrist_src = dict(xy=[float(xy[0]), float(xy[1])], z=float(zs), seq=int(seq), t=float(self.t))
            self._gp_defer = (np.asarray(xy, float).copy(), float(zs), seq)
            self._ev('grasp_submit_deferred', seq=int(seq))
            self._lane_decide()
            if self.tlane is None:
                return
        self._gp_defer = None
        super()._gp_submit(xy, zs, seq)

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and self.tlane is None:
            self._lane_decide()
        if self.judge.stop is None and self.tlane is not None and self._gp_defer is not None:
            xy, zs, seq = self._gp_defer
            self._gp_defer = None
            self._ev('grasp_submit_released', seq=int(seq))
            self._gp_submit(xy, zs, seq)                                       # 車道已成立：照基線送抓取計畫（tlane 子類見車道已定不再建名目車道）
        super()._advance_inner(T_tcp, close_done, q)

    # ---- 觀察觸發
    def _view_for(self, sq, look):
        """以「U 段結束時的預測底盤位姿」生成觀看視角；q1 限在已核範圍。回 (q_obs, 資訊, T_u, 預測底盤位姿)。"""
        T_u = max(1.0, math.ceil(1.875 * float(np.abs(self.q_view_tmpl - np.asarray(sq['qb'])).max()) / B.W_ARM['v_arm'] / 0.05) * 0.05)   # 同 segtarget_vhigh._T
        s_end = float(self.s_ref) + float(self.v_b) * T_u
        pxy = self.path.xy(np.array([s_end]))[0]
        bp = np.r_[pxy, float(self.hd.yaw(s_end)) if self.hd is not None else 0.0]
        q_obs, info = VG.view_q1(self.K, bp, self.q_view_tmpl, look, -Q1_LIM, Q1_LIM)
        if q_obs is None:
            return None, info, T_u, bp
        return np.asarray(q_obs, float), info, T_u, bp

    def _useful(self, q_obs, bp, look):
        """抬高到 q_obs 時，觀看點下方支撐（z＝區域支撐中點）與瓶頂是否在畫面內、俯角 ≥ 校準門檻、相機距離 ≤ VG.D_MAX（幾何必要條件，不代表辨識成功）。"""
        P = VG.bottle_pts(float(look[0]), float(look[1]), Z_SUP_REGION)
        return bool(VG.visible(self.K, bp, q_obs, P, None)['ok'])          # 含距離上限 VG.D_MAX（解析度只校準到 1.79 m）

    def _vstep(self, q):
        sq = self.vh_seq
        if self.judge.stop is None and sq['ph'] == 'TUCK':
            x = float(q[0])
            if self.obs_trig is None:
                guide = self._guide_fresh()
                look = np.r_[self.guide['xy'], GZ.LOOK[2]] if guide else np.asarray(self.look, float)
                why, cand = None, None
                if guide:
                    why = 'guide'
                elif self.n_tick % USEFUL_EVERY == 0 or x >= self.x_dead:
                    cand = self._view_for(sq, look)
                    if cand[0] is not None and self._useful(cand[0], cand[3], look):
                        why = 'view_useful'                                    # 依區域與當下構型提早抬高（不必等底盤先找到）
                    elif x >= self.x_dead:
                        why = 'deadline'
                if why is not None:
                    if not self._at(q, sq['qb']):
                        self._wait('tuck_not_reached', x)
                        return
                    q_obs, info, T_u, bp = cand if cand is not None else self._view_for(sq, look)
                    if q_obs is None:                                          # 視角生成失敗：不退回固定左看樣板（Codex 054341）
                        self.obs_fail.append(dict(t=round(float(self.t), 3), why=why, x=round(x, 4), info=info))
                        self._ev('obs_view_failed', why=why, x=round(x, 4))
                        if x >= self.x_dead:
                            self.judge._stop(self.t, 'obs_view_infeasible', x=round(x, 4), n_fail=len(self.obs_fail))   # 期限已到仍無可行視角：受控拒絕
                        return                                                 # 期限前：保持收攏，下一次判斷再試（有界：期限即最後界線）
                    self.q_vhigh = q_obs
                    self.look = np.asarray(look, float)                       # 視向追蹤對準同一觀看點
                    self.obs_trig = dict(t=round(float(self.t), 3), why=why, x=round(x, 4), look=np.round(look, 4).tolist(), q1=round(float(q_obs[0]), 4),
                                         T_u=T_u, view=info, guide_cam=self.guide['cam'] if why == 'guide' else None)
                    self._ev('obs_trigger', why=why, x=round(x, 4), q1=round(float(q_obs[0]), 4), guide_cam=self.obs_trig['guide_cam'])
                    self._vset('U', sq['qb'], self.q_vhigh, x)
                    return
                return                                                         # 未觸發：保持收攏，不讓基線在 X_U0 自行左擺
        if self.obs_trig is not None and self.obs_trig['why'] != 'guide' and self._guide_fresh() and sq['ph'] in ('U', 'H'):
            new_look = np.r_[self.guide['xy'], GZ.LOOK[2]]
            if float(np.linalg.norm(new_look - np.asarray(self.look, float))) > 1e-4:
                self.look = new_look                                           # 有導引後以量測位置重新瞄準（視向偏移有界追蹤，不跳設定點）
                self.obs_trig['reaim'] = dict(t=round(float(self.t), 3), cam=self.guide['cam'])
                self._ev('obs_reaim', cam=self.guide['cam'])
        super()._vstep(q)

    def report(self):
        r = super().report()
        r['seg_validity'] = dict(policy='有效期 15 s 由最後有效驗證（新合格抓取快照）的來源擷取時刻起算；首次凍結與幾何基準不重設（Codex 142637）',
                                 log=getattr(self, 'seg_valid_log', []), first_freeze_t=None if self.vfix is None else round(float(self.vfix['t']), 3),
                                 expire_old_rule=None if self.vfix is None else round(float(self.vfix['t']) + float(self.seg['expire_s']), 3),
                                 expire_new=None if getattr(self, 'seg_valid', None) is None else round(self.seg_valid['t_src'] + float(self.seg['expire_s']), 3))
        r['obs'] = dict(trigger=self.obs_trig, standoff=self.so, standoff_hist=getattr(self, 'so_hist', [])[-20:], view_fail=self.obs_fail[-20:], guide_block=self.guide_block, x_dead=self.x_dead, guide_n=self.guide_n, guide_last=None if self.guide is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.guide.items()},
                        guide_lane=self.guide_lane, guide_max_age=GUIDE_MAX_AGE,
                        note='觀察觸發＝導引就緒或觀察期限（X_DEAD＝現行 X_U0）先到者；觀看點＝導引 xy 或區域觀看點；車道可由導引提早決定')
        return r
