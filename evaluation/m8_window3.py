#!/usr/bin/env python3
"""第八次第一包候選 r6：決策狀態機收斂（Codex 20261110_215340 A）。WindowLayer2（r4／r5 的評估與預測，參數不變、B 未改）之上加持續規則：
1 有效動作持續執行、失效立即處理：
  EXIT   當次目標與側別固定；每拍以「零時間餘裕」驗證剩餘離開線（靜態可行、各點機器人佔用 [t_lb, t_ub] 不與任何移動物覆蓋重疊、
         終點在抵達後 T_CLEAR 內不被覆蓋）。通過 ⇒ 繼續、不重建；不通過 ⇒ 記 exit_invalid（含原因）並立即交給 WindowLayer2 重新評估。
         到達目標（< EXIT_REACHED）後才交回評估。
  COMMIT 以零時間餘裕驗證剩餘穿越段（到離開所有覆蓋為止）；通過 ⇒ 繼續；不通過 ⇒ 立即重新評估。
  進入新動作仍要求 WindowLayer2 的 DT_MARGIN（遲滯＝進入要餘裕、持續只要不衝突），不新增時間參數。
2 停止不一律視為較保守：STOP_BLOCKED／原地停時，評估若找到可接受的 EXIT／COMMIT 就立即採用；EXIT 只在失效時才可能退回 STOP。
3 失效資料（航跡過期）照 WindowLayer2 立即 HOLD_STALE，不受持續規則延後；持續驗證每拍都用最新航跡。
回 CRUISE 仍需 T_CLEAR 1 s 連續無衝突（沿用）。
"""
import math

import numpy as np

import m8_window2 as W2
from m8_window2 import DT_MARGIN, RES, T_CLEAR, TRACK_MAX_AGE, V_LAT  # noqa: F401

EXIT_REACHED = 0.15
T_DELAY_LB, V_LB = W2.T_DELAY_LB, W2.V_LB


class WindowLayer3(W2.WindowLayer2):
    def __init__(self, blocked=None):
        super().__init__(blocked)
        self.kept = dict(EXIT=0, COMMIT=0)

    def _conflict_free(self, P, t_lb, t_ub, M, margin):
        for x, k in M:
            cov, taus = W2.cover_times(P, x, k)
            for i in np.flatnonzero(cov.any(axis=1)):
                tt = taus[cov[i]]
                if ((tt >= t_lb[i] - margin) & (tt <= t_ub[i] + margin)).any():
                    return False, dict(i=int(i), tid=x.get('tid'), kind=k)
        return True, None

    def _exit_continue(self, rp, goal, M):
        line = W2.densify(rp, goal)
        if not self._ok_line(rp, goal):
            return False, 'static'
        d = np.linalg.norm(line - rp, axis=1)
        t_lb = T_DELAY_LB + d / V_LB
        t_ub = W2.T_DELAY + d / V_LAT
        t_ub[-1] += T_CLEAR                                          # 終點抵達後仍需 T_CLEAR 不被覆蓋
        ok, info = self._conflict_free(line, t_lb, t_ub, M, 0.0)
        return ok, (None if ok else dict(why='mover', **info))

    def _commit_continue(self, rp, path, vel, M):
        hd = path[min(len(path) - 1, 4)] - rp
        v_along = max(0.0, float(np.asarray(vel) @ hd) / max(np.linalg.norm(hd), 1e-9))
        s, t_lb, t_ub = W2.robot_window(path, rp, v_along)
        cov_any = np.zeros(len(path), bool)
        for x, k in M:
            cov, _ = W2.cover_times(path, x, k)
            cov_any |= cov.any(axis=1)
        nc = np.flatnonzero(~cov_any)
        i_out = int(nc[0]) if len(nc) else len(path) - 1
        if not self._ok_line(rp, path[0]):
            return False, 'static'
        ok, info = self._conflict_free(path[:i_out + 1], t_lb[:i_out + 1], t_ub[:i_out + 1], M, 0.0)
        return ok, (None if ok else dict(why='mover', **info))

    def update(self, t, pose, vel, path, tracks, tracks_age):
        rp = np.array(pose[:2], float)
        if tracks_age is None or tracks_age > TRACK_MAX_AGE:
            return super().update(t, pose, vel, path, tracks, tracks_age)      # 失效資料立即處理
        if self.mode in ('EXIT', 'COMMIT'):
            self.hist.update(t, tracks)
            M = [(x, self.hist.kind(t, x)) for x in tracks if x.get('confirmed')]
            if self.mode == 'EXIT' and self.hold is not None and np.linalg.norm(rp - np.array(self.hold['goal'])) > EXIT_REACHED:
                ok, why = self._exit_continue(rp, np.array(self.hold['goal']), M)
                if ok:
                    self.kept['EXIT'] += 1
                    self.t_noconf = None
                    return dict(mode='EXIT', hold=self.hold, changed=False, kept=True)
                self.log.append(dict(t=round(float(t), 3), mode='EXIT', event='exit_invalid', why=why))
            elif self.mode == 'COMMIT' and len(path) and self._first_arrival(rp, M) is not None:   # 已離開所有帶 ⇒ 交回評估（經 T_CLEAR 回巡航）
                ok, why = self._commit_continue(rp, np.asarray(path, float), vel, M)
                if ok:
                    self.kept['COMMIT'] += 1
                    return dict(mode='COMMIT', kept=True)
                self.log.append(dict(t=round(float(t), 3), mode='COMMIT', event='commit_invalid', why=why))
        return super().update(t, pose, vel, path, tracks, tracks_age)
