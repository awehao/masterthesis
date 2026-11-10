#!/usr/bin/env python3
"""導引候選轉接（雙相機 (b)(c)；Codex 20261111_050821）：/bottle/target_guide（bottle_seg3d_candidates_dual_node）→ ctrl.guide_intake。
在 Isaac 同程序的 bridge 節點上訂閱；主迴圈每步 tick(t_task) 依收到順序處理。訊息內時間是 ROS 模擬時鐘；轉送時換成任務時刻（ROS − T0）。
驗證（全部通過才轉送；失效關閉）：標籤含 bottle_target_guide 與欄位名、欄位齊全；t_eval 有限、非未來、嚴格遞增；guide_cam／guide_state／grasp_state 為精確整數碼；
  有導引（guide_cam ≠ 0）時 cx、cy、t_last 有限且 t_last ≤ t_eval；年齡在轉送當下由擷取戳重算（含轉送延遲，不用訊息內的 age_s）。
xy：世界 → 區域（減 region_offset）；只轉一次。reject＝1 照樣轉送（控制端受控停止）。
"""
import numpy as np

REQ = ('t_eval', 'guide_cam', 'guide_state', 'cx', 'cy', 't_first', 't_last', 'n', 'grasp_cam', 'grasp_state', 'reject')
CAMS = {0: None, 1: 'v1', 2: 'w1'}
STATES = {0: 'none', 1: 'selected', 2: 'kept', 3: 'selected_unchecked', 4: 'kept_unchecked', 5: 'inconsistent', 6: 'reject', 7: 'ambiguous_two_objects', 8: 'adopted_object_lost'}


def _code(v, allowed):
    if v is None or not np.isfinite(v) or abs(v - round(v)) > 1e-9 or int(round(v)) not in allowed:
        return None
    return int(round(v))


class GuideLink:
    def __init__(self, node, ctrl, msg_type, region_offset=(0.0, 0.0), topic='/bottle/target_guide'):
        self.ctrl, self.O = ctrl, np.asarray(region_offset, float)
        self.q, self.T0, self.last_eval = [], None, None
        self.n_fwd, self.rej, self.log = 0, {}, []
        node.create_subscription(msg_type, topic, lambda m: self.q.append((m.layout.dim[0].label if m.layout.dim else '', list(m.data))), 20)

    def set_T0(self, T0):
        self.T0 = float(T0)

    def _bad(self, why):
        self.rej[why] = self.rej.get(why, 0) + 1

    def tick(self, t_task):
        if self.T0 is None:
            return
        now_ros = float(t_task) + self.T0
        while self.q:
            lab, data = self.q.pop(0)
            if 'bottle_target_guide' not in lab or 'fields=' not in lab:
                self._bad('label')
                continue
            fs = lab.split('fields=')[1].split(',')
            if len(fs) != len(data) or any(k not in fs for k in REQ):
                self._bad('fields')
                continue
            r = dict(zip(fs, data))
            te = r['t_eval']
            if not np.isfinite(te) or te > now_ros + 1e-6 or (self.last_eval is not None and te <= self.last_eval):
                self._bad('t_eval')
                continue
            gc, gs, ks = _code(r['guide_cam'], CAMS), _code(r['guide_state'], STATES), _code(r['grasp_state'], STATES)
            rj = _code(r['reject'], (0, 1))
            if None in (gc, gs, ks, rj):
                self._bad('codes')
                continue
            g = dict(t_eval=te - self.T0, cam=CAMS[gc], state=STATES[gs], grasp_cam=CAMS[_code(r['grasp_cam'], CAMS) or 0], grasp_state=STATES[ks], reject=bool(rj))
            if gc != 0:
                if not (np.isfinite(r['cx']) and np.isfinite(r['cy']) and np.isfinite(r['t_last']) and r['t_last'] <= te + 1e-9):
                    self._bad('target')
                    continue
                g.update(xy=[float(r['cx']) - float(self.O[0]), float(r['cy']) - float(self.O[1])], t_last=float(r['t_last']) - self.T0,
                         age_s=float(now_ros - r['t_last']), n=int(r['n']))
            self.last_eval = te
            self.n_fwd += 1
            if len(self.log) < 400:
                self.log.append(dict(t=round(float(t_task), 3), cam=g['cam'], state=g['state'], grasp_state=g['grasp_state'], reject=g['reject'],
                                     xy=None if 'xy' not in g else [round(v, 5) for v in g['xy']], age_s=None if 'age_s' not in g else round(g['age_s'], 3)))
            self.ctrl.guide_intake(float(t_task), g)

    def report(self):
        return dict(n_fwd=self.n_fwd, rejected=self.rej, log=self.log[-60:])
