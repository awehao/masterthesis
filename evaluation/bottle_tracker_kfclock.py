#!/usr/bin/env python3
"""追蹤器候選：新軌跡的 KF 時鐘設在誕生量測時刻（版本化，2026-11-01；Codex 20261101_135333「先處理追蹤器的虛假高速」）。原 scan_obstacle_tracker／kalman_tracker 不改。
機制（bottle_w1_dyn2sr_g1 重建掃描回放精確重現）：KalmanTracker2D 建構時時鐘為 None，第一次 step() 只設時鐘、丟棄該筆量測（位置仍是誕生點）；
於是第二次觀測的量測被丟、第三次觀測時以「第二次→第三次」的 dt 去解釋「誕生→第三次」的累積位移 ⇒ 速度至少高估約 2 倍，誕生後若有漏觀測更多。
該趟同一移動物（0.45 m/s）的第二個叢集 21.9 s 誕生軌跡、漏 3 拍、22.3 s 重新關聯（量測被丟）、22.4 s 速度 1.42／−0.51 m/s 並在 age 3 發布為移動物。
修法：每次更新後，尚未設時鐘的軌跡（只看過誕生那一筆）把 KF 時鐘設為誕生量測時刻（last_obs_t_ns）；第二次觀測即依實際經過時間預測並更新。
不刪軌跡、不設速度上限、不改關聯／分類／發布門檻與任何參數；未確認的障礙仍走原本的靜態／未分類保護。
"""


class KFClockMixin:
    def _update_tracks(self, clusters, t_ns: int):
        super()._update_tracks(clusters, t_ns)
        for tr in self._tracks:
            if tr.kf._last_t_ns is None:
                tr.kf._last_t_ns = int(tr.last_obs_t_ns)


def make(base):
    return type('KFClock' + base.__name__, (KFClockMixin, base), {})


class AssocMahaMixin:
    """候選 2（Codex 20261101_140323）：開啟追蹤器既有的「預測位置＋馬氏距離關聯」（assoc_predict、assoc_maha），χ²、gate_max 用原預設，不另調數。
    機制：常被觀測的軌跡共變異小、閘門緊，不搶另一物體的叢集（bottle_w1c_1：第二桌兩腳相距 0.66 m < 固定閘門 0.8 m 時的跨物體跳動）；滑行軌跡共變異變大、閘門自然放寬。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.assoc_predict, self.assoc_maha = True, True


def effective_params(tr):
    keys = ('assoc_predict', 'assoc_maha', 'assoc_chi2', 'assoc_gate', 'assoc_gate_max', 'min_age', 'min_speed', 'release_speed', 'release_frames', 'min_net_speed', 'static_window_s',
            'track_timeout', 'coast_publish_s', 'fragment_merge', 'fragment_gate', 'default_radius', 'cluster_gap', 'min_pts', 'surf_max', 'surf_range', 'surf_min_sep', 'static_cbf_range',
            'use_map_subtraction', 'global_frame', 'publish_frame')
    out = {k: getattr(tr, k) for k in keys if hasattr(tr, k)}
    out['kf'] = dict(getattr(tr, '_kf_kwargs', {}))
    out['kf_clock_at_birth'] = isinstance(tr, KFClockMixin)
    return out


def make2(base):
    return type('KFClockMaha' + base.__name__, (AssocMahaMixin, KFClockMixin, base), {})
