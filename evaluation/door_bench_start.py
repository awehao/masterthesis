#!/usr/bin/env python3
"""台架初態啟動的純邏輯（Codex r19／r20：核心試驗專用啟動模式；執行端 ev9、求解端 ADM17、任務節點 v7 共用）。
啟動契約（預設關閉；關閉時各節點行為與原版相同）：
  1. 初態：關節位置＝q_nominal（--stow-q），致動器初始設定點＝s_init＝q_nominal − d_init（登錄 r9 bench）。名目、實測、寫入設定點分開記錄。
  2. 控制權：執行端初始控制者＝wholebody（不經 nav→wholebody 交棒；不偽稱交棒或停車閘門通過）。命令鏈以 (u＝0, 設定點＝s_init) 建立種子，
     source＝'bench_start'；之後承接的就是這個真正寫入的設定點。停車保持監看的錨點＝生成位姿（bench_anchor），另記，不寫入閘門 passed。
  3. 凍結：釋放前執行端不呼叫 world.step（模擬時間停在 t0、物理不演化），只重發 t0 的狀態與回報（設定點 ready＝1、api＝1：s_init 已寫入致動器）。
  4. 釋放條件（全部成立才釋放）：求解端 ADM17 發 /bench/first_solution（run_id 相符、source_seq 為正整數、sim_t＝t0）宣告第一筆**正常求解**的序號，
     且執行端收到的最新全身封裝其 source_seq 等於該序號（握手全零命令不算解）。牆鐘逾時 ⇒ 中止、不釋放。
  5. 求解端偏移估計初值 d0＝d_init（IntervalOffsetObserver 既有 d0 介面）；初值不計為觀測更新（n_upd 仍 0）、不冒充握手。
BenchConfig.validate：三組六軸有限值；s_init 與 q_nominal − d_init 一致（1e-9）；|d_init| ≤ offset_max（超界拒跑，不截斷）。
BenchRelease：凍結期狀態機（frozen → released／aborted），只吃已解析的訊息；不碰模擬器 API，可離線測。
"""
from __future__ import annotations

import math

RELEASE_TOPIC = '/bench/first_solution'
STATE_TOPIC = '/bench/state'
TOL = 1e-9


def _six(x, name):
    try:
        v = [float(t) for t in x]
    except (TypeError, ValueError):
        raise ValueError(f'{name} 不是數列')
    if len(v) != 6 or not all(math.isfinite(t) for t in v) or any(isinstance(t, bool) for t in x):
        raise ValueError(f'{name} 須恰為 6 個有限數')
    return v


class BenchConfig:
    def __init__(self, reg: dict, offset_max: float):
        if not isinstance(reg, dict):
            raise ValueError('bench 登錄缺')
        self.q_nominal = _six(reg.get('q_nominal'), 'q_nominal')
        self.s_init = _six(reg.get('s_init'), 's_init')
        self.d_init = _six(reg.get('d_init'), 'd_init')
        if not (isinstance(offset_max, (int, float)) and math.isfinite(offset_max) and offset_max > 0):
            raise ValueError('offset_max 須為正有限值')
        if max(abs(a - b - s) for a, b, s in zip(self.q_nominal, self.d_init, self.s_init)) > TOL:
            raise ValueError('s_init ≠ q_nominal − d_init')
        if max(abs(d) for d in self.d_init) > float(offset_max) + TOL:
            raise ValueError(f'|d_init| 超過偏移上限 {offset_max}（拒跑，不截斷）')
        self.release_timeout_wall_s = float(reg.get('release_timeout_wall_s', 120.0))
        if not (math.isfinite(self.release_timeout_wall_s) and self.release_timeout_wall_s > 0):
            raise ValueError('release_timeout_wall_s 須為正有限值')

    def check_cli(self, stow_q, setpoint_cli):
        """執行端 --stow-q／--bench-setpoint 與登錄一致（CLI 九位小數 ⇒ 1e-8）。"""
        sq, sp = _six(stow_q, 'stow_q'), _six(setpoint_cli, 'bench_setpoint')
        if max(abs(a - b) for a, b in zip(sq, self.q_nominal)) > 1e-8:
            raise ValueError('--stow-q 與 bench.q_nominal 不符')
        if max(abs(a - b) for a, b in zip(sp, self.s_init)) > 1e-8:
            raise ValueError('--bench-setpoint 與 bench.s_init 不符')
        return sq, sp


class BenchRelease:
    def __init__(self, run_id: str, t0: float, timeout_wall_s: float, wall0: float):
        self.run_id, self.t0 = str(run_id), float(t0)
        self.timeout, self.wall0 = float(timeout_wall_s), float(wall0)
        self.state = 'frozen'
        self.announced = None
        self.rejects = {}
        self.released = None
        self.n_frozen_iters = 0
        self.n_env_before_announce = 0

    def _rej(self, why):
        self.rejects[why] = self.rejects.get(why, 0) + 1
        return False

    def on_announce(self, msg):
        if self.state != 'frozen':
            return self._rej('after_' + self.state)
        if not isinstance(msg, dict):
            return self._rej('not_dict')
        if msg.get('run_id') != self.run_id:
            return self._rej('run_id')
        sq = msg.get('source_seq')
        if isinstance(sq, bool) or not isinstance(sq, (int, float)) or not math.isfinite(float(sq)) or float(sq) != int(sq) or int(sq) < 1:
            return self._rej('source_seq')
        st = msg.get('sim_t')
        if isinstance(st, bool) or not isinstance(st, (int, float)) or not math.isfinite(float(st)) or abs(float(st) - self.t0) > 1e-6:
            return self._rej('sim_t_not_t0')
        if self.announced is not None and self.announced['source_seq'] != int(sq):
            return self._rej('conflict')
        self.announced = {'source_seq': int(sq), 'sim_t': float(st)}
        return True

    def try_release(self, env_source_seq, wall_now):
        """env_source_seq：本輪執行端收到的最新全身封裝之 source_seq（沒有則 None）。回傳 'frozen'／'released'／'aborted'。"""
        if self.state != 'frozen':
            return self.state
        self.n_frozen_iters += 1
        if float(wall_now) - self.wall0 > self.timeout:
            self.state = 'aborted'
            self.released = {'why': 'release_timeout', 'wall_s': float(wall_now) - self.wall0}
            return self.state
        if env_source_seq is not None and self.announced is None:
            self.n_env_before_announce += 1
        if self.announced is not None and env_source_seq is not None and int(env_source_seq) == self.announced['source_seq']:
            self.state = 'released'
            self.released = {'source_seq': int(env_source_seq), 'wall_s': float(wall_now) - self.wall0, 'n_frozen_iters': self.n_frozen_iters}
        return self.state

    def record(self):
        return {'state': self.state, 't0': self.t0, 'announced': self.announced, 'released': self.released, 'rejects': dict(self.rejects),
                'n_frozen_iters': self.n_frozen_iters, 'n_env_before_announce': self.n_env_before_announce}
