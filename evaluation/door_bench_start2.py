#!/usr/bin/env python3
"""台架初態啟動的純邏輯 r2（door_bench_start.py 的版本化修正；Codex r21 §1、§2；r1 保留不改）。
r1 缺口：只比對宣告與封裝 source_seq（kind＝handshake 的宣告也被接受）；真正入口／命令鏈驗證在退出凍結之後；第一物理步不核實際套用。
r2 新增：
  - on_announce 須 kind＝'solution'（另核 run_id、正整數 seq、sim_t＝t0）。
  - release_intake(...)：凍結期內把指定封裝交給**真正入口**（K3MSRC.EnvIntake.receive）與命令鏈；須回 'fed'、鏈未新增拒收、鏈未失效、同輪未中止；
    封裝 source_seq 須等於宣告。之後呼叫端清除最新封裝並在第 1 步不再收新封裝（首筆固定，不被後續回呼替換）。
  - verify_first_step(...)：第 1 物理步寫入 API 之前核對：控制者 wholebody、套用種類 wholebody_applied、套用來源 seq＝宣告且鏈模式 0、
    九軸有限、設定點＝s_init＋dt·u_arm（1e-9；承接真正寫入的 s_init）。不符 ⇒ 呼叫端閂鎖中止、不寫入。
  - StateFilter：任務端驗 /bench/state（本趟 run_id、t0 首見後不變、狀態次序 frozen → released／aborted，不回退）。
以下為 r1 說明。
台架初態啟動的純邏輯（Codex r19／r20：核心試驗專用啟動模式；執行端 ev9、求解端 ADM17、任務節點 v7 共用）。
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
        if msg.get('kind') != 'solution':
            return self._rej('kind_not_solution')
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


def release_intake(env_in, chain, env, announced_seq, t0, filt, aborted):
    """凍結期釋放當輪：把指定封裝交給真正入口與命令鏈。回傳 (ok, why, record)。"""
    if aborted:
        return False, 'aborted_same_iteration', None
    if env is None:
        return False, 'no_env', None
    if announced_seq is None or env.get('source_seq') != announced_seq:
        return False, 'seq_not_announced', None
    rej0 = int(getattr(chain, 'n_rejected', 0))
    st = env_in.receive(env, chain, float(t0), filt=filt)
    if st != 'fed':
        return False, f'intake:{st}', None
    if int(getattr(chain, 'n_rejected', 0)) != rej0:
        return False, 'chain_rejected', None
    if getattr(chain, 'fail', None) is not None:
        return False, 'chain_failed', None
    return True, None, {'source_seq': int(env['source_seq']), 'n_recv': int(chain.n_recv), 'recv_sim_t': float(t0)}


def verify_first_step(owner, kind, applied_src_seq, mode_code, u_applied, arm_setpoint, s_init, dt, announced_seq, owner_wb='wholebody'):
    """第 1 物理步（寫入 API 之前）。回傳 (ok, why, record)。"""
    rec = {'owner': owner, 'kind': kind, 'applied_source_seq': applied_src_seq, 'mode_code': mode_code}
    if owner != owner_wb:
        return False, 'owner', rec
    if kind != 'wholebody_applied':
        return False, f'kind:{kind}', rec
    if applied_src_seq != announced_seq or mode_code != 0:
        return False, 'applied_source', rec
    try:
        u = [float(x) for x in u_applied]
        sp = [float(x) for x in arm_setpoint]
    except (TypeError, ValueError):
        return False, 'nonnumeric', rec
    if len(u) != 9 or len(sp) != 6 or not all(math.isfinite(x) for x in u + sp):
        return False, 'shape_or_nonfinite', rec
    exp = [a + float(dt) * r for a, r in zip(s_init, u[3:])]
    dev = max(abs(a - b) for a, b in zip(sp, exp))
    rec.update(u_applied=u, arm_setpoint=sp, setpoint_expected=exp, setpoint_dev=dev)
    if dev > 1e-9:
        return False, 'setpoint_carry', rec
    return True, None, rec


class StateFilter:
    """任務端 /bench/state 驗證：本趟、t0 不變、次序不回退。"""
    ORDER = {'frozen': 0, 'released': 1, 'aborted': 1}

    def __init__(self, run_id):
        self.run_id = str(run_id)
        self.state, self.t0 = None, None
        self.rejects = {}

    def _rej(self, why):
        self.rejects[why] = self.rejects.get(why, 0) + 1
        return False

    def on_msg(self, msg):
        if not isinstance(msg, dict):
            return self._rej('not_dict')
        if msg.get('run_id') != self.run_id:
            return self._rej('run_id')
        st, t0 = msg.get('state'), msg.get('t0')
        if st not in self.ORDER:
            return self._rej('state')
        if isinstance(t0, bool) or not isinstance(t0, (int, float)) or not math.isfinite(float(t0)):
            return self._rej('t0')
        if self.t0 is not None and abs(float(t0) - self.t0) > 1e-9:
            return self._rej('t0_changed')
        if self.state is not None and (self.ORDER[st] < self.ORDER[self.state] or (self.ORDER[self.state] == 1 and st != self.state)):
            return self._rej('order')
        self.state, self.t0 = st, float(t0)
        return True
