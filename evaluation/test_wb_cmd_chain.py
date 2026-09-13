"""命令鏈的純邏輯測試：**不開模擬器**。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb_cmd_chain import CmdChain, N_DOF

DT = 0.01
LO = (-2.0,) * 6
HI = (2.0,) * 6
fails = []


def ok_wheels(vx, vy, wz):
    return (True, None)


def slow_only(limit=0.05, wlim=0.2):
    """尚未實作正常輸出限制時的介面測試用：明確低速界限、越界即中止。"""
    def f(vx, vy, wz):
        if (vx * vx + vy * vy) ** 0.5 > limit:
            return (False, f'底盤線速度 > {limit} m/s')
        if abs(wz) > wlim:
            return (False, f'底盤角速度 > {wlim} rad/s')
        return (True, None)
    return f


def mk(**kw):
    d = dict(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=ok_wheels,
             joint_lower=LO, joint_upper=HI)
    d.update(kw)
    return CmdChain(**d)


def check(name, cond, detail=''):
    print(f'  {"通過" if cond else "**未通過**"}  {name}' + (f'  {detail}' if detail else ''))
    if not cond:
        fails.append(name)


print('A 結構有效性：長度、有限值')
c = mk()
check('長度 8 整筆拒收', not c.receive([0.0] * 8, 0.0), c.last_reject)
check('整體失效已設定', c.fail is not None, c.fail)
c = mk()
check('含 NaN 整筆拒收', not c.receive([0.0] * 4 + [float('nan')] + [0.0] * 4, 0.0),
      c.last_reject)
c = mk()
check('含 Inf 整筆拒收', not c.receive([float('inf')] + [0.0] * 8, 0.0), c.last_reject)
c = mk()
check('長度 9 且全有限 ⇒ 接受', c.receive([0.0] * 9, 0.0), '')

print('B 同一物理步使用同一份快照')
c = mk()
c.receive([0.01, 0, 0] + [0.1] * 6, 0.0)
r1 = c.step(0.0, DT, [0.0] * 6)
s_used = c.applied
c.receive([0.02, 0, 0] + [0.2] * 6, 0.005)   # 同一步內又來一筆
check('已取用的快照不被本步覆寫', c.applied is s_used,
      f'recv_seq={s_used.recv_seq}')
r2 = c.step(0.01, DT, [0.0] * 6)
check('下一步才換新快照', c.applied.recv_seq == 2, f'recv_seq={c.applied.recv_seq}')

print('C 手臂積分：初始化、限位、過期凍結')
c = mk()
c.receive([0, 0, 0] + [0.5] * 6, 0.0)
q0 = [0.3, -0.2, 0.1, 0.0, 0.4, -0.1]
base, sp = c.step(0.0, DT, q0)
check('初始設定點由實測關節位置建立',
      all(abs(a - b) < 1e-12 for a, b in zip(sp, [x + 0.5 * DT for x in q0])),
      f'首步 {tuple(round(x,4) for x in sp)}')
check('初始化事件有記錄',
      any(e[1] == 'setpoint_init' for e in c.events))
for k in range(1, 50):
    c.receive([0, 0, 0] + [0.5] * 6, k * DT)
    c.step(k * DT, DT, [9.9] * 6)            # 量測值故意亂給
check('初始化後不再被量測值覆寫', abs(c.setpoint[0] - (q0[0] + 0.5 * 0.5)) < 1e-9,
      f'setpoint[0]={c.setpoint[0]:.4f}')

c = mk()
c.receive([0, 0, 0] + [1.0] + [0.0] * 5, 0.0)
t = 0.0
while c.fail is None and t < 5.0:
    c.receive([0, 0, 0] + [1.0] + [0.0] * 5, t)
    c.step(t, DT, [1.99] * 6)
    t += DT
check('積分超出限位 ⇒ 整體失效', c.fail is not None and '限位' in c.fail, c.fail)

print('D 命令過期：停止積分，設定點凍結')
c = mk(max_cmd_age_s=0.05)
c.receive([0.01, 0, 0] + [0.5] * 6, 0.0)
c.step(0.0, DT, [0.0] * 6)
sp_before = tuple(c.setpoint)
out = c.step(0.30, DT, [0.0] * 6)            # 已過期
check('過期時底盤分量歸零', out is not None and out[0] == (0.0, 0.0, 0.0), str(out[0]))
check('過期時設定點凍結', out[1] == sp_before, '')
check('停止積分有計數', c.n_frozen == 1, f'n_frozen={c.n_frozen}')
check('凍結不等於實際速度為零（僅語意，須另量測）', True,
      '見 summary 的 fail_note / 規格 §1.1')

print('E 輪級檢查：越界即整體失效，不只縮底盤')
c = mk(wheel_ok=slow_only(0.05, 0.2))
c.receive([0.03, 0.0, 0.0] + [0.0] * 6, 0.0)
check('低速通過', c.step(0.0, DT, [0.0] * 6) is not None)
c.receive([0.50, 0.0, 0.0] + [0.1] * 6, 0.01)
out = c.step(0.01, DT, [0.0] * 6)
check('超界 ⇒ 整筆停止（回傳 None）', out is None)
check('整體失效原因為輪級', c.fail is not None and '輪級' in c.fail, c.fail)
check('未縮底盤後續跑', c.step(0.02, DT, [0.0] * 6) is None)

print('F 時間重設')
c = mk()
c.receive([0] * 9, 0.0)
c.step(0.0, DT, [0.0] * 6)
c.note_time_reset(0.0)
check('時間非單調 ⇒ 整體失效', c.fail is not None and '單調' in c.fail, c.fail)

print('G 接收端編號不冒稱來源時間')
c = mk()
c.receive([0] * 9, 1.234)
check('欄位名為 recv_seq / recv_sim_t',
      hasattr(c.snap, 'recv_seq') and hasattr(c.snap, 'recv_sim_t'),
      f'recv_seq={c.snap.recv_seq}, recv_sim_t={c.snap.recv_sim_t}')
check('summary 註明不得冒稱端到端延遲',
      '端到端' in c.summary()['recv_seq_note'])

print('H 模式錯誤：接收端拒收，不事後切掉分量')
c = mk(mode='base')
check('base 模式收到手臂分量 ⇒ 拒收',
      not c.receive([0.01, 0, 0] + [0.1] + [0.0] * 5, 0.0), c.last_reject)
check('整體失效（非只把手臂歸零）', c.fail is not None and '模式' in c.fail, c.fail)
c = mk(mode='base')
check('base 模式純底盤 ⇒ 接受', c.receive([0.01, 0, 0] + [0.0] * 6, 0.0))
c = mk(mode='arm')
check('arm 模式收到底盤分量 ⇒ 拒收',
      not c.receive([0.01, 0, 0] + [0.0] * 6, 0.0), c.last_reject)
c = mk(mode='arm')
check('arm 模式純手臂 ⇒ 接受', c.receive([0, 0, 0] + [0.1] * 6, 0.0))
c = mk(mode='sync')
check('sync 模式兩者皆可', c.receive([0.01, 0, 0] + [0.1] * 6, 0.0))

print('I 失效停止：底盤停止、手臂保持設定點、可繼續量測')
c = mk()
c.receive([0.01, 0, 0] + [0.2] * 6, 0.0)
c.step(0.0, DT, [0.1] * 6)
sp_held = tuple(c.setpoint)
c.receive([0] * 8, 0.01)                    # 觸發失效
check('已失效', c.fail is not None, c.fail)
base, sp = c.stop_command()
check('底盤停止', base == (0.0, 0.0, 0.0), str(base))
check('手臂保持在失效前的設定點', sp == sp_held, '')
b2, s2 = c.stop_command()
check('可重複呼叫（迴圈可繼續量測）', (b2, s2) == (base, sp))
c2 = mk()
c2.receive([0] * 8, 0.0)
b3, s3 = c2.stop_command()
check('設定點未建立時不對手臂下命令', s3 is None, f'base={b3}')

print('J 監看失效：讀不到不等於安全')
c = mk()
c.receive([0] * 9, 0.0)
c.step(0.0, DT, [0.0] * 6)
c.note_monitor_failure('cpu_temp', '讀不到（來源 None）', 0.5)
check('監看失效 ⇒ 整體失效', c.fail is not None and '監看失效' in c.fail, c.fail)
check('失效後 step 不再執行命令', c.step(0.51, DT, [0.0] * 6) is None)
check('但停止命令仍可取得', c.stop_command()[0] == (0.0, 0.0, 0.0))

print('K pregrasp 不可由參數繞過')
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'isaac_wholebody_sim.py')).read()
check('無 --wheel-limit-implemented 旗標', '--wheel-limit-implemented' not in src)
check('由程式常數把關', 'WHEEL_LIMIT_IMPLEMENTED = False' in src)
check('pregrasp 檢查存在',
      "a.mode == 'pregrasp' and not WHEEL_LIMIT_IMPLEMENTED" in src)
check('回報為低速介面界限而非輪級限制',
      'low_speed_interface_bound' in src and
      "'wheel_level_limiting_implemented': WHEEL_LIMIT_IMPLEMENTED" in src)

print('L 回授確實發布')
for topic in ('js_pub.publish', 'odom_pub.publish', 'status_pub.publish',
              'clock_pub.publish'):
    check(f'{topic} 有呼叫', topic in src)

# ============ M 關節順序核對（adapter 的純函式，不開 ROS、不開 Isaac）============
print('\nM 關節順序核對')
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'arm_vel_adapter.py')).read()
_ns = {}
exec(_src[_src.index('def verify_order'):_src.index('class Adapter')], _ns)
verify_order = _ns['verify_order']
ARM6 = [f'joint{i}' for i in range(1, 7)]

ok, why = verify_order(ARM6, ARM6)
check('正確順序 ⇒ 通過', ok and why is None)

_sw = ARM6[:]; _sw[1], _sw[2] = _sw[2], _sw[1]
ok, why = verify_order(_sw, ARM6)
check('交換兩個關節 ⇒ 拒絕', not ok and '順序不同' in (why or ''), why)

ok, why = verify_order(None, ARM6)
check('服務缺失／回應無效 ⇒ 拒絕', not ok and '服務缺失' in (why or ''), why)

ok, why = verify_order('joint1', ARM6)
check('型別不對 ⇒ 拒絕', not ok and '型別' in (why or ''), why)

ok, why = verify_order(['a'] * 6, ARM6)
check('關節集合不同 ⇒ 拒絕', not ok and '集合不同' in (why or ''), why)

print('N 消費端節點可設定，預設仍為 Gazebo 控制器')
check('預設常數為 /lite6_vel_controller',
      "CTRL_DEFAULT = '/lite6_vel_controller'" in _src)
check('Adapter 接受 ctrl 參數', 'def __init__(self, ctrl=CTRL_DEFAULT)' in _src)
check('查詢路徑用 self.ctrl', "f'{self.ctrl}/get_parameters'" in _src)
check('新增 --consumer-node 旗標', "'--consumer-node'" in _src)
check('order_ok 為假時仍拒絕轉送',
      'not self.order_ok' in _src and 'refusing to forward commands' in _src)

print('O Isaac 端的 joints 與實際命令映射同源')
_isrc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'isaac_wholebody_sim.py')).read()
check('單一命令順序常數', 'CMD_JOINT_ORDER = [' in _isrc)
check('ARM 指向同一份', 'ARM = CMD_JOINT_ORDER' in _isrc)
check('joints 參數由該順序宣告',
      "self.declare_parameter('joints', list(joint_order))" in _isrc)
check('啟動時檢查重複關節',
      'len(set(CMD_JOINT_ORDER)) != len(CMD_JOINT_ORDER)' in _isrc)
check('啟動時檢查重複 DOF 索引', 'len(set(dof_ids)) != len(dof_ids)' in _isrc)
check('記錄 命令欄位 → 關節 → DOF 索引', "'cmd_field': 3 + k" in _isrc)

print(f'\n{"全部通過" if not fails else "**未通過：" + ", ".join(fails) + "**"}')
sys.exit(1 if fails else 0)
