#!/usr/bin/env python3
"""求解器的**執行期目標話題**：檢查與來源解析。

為什麼需要這條路
----------------
啟動參數只能給**固定**目標。抽屜實驗做不到：接觸前的退讓目標要由**實測
把手位姿**算出，而開啟／關閉段的目標會隨實測開度移動
（evaluation/drawer_target.py）。

這一層要釘住三件事：
  1. 不合格的目標**說出理由**而不是靜默丟棄 —— 靜默丟棄會讓「求解器還在用
     舊目標」看起來像「目標沒變」。
  2. 過期的目標**沿用上一筆已接受的**並記錄，不當成新鮮值。
  3. 求解與到達判定**用同一份**目標。兩處各自取會變成「用一個目標求解、
     用另一個目標判到達」。
  4. 預設關閉時行為與既有趟次一位元相同。
"""
from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'ammr_wholebody_mpc'))

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


# 只取純函式與未綁定方法，不建 ROS 節點
import importlib.util                                            # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'wg2node', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'wgmpc_wg2_node.py'))


def _load():
    """只讀取原始碼取出要測的片段 —— 匯入整個節點會拉進 rclpy 與 ROS 型別。"""
    import ast as _ast
    src = open(_spec.origin).read()
    tree = _ast.parse(src)
    want = {'validate_target_matrix'}
    keep = [n for n in tree.body
            if isinstance(n, _ast.FunctionDef) and n.name in want]
    ns = {'math': math, 'np': np}
    mod = _ast.Module(body=keep, type_ignores=[])
    exec(compile(mod, '<wg2node-subset>', 'exec'), ns)
    # _resolve_target 是方法，單獨取出來
    cls = next(n for n in tree.body
               if isinstance(n, _ast.ClassDef)
               and any(isinstance(b, _ast.FunctionDef)
                       and b.name == '_resolve_target' for b in n.body))
    meth = next(b for b in cls.body
                if isinstance(b, _ast.FunctionDef)
                and b.name == '_resolve_target')
    ns2 = {'math': math, 'np': np}
    exec(compile(_ast.Module(body=[meth], type_ignores=[]),
                 '<wg2node-subset>', 'exec'), ns2)
    return ns['validate_target_matrix'], ns2['_resolve_target']


validate_target_matrix, resolve = _load()


def mat(R=None, p=(1.0, 2.0, 3.0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def flat(T):
    return [float(x) for x in np.asarray(T, float).reshape(-1)]


# ---- A 組：檢查 ----------------------------------------------------------
T, why = validate_target_matrix(flat(mat()))
chk('A1 合格的剛體變換通過', why is None and T is not None, str(why))
chk('A2 通過後回傳的矩陣與輸入相同',
    T is not None and np.allclose(T, mat(), atol=0.0, rtol=0.0))

for nm, d, frag in (
        ('長度不是 16', [0.0] * 15, '長度'),
        ('含 NaN', flat(mat())[:-1] + [float('nan')], '非有限'),
        ('含 inf', [float('inf')] + flat(mat())[1:], '非有限'),
):
    T2, w2 = validate_target_matrix(d)
    chk(f'A {nm} ⇒ 拒絕並說出理由',
        T2 is None and w2 is not None and frag in w2, str(w2))

bad = mat(); bad[3] = (0.0, 0.0, 0.0, 2.0)
T3, w3 = validate_target_matrix(flat(bad))
chk('A3 最後一列不是 (0,0,0,1) ⇒ 拒絕',
    T3 is None and '最後一列' in str(w3), str(w3))

bad2 = mat(R=np.diag([1.0, 1.0, 2.0]))     # 非正交
T4, w4 = validate_target_matrix(flat(bad2))
chk('A4 旋轉塊非正交 ⇒ 拒絕', T4 is None and '正交' in str(w4), str(w4))

bad3 = mat(R=np.diag([1.0, 1.0, -1.0]))    # 正交但 det = -1（鏡射）
T5, w5 = validate_target_matrix(flat(bad3))
chk('A5 旋轉塊是鏡射（det = −1）⇒ 拒絕',
    T5 is None and '行列式' in str(w5), str(w5))

# 真的旋轉要通過
th = 0.7
Rz = np.array([[math.cos(th), -math.sin(th), 0.0],
               [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
T6, w6 = validate_target_matrix(flat(mat(R=Rz)))
chk('A6 真正的旋轉通過', w6 is None, str(w6))


# ---- B 組：來源解析 ------------------------------------------------------
def node(topic='', max_age=0.5, T_rx=None, T_rx_t=None):
    o = types.SimpleNamespace()
    o.a = types.SimpleNamespace(target_topic=topic, target_max_age_s=max_age)
    o._T_rx, o._T_rx_t = T_rx, T_rx_t
    o._n_by_target_src = {}
    o._max_target_age_s = 0.0
    return o


T_LAUNCH = mat(p=(0.0, 0.0, 0.0))
T_TOPIC = mat(p=(9.0, 9.0, 9.0))

o = node(topic='')
T7, src, age = resolve(o, T_LAUNCH, 10.0)
chk('B1 關閉時用啟動參數的固定目標',
    src == 'launch_arg' and np.allclose(T7, T_LAUNCH) and age is None, src)
chk('B2 關閉時不記年齡', o._max_target_age_s == 0.0)

o = node(topic='/t')
T8, src, age = resolve(o, T_LAUNCH, 10.0)
chk('B3 開著但還沒收到 ⇒ 退回啟動參數，且來源標示得出來',
    src == 'launch_arg_fallback' and np.allclose(T8, T_LAUNCH), src)

o = node(topic='/t', T_rx=T_TOPIC, T_rx_t=9.9)
T9, src, age = resolve(o, T_LAUNCH, 10.0)
chk('B4 新鮮的話題目標 ⇒ 採用',
    src == 'topic' and np.allclose(T9, T_TOPIC), src)
chk('B5 年齡算得出來', abs(age - 0.1) < 1e-9, str(age))

o = node(topic='/t', max_age=0.5, T_rx=T_TOPIC, T_rx_t=8.0)
T10, src, age = resolve(o, T_LAUNCH, 10.0)
chk('B6 過期 ⇒ **沿用上一筆已接受的**，不退回啟動參數也不當成新鮮',
    src == 'topic_stale_hold' and np.allclose(T10, T_TOPIC), src)
chk('B7 過期的年齡有記錄', abs(o._max_target_age_s - 2.0) < 1e-9,
    str(o._max_target_age_s))

o = node(topic='/t', T_rx=T_TOPIC, T_rx_t=9.9)
for _ in range(3):
    resolve(o, T_LAUNCH, 10.0)
chk('B8 逐輪的來源次數有累計', o._n_by_target_src.get('topic') == 3,
    str(o._n_by_target_src))

# **同一輪只解析一次**：連續兩次呼叫在來源統計上要是兩輪，不是一輪
o = node(topic='/t', T_rx=T_TOPIC, T_rx_t=9.9)
a1 = resolve(o, T_LAUNCH, 10.0)
a2 = resolve(o, T_LAUNCH, 10.0)
chk('B9 兩次呼叫回傳同一份矩陣（求解與到達判定共用的前提）',
    np.allclose(a1[0], a2[0], atol=0.0, rtol=0.0))

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
