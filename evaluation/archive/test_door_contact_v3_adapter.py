#!/usr/bin/env python3
"""door_contact_v3_adapter 純函式／替身驗證（Codex r47 §3；不匯入 SimulationApp、不啟動物理）。
  A1 切片與部位展開：三個剛體 count (2,0,1)、start (2,0,0)；點依剛體名順序、部位逐點正確；負法向力 ⇒ 大小為正、fn_signed 保留負號；
     未被任何切片引用的緩衝項（第 1 筆 7.7 N）不計入；遠點 ⇒ ambiguous；
  A2 讀取緩衝拋例外 ⇒ query_failed、sums None；門角讀取拋例外 ⇒ query_failed；門根變換含縮放 ⇒ query_failed；
  A3 緩衝不完整 ⇒ query_failed（不記零）：切片超出、count 負、count 非整數、pair 數與名數不符、切片重疊、點與力長度不符、力緩衝 (M,2)、回傳項數錯；
  A4 全部 count 0 ⇒ ok、全 0、complete True（真的無接觸）；
  A5 某點法向力 NaN ⇒ 該點 invalid、complete False、有效小計保留；
  A6 Summary 累計：狀態數、完整步數、invalid 點數、各類最大力和；
  A7 類 torch 張量（.cpu().numpy()）可讀。
  A8（r2）法向／分離 NaN ⇒ complete False；dt 0／負／NaN／None ⇒ query_failed；get_data 回 None ⇒ query_failed（帶步號、時刻）；failure_record；覆蓋對帳。

    python3 evaluation/test_door_contact_v3_adapter.py
"""
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_contact_v3_adapter as AD                   # noqa: E402
import door_geom_v3 as G3                              # noqa: E402

D = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')))
H = D['handle']
TOL, DT = 0.006, 1 / 120
Q = math.radians(30.0)
NAMES = ['uflite_finger1', 'link6', 'uflite_finger2']
fails = []


def ok(c, m):
    if not c:
        fails.append(m)


def bar_pt(T, ang=math.pi):
    return T[:3, 3] + H['bar']['radius'] * (math.cos(ang) * T[:3, 1] + math.sin(ang) * T[:3, 2])


To, Ti = G3.handle_pose(D, Q, H['height']), G3.handle_pose_inner(D, Q)
far = np.array([5.0, 5.0, 5.0])


def buffers(M=4, fn=None, pts=None, cnt=(2, 0, 1), start=(2, 0, 0)):
    P = np.array([bar_pt(Ti), bar_pt(To), bar_pt(Ti, 0.5 * math.pi), far]) if pts is None else pts
    f = np.array([[3.0], [7.7], [-2.0], [9.9]]) if fn is None else fn
    return (f, P, np.tile([0.0, 0.0, 1.0], (len(P), 1)), np.zeros((len(P), 1)), np.array(cnt, float), np.array(start, float))


def rec(data, q=Q, T=np.eye(4)):
    return AD.step_record(lambda: data, lambda: q, lambda: T, D, TOL, DT, NAMES, step=7, t=0.5, spec_id='door_local_v3')


# A1：finger1 取 [2,4)＝bar_in（−2.0）、遠點（9.9）；finger2 取 [0,1)＝bar_in（3.0）；第 1 筆（bar_out 7.7）未被引用
r = rec(buffers())
ok(r['status'] == 'ok', f"A1 status {r['status']} {r.get('why')}")
ok([p['robot_link'] for p in r['points']] == ['uflite_finger1', 'uflite_finger1', 'uflite_finger2'], f"A1 部位 {[p['robot_link'] for p in r['points']]}")
ok([p['class'] for p in r['points']] == ['bar_in', 'ambiguous', 'bar_in'], f"A1 類別 {[p['class'] for p in r['points']]}")
ok(r['points'][0]['f'] == 2.0 and r['points'][0]['fn_signed'] == -2.0, 'A1 負法向力大小／原值')
ok(abs(r['sums']['bar_in'] - 5.0) < 1e-12 and r['sums']['bar_out'] == 0.0 and abs(r['sums']['ambiguous'] - 9.9) < 1e-12
   and r['reconciled'] and r['complete'], f"A1 力和 {r['sums']}")
ok(r['dt_s'] == DT and '法向' in r['force_semantics'] and r['spec_id'] == 'door_local_v3', 'A1 dt／語意／規格識別')


# A2
def boom():
    raise RuntimeError('tensor not ready')


r = AD.step_record(boom, lambda: Q, lambda: np.eye(4), D, TOL, DT, NAMES)
ok(r['status'] == 'query_failed' and r['sums'] is None and 'tensor not ready' in r['why'], f"A2 讀取失敗 {r['status']}")
r = AD.step_record(lambda: buffers(), boom, lambda: np.eye(4), D, TOL, DT, NAMES)
ok(r['status'] == 'query_failed' and r['sums'] is None, 'A2 門角失敗')
r = rec(buffers(), T=np.diag([1.0, 1.0, 1.001, 1.0]))
ok(r['status'] == 'query_failed' and r['sums'] is None and '變換' in r['why'], f"A2 變換縮放 {r['status']} {r.get('why')}")

# A3
bad = {'overflow': buffers(cnt=(2, 0, 1), start=(2, 0, 4)), 'neg': buffers(cnt=(2, 0, -1)), 'nonint': buffers(cnt=(2, 0, 1.5)),
       'names': buffers(cnt=(2, 0), start=(2, 0)), 'overlap': buffers(cnt=(2, 0, 2), start=(2, 0, 1)),
       'len': buffers(fn=np.array([[1.0], [2.0], [3.0]])), 'fn2col': buffers(fn=np.ones((4, 2)))}
for k, data in bad.items():
    r = rec(data)
    ok(r['status'] == 'query_failed' and r['sums'] is None and r['total_valid'] is None, f"A3 {k} → {r['status']} {r.get('why')}")
r = AD.step_record(lambda: buffers()[:5], lambda: Q, lambda: np.eye(4), D, TOL, DT, NAMES)
ok(r['status'] == 'query_failed', 'A3 回傳項數')

# A4
r = rec(buffers(cnt=(0, 0, 0)))
ok(r['status'] == 'ok' and r['complete'] and all(v == 0.0 for v in r['sums'].values()) and r['points'] == [], 'A4 無接觸')

# A5
r = rec(buffers(fn=np.array([[3.0], [7.7], [-2.0], [np.nan]])))
ok(r['status'] == 'ok' and not r['complete'] and r['n_invalid'] == 1 and abs(r['sums']['bar_in'] - 5.0) < 1e-12, f"A5 {r['counts']}")

# A6
S = AD.Summary()
for d_ in (buffers(), buffers(cnt=(0, 0, 0)), buffers(fn=np.array([[3.0], [7.7], [-2.0], [np.nan]]))):
    S.add(rec(d_))
S.add(AD.step_record(boom, lambda: Q, lambda: np.eye(4), D, TOL, DT, NAMES))
s = S.as_dict()
ok(s['n_steps'] == 4 and s['n_ok'] == 3 and s['n_complete'] == 2 and s['n_query_failed'] == 1 and s['n_invalid_points'] == 1
   and abs(s['max_sum_by_class']['bar_in'] - 5.0) < 1e-12 and not s['complete_all_steps'], f'A6 {s}')


# A7
class FakeT:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


r = rec(tuple(FakeT(x) for x in buffers()))
ok(r['status'] == 'ok' and [p['class'] for p in r['points']] == ['bar_in', 'ambiguous', 'bar_in'], 'A7 類張量')

# A8（r2，Codex r48 §1）
d_ = list(buffers())
d_[2] = d_[2].copy()
d_[2][2] = [np.nan, 0.0, 1.0]                     # finger1 第一點（緩衝第 2 筆）法向 NaN
r = rec(tuple(d_))
ok(r['status'] == 'ok' and r['complete'] is False and r['n_evidence_incomplete'] == 1 and r['points'][0]['evidence_ok'] is False, f"A8 法向 NaN {r.get('complete')}")
d_ = list(buffers())
d_[3] = d_[3].copy()
d_[3][0] = np.nan                                  # finger2 點的分離量 NaN
r = rec(tuple(d_))
ok(r['status'] == 'ok' and r['complete'] is False and r['n_evidence_incomplete'] == 1, 'A8 分離 NaN')
for bad_dt in (0.0, -0.01, float('nan'), None):
    r = AD.step_record(lambda: buffers(), lambda: Q, lambda: np.eye(4), D, TOL, bad_dt, NAMES, step=3, t=0.2)
    ok(r['status'] == 'query_failed' and r['sums'] is None and 'dt' in r['why'] and r['step'] == 3, f'A8 dt {bad_dt}')
r = AD.step_record(lambda: None, lambda: Q, lambda: np.eye(4), D, TOL, DT, NAMES, step=9, t=0.9)
ok(r['status'] == 'query_failed' and r['step'] == 9 and r['t'] == 0.9 and r['sums'] is None, f"A8 get_data None {r['status']}")
fr = AD.failure_record(11, 1.1, RuntimeError('x'))
ok(fr['status'] == 'query_failed' and fr['step'] == 11 and fr['complete'] is False and fr['sums'] is None, 'A8 failure_record')
S = AD.Summary()
for _ in range(3):
    S.add(rec(buffers()))
ok(S.as_dict(expected_steps=3, n_written=3)['coverage_ok'] and not S.as_dict(expected_steps=4, n_written=3)['coverage_ok']
   and not S.as_dict(expected_steps=3, n_written=2)['coverage_ok'] and not S.as_dict()['coverage_ok'], 'A8 覆蓋對帳')

print('全部通過' if not fails else f'失敗 {len(fails)}')
for f in fails:
    print('  ', f)
sys.exit(1 if fails else 0)
