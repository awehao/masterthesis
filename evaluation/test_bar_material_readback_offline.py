#!/usr/bin/env python3
"""EV9r7 雙橫桿摩擦讀回段的離線重現（記憶體 USD 舞台，不啟動 Kit／物理）：
  M1 以 door_asset_v3 建門 → 以執行端同一規則（collision_prims 子字串 'handle_bar'）對兩支橫桿 bind_friction(1.0, 0.8) →
     以執行端同一段邏輯讀回 ⇒ 兩支橫桿綁定位置為自身、μ＝登錄值 ⇒ ok；子字串規則確實選到兩支（handle_bar、handle_bar_inner）；
  M2 夾持參數 μs 0.9（≠ 登錄 1.0）⇒ 不合格；
  M3 只綁外側橫桿（內側沿用資產的 door_handle 材質，μ 同值但綁定位置仍為自身）⇒ ok；把內側橫桿的 physics 綁定移除 ⇒ 不合格（bound_at 不是自身）。

    bash evaluation/run_usd_offline.sh evaluation/test_bar_material_readback_offline.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_asset_v3 as A3                             # noqa: E402
from isaac_common import bind_friction, collision_prims   # noqa: E402

A3.register_physx_plugins()
from pxr import Usd, UsdShade                         # noqa: E402

D = A3.load(os.path.join(os.path.dirname(HERE), 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml'))
ROOT = '/World/door'
fails = []


def ok(c, m):
    if not c:
        fails.append(m)


def readback_ok(stage, mu_s, mu_d):
    """與 isaac_door_k2_sim_ev9r7 的讀回段同一邏輯。"""
    _hm = D['physics']['handle_material']
    _bar_mat = {b: A3._material_of(stage, f'{ROOT}/leaf/{b}') for b in ('handle_bar', 'handle_bar_inner')}
    return (abs(mu_s - _hm['static_friction']) < 1e-9 and abs(mu_d - _hm['dynamic_friction']) < 1e-9
            and all(m.get('bound_at') == f'{ROOT}/leaf/{b}'
                    and abs(m.get('static_friction', float('nan')) - _hm['static_friction']) < 1e-6
                    and abs(m.get('dynamic_friction', float('nan')) - _hm['dynamic_friction']) < 1e-6 for b, m in _bar_mat.items())), _bar_mat


def build():
    st = Usd.Stage.CreateInMemory()
    A3.build_usd(st, D, ROOT)
    return st


st = build()
grip = [pr for pr in collision_prims(st, ROOT) if 'handle_bar' in str(pr.GetPath()).lower()]
ok(sorted(str(p.GetPath()).split('/')[-1] for p in grip) == ['handle_bar', 'handle_bar_inner'], f'M1 子字串選取 {[str(p.GetPath()) for p in grip]}')
bind_friction(st, grip, 1.0, 0.8)
r, mats = readback_ok(st, 1.0, 0.8)
ok(r and all(m['material'] == '/World/PhysicsMaterials/custom' for m in mats.values()), f'M1 {mats}')
r, _ = readback_ok(st, 0.9, 0.8)
ok(not r, 'M2 夾持參數不符未拒')
st = build()
bind_friction(st, [st.GetPrimAtPath(f'{ROOT}/leaf/handle_bar')], 1.0, 0.8)
r, mats = readback_ok(st, 1.0, 0.8)
ok(r and mats['handle_bar_inner']['material'] == '/World/PhysicsMaterials/door_handle', f'M3 只綁外側 {mats}')
UsdShade.MaterialBindingAPI(st.GetPrimAtPath(f'{ROOT}/leaf/handle_bar_inner')).UnbindDirectBinding('physics')
r, mats = readback_ok(st, 1.0, 0.8)
ok(not r, f'M3 內側未綁仍通過 {mats}')
print('全部通過' if not fails else f'失敗 {len(fails)}')
for f in fails:
    print('  ', f)
sys.exit(1 if fails else 0)
