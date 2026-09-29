"""退出方向：目標、參考起點與有號位移判定必須用**同一套定義**（離線）。

用既有趟次的實測位姿核對方向，**不以絕對值掩蓋方向錯誤**。
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics    # noqa
from coman_pull_policy import PullTaskPolicy, TaskState                    # noqa

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
RUN = 'drawer_220102_offset20'
EZ = np.array([0.0, 0.0, 1.0])


def retreat_axis_world(T):
    """**與求解節點同一個定義**（向外＝該夾爪座標系的 −z）。"""
    return -(T[:3, :3] @ EZ)


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:52s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    d = json.load(open(os.path.join(HERE, 'runs', RUN, 'sim',
                                    'drawer_run.json'), encoding='utf-8'))
    ix = {k: j for j, k in enumerate(d['log_cols'])}
    K = WholeBodyKinematics.from_urdf_file(URDF)
    park = d['park']

    def pose(r):
        q = np.zeros(len(K.dof_names))
        q[0], q[1], q[2] = park
        q[3:9] = [float(r[ix[f'joint{i}']]) for i in range(1, 7)]
        return K.fk(q, 'uflite_gripper_link')

    rows = [r for r in d['log'] if str(r[ix['phase']]) == 'retreat']
    T0 = pose(rows[0])
    p0, axis = T0[:3, 3].copy(), retreat_axis_world(T0)

    # 橫桿在哪一側：向外的軸必須**遠離**橫桿
    spec = yaml.safe_load(open(os.path.join(
        WS, 'src/my_omnibot_description/config/drawer_unit.yaml'),
        encoding='utf-8'))
    bar = spec['drawer']['handle']['bar']['center']
    bw = np.array([d['pose'][0] + bar[0],
                   d['drawer_y0'] - 0.019987 + bar[1], bar[2]])
    proj_bar = float((bw - p0) @ axis)
    check('退出軸方向**遠離**橫桿（沿該軸看，橫桿在負側）', proj_bar < 0.0,
          f'  {proj_bar*1000:+.2f} mm')

    # 實測位移沿該軸為**正**
    r_end = float((pose(rows[-1])[:3, 3] - p0) @ axis)
    check('實測退出位移沿同一軸為正', r_end > 0.0, f'  {r_end*1000:+.2f} mm')
    mono = [float((pose(r)[:3, 3] - p0) @ axis) for r in rows]
    check('有號位移單調不減（沒有方向翻轉）',
          all(b >= a - 1e-9 for a, b in zip(mono, mono[1:])))

    # 目標與判定同軸：沿目標位移後的有號量等於 retreat_m
    retreat_m = 0.040
    T_t = T0.copy()
    T_t[:3, 3] = T_t[:3, 3] + retreat_m * retreat_axis_world(T_t)
    check('目標位移在判定軸上恰為 +retreat_m',
          abs(float((T_t[:3, 3] - p0) @ axis) - retreat_m) < 1e-12,
          f'  {float((T_t[:3,3]-p0)@axis)*1000:+.2f} mm')
    # 舊寫法（目標 −z、判定 +z）會是負的 ⇒ 永遠達不到門檻
    old_axis = T0[:3, :3] @ EZ
    T_old = T0.copy()
    T_old[:3, 3] = T_old[:3, 3] + T_old[:3, :3] @ np.array([0, 0, -retreat_m])
    old_signed = float((T_old[:3, 3] - p0) @ old_axis)
    check('**漏洞重現**：舊寫法的有號量為負 ⇒ 判定永不達標',
          old_signed < 0.0, f'  {old_signed*1000:+.2f} mm')

    # 政策端：用同一定義才會判 DONE
    st = lambda rs: TaskState(sim_t=5.0, state_age_s=0.01, attached=False,
                              handover_pass=True, hold_tracking_pass=True,
                              decouple_confirmed=True, pos_err_m=0.001,
                              rot_err_rad=0.001, cmd_max_abs=0.01,
                              retreat_signed_m=rs)

    def run_to_retreat(rs):
        P = PullTaskPolicy(pull_duration_s=0.01, retreat_clear_m=0.0233)
        P.phase = 'RETREAT'
        return P.step(st(rs))

    check('政策端：新定義下 23.3 mm 達標即完成',
          run_to_retreat(0.0233)['done'])
    check('政策端：舊定義的負值不會完成', not run_to_retreat(old_signed)['done'])
    check('政策端：不足門檻不完成', not run_to_retreat(0.020)['done'])

    print('退出方向離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
