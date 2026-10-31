#!/usr/bin/env python3
"""K1（低速介面放寬決策資料）步驟 2：速度框跨層一致性測試（只讀、不改任何預設）。
從實跑跑法 run_mission_roundtrip2.sh 的實際生效值（跑法覆寫 > argparse 預設 > 模組常數）取出各層框值，
檢查層間關係；再模擬「只放大求解框 ×k」與「各層一起放大 ×k」，列出哪些關係斷掉＝放寬時必須一起改的地方。
    python3 evaluation/test_k1_box_consistency.py [--out evaluation/results/k1/box_consistency.json]
"""
import argparse
import inspect
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wgmpc_core import WGMPCConfig            # noqa: E402
from ammr_wholebody_mpc.wgmpc_handover import NavHandoverConfig  # noqa: E402
import door_k3m_applied_src_tb as K3MSRCTB                       # noqa: E402
import door_k3m_pause as PZ                                      # noqa: E402
import door_regrasp_mode1 as RGM                                 # noqa: E402
import door_regrasp_mode3 as RGM3                                # noqa: E402
import door_regrasp_policy1 as RGP1                              # noqa: E402
from drawer_target import DrawerTargetConfig                     # noqa: E402

RUNNER = os.path.join(HERE, 'run_mission_roundtrip2.sh')
REF = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60o.json')


def cli_default(fname, opt):
    txt = open(os.path.join(HERE, fname), encoding='utf-8').read()
    m = re.search(r"add_argument\('" + re.escape(opt) + r"'[^)]*?default\s*=\s*([0-9.eE+-]+)", txt, re.S)
    if not m:
        raise SystemExit(f'{fname} 找不到 {opt} 預設')
    return float(m.group(1))


def effective(fname, opt):
    """跑法對該節點檔有覆寫就用覆寫值（所有出現處須一致），否則用 argparse 預設。"""
    vals = set()
    for ln in open(RUNNER, encoding='utf-8'):
        if fname in ln and not ln.lstrip().startswith('#'):
            for m in re.finditer(re.escape(opt) + r'[= ]"?([0-9.]+)', ln):
                vals.add(float(m.group(1)))
    if len(vals) > 1:
        raise SystemExit(f'跑法對 {fname} {opt} 覆寫不一致：{vals}')
    return (vals.pop(), 'runner') if vals else (cli_default(fname, opt), 'default')


def sig_default(cls, name):
    return float(inspect.signature(cls.__init__).parameters[name].default)


def collect():
    c = WGMPCConfig()
    ho = NavHandoverConfig()
    P = json.load(open(REF))
    v, src = {}, {}

    def put(k, val, where):
        v[k] = float(val)
        src[k] = where
    put('solver_lin', c.v_base_lin, 'wgmpc_core.WGMPCConfig.v_base_lin（求解節點未覆寫）')
    put('solver_ang', c.v_base_ang, 'wgmpc_core.WGMPCConfig.v_base_ang')
    put('solver_arm', c.v_arm, 'wgmpc_core.WGMPCConfig.v_arm')
    put('wheel_r', c.wheel_radius, 'wgmpc_core.wheel_radius')
    put('wheel_L', c.wheel_base_L, 'wgmpc_core.wheel_base_L')
    put('wheel_w_max', c.wheel_w_max, 'wgmpc_core.wheel_w_max')
    for k, opt in (('exec_lin_max', '--base-lin-max'), ('exec_ang_max', '--base-ang-max'), ('exec_arm_max', '--arm-rate-max'),
                   ('exec_switch_lin', '--v-box-lin'), ('exec_switch_ang', '--v-box-ang')):
        val, how = effective('isaac_door_k2_sim_ev9r8z.py', opt)
        put(k, val, f'ev9r8z {opt}（{how}）')
    put('intake_lin', K3MSRCTB.V_LIN, 'door_k3m_applied_src_tb.V_LIN（模組常數）')
    put('intake_ang', K3MSRCTB.V_ANG, 'door_k3m_applied_src_tb.V_ANG')
    put('pause_lin', PZ.BaseLimits().v_lin, 'door_k3m_pause.BaseLimits.v_lin')
    put('pause_ang', PZ.BaseLimits().v_ang, 'door_k3m_pause.BaseLimits.v_ang')
    put('handover_lin', ho.v_box_lin_mps, 'wgmpc_handover.NavHandoverConfig（drawer_mission_node 用預設）')
    put('handover_ang', ho.v_box_ang_rps, 'wgmpc_handover.NavHandoverConfig')
    for nm, cls in (('rg1', RGM.RegraspAuthority), ('rg3', RGM3.RegraspAuthority)):
        put(f'{nm}_lin', sig_default(cls, 'v_lin'), f'{cls.__module__}.RegraspAuthority v_lin（建構未覆寫）')
        put(f'{nm}_ang', sig_default(cls, 'v_ang'), f'{cls.__module__}.RegraspAuthority v_ang')
        put(f'{nm}_world', sig_default(cls, 'v_world'), f'{cls.__module__}.RegraspAuthority v_world')
    put('drawer_rate', DrawerTargetConfig().rate_max_mps, 'drawer_target.DrawerTargetConfig.rate_max_mps')
    put('guide_v_cap', RGP1.REG['v_cap'], 'door_regrasp_policy1.REG v_cap（世界每軸截斷）')
    put('guide_w_cap', RGP1.REG['w_cap'], 'door_regrasp_policy1.REG w_cap')
    put('r1_w_cap', P['m4r']['r1_w_cap'], 'pb60o m4r.r1_w_cap')
    put('r3_w_cap', P['m4r']['r3_w_cap'], 'pb60o m4r.r3_w_cap')
    put('r3_ref_v', P['r3_retime2']['v_max'], 'pb60o r3_retime2.v_max（重時間化參考）')
    for k, f, opt in (('nav1_glide_v', 'door_nav1_mission_node_r11.py', '--v-max'), ('nav1_ready_v', 'door_nav1_mission_node_r11.py', '--ready-v-max'),
                      ('nav1_ready_w', 'door_nav1_mission_node_r11.py', '--ready-w'), ('nav1_motm_v', 'door_nav1_mission_node_r11.py', '--motm-v-cap'),
                      ('wb_motm_v', 'door_k2_wholebody_node.py', '--motm-v-cap'), ('wb_motm_w', 'door_k2_wholebody_node.py', '--motm-w-cap'),
                      ('task_motm_v', 'door_coopen_core_task_node10p.py', '--motm-v-cap'), ('ret_bridge_v', 'door_return_task_node.py', '--bridge-v-max'),
                      ('ret_bridge_w', 'door_return_task_node.py', '--bridge-w-max'), ('glide_v_roll', 'drawer_glide_node.py', '--v-roll'),
                      ('dm_slow_v', 'drawer_mission_node.py', '--slow-v'), ('drawer_motm_v', 'drawer_task_node4.py', '--motm-v-cap')):
        val, how = effective(f, opt)
        put(k, val, f'{f} {opt}（{how}）')
    return v, src


RUNNER3 = os.path.join(HERE, 'run_mission_roundtrip3.sh')
REF3 = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60p.json')


def collect3():
    """--runner3：讀 run_mission_roundtrip3.sh 的實際生效值（框 ×2 介面＋R3／S8 ×1.5）。"""
    global RUNNER
    RUNNER = RUNNER3
    v, src = collect()
    c = WGMPCConfig()
    P = json.load(open(REF3))
    sc = {f: effective(f, '--base-box-scale') for f in ('dl3_wgmpc_wg2_adm24_node.py', 'wgmpc_wg2_node4.py')}
    if len({x[0] for x in sc.values()}) != 1:
        raise SystemExit(f'兩個求解節點框倍率不一致：{sc}')
    k = next(iter(sc.values()))[0]
    c.base_box_scale = k
    vb = c.vbox()

    def put(key, val, where):
        v[key] = float(val)
        src[key] = where
    put('solver_lin', vb[0], f'WGMPCConfig.vbox()（base_box_scale {k}，ADM24／node4 --base-box-scale）')
    put('solver_ang', vb[2], 'WGMPCConfig.vbox() 角')
    put('solver_arm', vb[3], 'WGMPCConfig.vbox() 手臂（不放大）')
    for key, opt in (('exec_lin_max', '--base-lin-max'), ('exec_ang_max', '--base-ang-max'), ('exec_arm_max', '--arm-rate-max'),
                     ('exec_switch_lin', '--v-box-lin'), ('exec_switch_ang', '--v-box-ang')):
        val, how = effective('isaac_door_k2_sim_ev9r9a.py', opt)
        put(key, val, f'ev9r9a {opt}（{how}）')
    for key, opt in (('intake_lin', '--wb-box-lin'), ('intake_ang', '--wb-box-ang'), ('rg3_lin', '--wb-box-lin'), ('rg3_ang', '--wb-box-ang'), ('rg3_world', '--wb-v-world')):
        val, how = effective('isaac_door_k2_sim_ev9r9a.py', opt)
        put(key, val, f'ev9r9a {opt}（{how}）→ 入口投影／換抓權威 mode3')
    for nm in ('rg1_lin', 'rg1_ang', 'rg1_world'):                       # mode1 權威未在執行端建構
        v.pop(nm, None)
        src.pop(nm, None)
    put('pause_lin', vb[0], 'ADM24 PAUSE BaseLimits v_lin＝cfg.vbox()')
    put('pause_ang', vb[2], 'ADM24 PAUSE BaseLimits v_ang＝cfg.vbox()')
    for key, opt in (('handover_lin', '--v-box-lin'), ('handover_ang', '--v-box-ang')):
        val, how = effective('drawer_mission_node2.py', opt)
        put(key, val, f'drawer_mission_node2 {opt}（{how}）')
    put('r1_w_cap', P['m4r']['r1_w_cap'], 'pb60p m4r.r1_w_cap')
    put('r3_w_cap', P['m4r']['r3_w_cap'], 'pb60p m4r.r3_w_cap')
    put('r3_v_cap_body', P['m4r']['r3_v_cap_body'], 'pb60p m4r.r3_v_cap_body（R3 本體逐軸）')
    put('r3_ref_v', P['r3_retime2']['v_max'], 'pb60p r3_retime2.v_max')
    pk = P['k1_splice']['offline']['body_vel_peak']
    put('r3_ref_body_peak_lin', max(pk[0], pk[1]), 'pb60p 段 41 本體差分速度峰')
    put('r3_ref_body_peak_ang', pk[2], 'pb60p 段 41 本體差分角速度峰')
    R4 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_return_ref_v4.json')))
    pk8 = R4['k1_splice']['offline']['body_vel_peak']
    put('s8_ref_body_peak_lin', max(pk8[0], pk8[1]), 'v4 S8 前綴本體速度峰')
    put('s8_ref_body_peak_ang', pk8[2], 'v4 S8 前綴本體角速度峰')
    put('s8_end_row', R4['s8_overlap']['end_row'], 'v4 s8_overlap.end_row')
    m = re.search(r'--end-row (\d+) ', open(RUNNER3, encoding='utf-8').read())
    put('runner_end_row', int(m.group(1)), 'roundtrip3 A --end-row')
    return v, src


def relations3(v):
    R = [r for r in relations(v) if not r[0].startswith('rg1_') and 'mode1' not in r[0]]
    R += [
        ('R3 引導本體上限 ≤ 框', 'guide', lambda v: v['r3_v_cap_body'] <= v['solver_lin'] + 1e-12 and v['r3_w_cap'] <= v['solver_ang'] + 1e-12),
        ('R3 引導本體上限 ≥ R3 參考本體速度峰（留回授空間）', 'guide', lambda v: v['r3_v_cap_body'] >= v['r3_ref_body_peak_lin'] and v['r3_w_cap'] >= v['r3_ref_body_peak_ang']),
        ('R3 參考本體速度峰 ≤ 框', 'ref', lambda v: v['r3_ref_body_peak_lin'] <= v['solver_lin'] and v['r3_ref_body_peak_ang'] <= v['solver_ang']),
        ('S8 參考本體速度峰 ≤ 框', 'ref', lambda v: v['s8_ref_body_peak_lin'] <= v['solver_lin'] and v['s8_ref_body_peak_ang'] <= v['solver_ang']),
        ('跑法 A --end-row＝v4 s8 end_row', 'ref', lambda v: v['runner_end_row'] == v['s8_end_row']),
        ('角速度界只對角速度框（不乘 √2）且不低於框', 'latch', lambda v: v['solver_ang'] <= v['exec_ang_max'] <= v['solver_ang'] * 1.01),
    ]
    return R


# 關係：(名稱, 層別, 函式, 說明)。b＝求解框（lin, ang, arm）
def relations(v):
    L = v['wheel_L']
    W = v['wheel_r'] * v['wheel_w_max']
    R = []

    def eq(a, b):
        return abs(a - b) <= 1e-9

    for k in ('exec_switch', 'intake', 'pause', 'handover', 'rg1', 'rg3'):
        R.append((f'{k}_lin＝求解框', 'same_box', lambda v, k=k: eq(v[f'{k}_lin'], v['solver_lin']) or abs(v[f'{k}_lin'] - v['solver_lin']) < 1e-6))
        R.append((f'{k}_ang＝求解框', 'same_box', lambda v, k=k: abs(v[f'{k}_ang'] - v['solver_ang']) <= 1e-4))
    R += [
        ('執行端合速度界 ≥ √2·框（框角落不閂鎖）', 'latch', lambda v: v['exec_lin_max'] + 1e-9 >= math.sqrt(2) * v['solver_lin']),
        ('執行端角速度界 ≥ 框', 'latch', lambda v: v['exec_ang_max'] + 1e-9 >= v['solver_ang']),
        ('執行端手臂界 ≥ 框', 'latch', lambda v: v['exec_arm_max'] + 1e-9 >= v['solver_arm']),
        ('換抓權威 v_world ≥ √2·框（mode1）', 'latch', lambda v: v['rg1_world'] + 1e-9 >= math.sqrt(2) * v['solver_lin']),
        ('換抓權威 v_world ≥ √2·框（mode3）', 'latch', lambda v: v['rg3_world'] + 1e-9 >= math.sqrt(2) * v['solver_lin']),
        ('輪速：框單軸＋L·框角 ≤ r·ω_max', 'wheel', lambda v: v['solver_lin'] + L * v['solver_ang'] <= W + 1e-12),
        ('nav1 READY 線速上界 ≤ 框（合速度≤逐軸框）', 'handover', lambda v: v['nav1_ready_v'] <= v['exec_switch_lin'] + 1e-12),
        ('nav1 READY 角速上界 ≤ 框', 'handover', lambda v: v['nav1_ready_w'] <= v['exec_switch_ang'] + 1e-12),
        ('nav1 GLIDE v_max ≤ READY 上界（進得了窗）', 'handover', lambda v: v['nav1_glide_v'] <= v['nav1_ready_v'] + 1e-12),
        ('返程 GLIDE v_roll ≤ 框', 'handover', lambda v: v['glide_v_roll'] <= v['solver_lin'] + 1e-12),
        ('抽屜交棒 slow_v ≤ 框', 'handover', lambda v: v['dm_slow_v'] <= v['solver_lin'] + 1e-12),
        ('全身端進場 v_cap ≤ 框', 'guide', lambda v: max(v['wb_motm_v'], v['task_motm_v'], v['nav1_motm_v'], v['drawer_motm_v']) <= v['solver_lin'] + 1e-12),
        ('全身端進場 w_cap ≤ 框', 'guide', lambda v: v['wb_motm_w'] <= v['solver_ang'] + 1e-12),
        ('換抓引導 v_cap（世界逐軸）≤ 框', 'guide', lambda v: v['guide_v_cap'] <= v['solver_lin'] + 1e-12),
        ('換抓引導 v_cap 轉車體最壞 √2·v_cap ≤ 框', 'guide_body', lambda v: math.sqrt(2) * v['guide_v_cap'] <= v['solver_lin'] + 1e-12),
        ('R1／R3 局部 w_cap ≤ 框', 'guide', lambda v: max(v['r1_w_cap'], v['r3_w_cap'], v['guide_w_cap']) <= v['solver_ang'] + 1e-12),
        ('返程橋接 v／w ≤ 框', 'guide', lambda v: v['ret_bridge_v'] <= v['solver_lin'] + 1e-12 and v['ret_bridge_w'] <= v['solver_ang'] + 1e-12),
        ('R3 重時間化參考 v_max ≤ 框', 'ref', lambda v: v['r3_ref_v'] <= v['solver_lin'] + 1e-12),
        ('抽屜目標移動率 ≤ 框', 'ref', lambda v: v['drawer_rate'] <= v['solver_lin'] + 1e-9),
    ]
    return R


# 放寬時「受益於放大」的低速上限（不放大就沒有收益，但也不會違規）
SPEED_KNOBS = ('nav1_glide_v', 'nav1_ready_v', 'nav1_motm_v', 'wb_motm_v', 'task_motm_v', 'drawer_motm_v', 'guide_v_cap', 'r3_ref_v',
               'ret_bridge_v', 'glide_v_roll', 'dm_slow_v', 'drawer_rate')
BOX_KEYS_LIN = ('solver_lin', 'exec_switch_lin', 'intake_lin', 'pause_lin', 'handover_lin', 'rg1_lin', 'rg3_lin')
BOX_KEYS_ANG = ('solver_ang', 'exec_switch_ang', 'intake_ang', 'pause_ang', 'handover_ang', 'rg1_ang', 'rg3_ang')
BOUND_KEYS = ('exec_lin_max', 'exec_ang_max', 'rg1_world', 'rg3_world')      # 上界類（隨框放大）


def evaluate(v):
    return [(n, layer, bool(f(v))) for n, layer, f in relations(v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'k1', 'box_consistency.json'))
    ap.add_argument('--runner3', action='store_true', help='核 run_mission_roundtrip3.sh 實際生效值（只核關係、不做倍率情境）')
    a = ap.parse_args()
    if a.runner3:
        v3, src3 = collect3()
        res = [(n, layer, bool(f(v3))) for n, layer, f in relations3(v3)]
        for k in v3:
            print(f'  {k:<22} {v3[k]:<10.6g} {src3[k]}')
        for n, layer, ok in res:
            print(f'  {"OK  " if ok else "FAIL"} [{layer}] {n}')
        nf = sum(not ok for _, _, ok in res)
        print(f'roundtrip3：{len(res)} 條，失敗 {nf}')
        out3 = os.path.join(HERE, 'results', 'k1', 'box_consistency_rt3.json')
        json.dump({'values': v3, 'source': src3, 'relations': res}, open(out3, 'w'), ensure_ascii=False, indent=1)
        return 0 if nf == 0 else 1
    v0, src = collect()
    out = {'values': v0, 'source': src, 'cases': {}}
    print('現行生效值：')
    for k in v0:
        print(f'  {k:<16} {v0[k]:<10.6g} {src[k]}')
    base = evaluate(v0)
    out['cases']['current'] = base
    print('\n現行配置關係：')
    for n, layer, ok in base:
        print(f'  {"OK  " if ok else "FAIL"} [{layer}] {n}')
    for kf in (1.5, 2.0):
        # A：只放大求解框（lin、ang）
        vA = dict(v0, solver_lin=v0['solver_lin'] * kf, solver_ang=v0['solver_ang'] * kf)
        rA = evaluate(vA)
        brokeA = [(n, layer) for (n, layer, ok), (_, _, ok0) in zip(rA, base) if ok0 and not ok]
        # B：框值與上界類一起放大（低速上限不動）
        vB = dict(v0)
        for k in BOX_KEYS_LIN + BOX_KEYS_ANG + BOUND_KEYS:
            vB[k] = v0[k] * kf
        rB = evaluate(vB)
        brokeB = [(n, layer) for (n, layer, ok), (_, _, ok0) in zip(rB, base) if ok0 and not ok]
        # C：再把低速上限一起放大（取得收益的情形）
        vC = dict(vB)
        for k in SPEED_KNOBS:
            vC[k] = v0[k] * kf
        rC = evaluate(vC)
        brokeC = [(n, layer) for (n, layer, ok), (_, _, ok0) in zip(rC, base) if ok0 and not ok]
        out['cases'][f'x{kf}'] = {'solver_only_broken': brokeA, 'box_and_bounds_broken': brokeB, 'all_scaled_broken': brokeC}
        print(f'\n×{kf}：只放大求解框 ⇒ 斷 {len(brokeA)} 條：')
        for n, layer in brokeA:
            print(f'    [{layer}] {n}')
        print(f'  框＋上界一起放大（低速上限不動）⇒ 斷 {len(brokeB)} 條' + ''.join(f'\n    [{l}] {n}' for n, l in brokeB))
        print(f'  再放大低速上限 ⇒ 斷 {len(brokeC)} 條' + ''.join(f'\n    [{l}] {n}' for n, l in brokeC))
    W = v0['wheel_r'] * v0['wheel_w_max']
    kmax = W / (v0['solver_lin'] + v0['wheel_L'] * v0['solver_ang'])
    out['wheel'] = {'r_w_max': W, 'corner_now': v0['solver_lin'] + v0['wheel_L'] * v0['solver_ang'], 'k_max_uniform': kmax}
    print(f'\n輪速：r·ω_max {W:.4f} m/s；框單軸＋L·框角 現 {out["wheel"]["corner_now"]:.4f}；同比例放大至 ×{kmax:.2f} 才觸輪級限制')
    n_lin = len(BOX_KEYS_LIN) + len(BOUND_KEYS)
    out['n_places'] = {'box_and_bounds': n_lin + len(BOX_KEYS_ANG), 'speed_knobs': len(SPEED_KNOBS)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(a.out, WS))
    fails = [n for n, _, ok in base if not ok]
    return 0 if not fails or fails == ['換抓引導 v_cap 轉車體最壞 √2·v_cap ≤ 框'] else 1


if __name__ == '__main__':
    sys.exit(main())
