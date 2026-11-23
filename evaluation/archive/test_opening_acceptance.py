"""開度判準：正式資格走 v1，舊的案例容差判定不得被當成通過（離線）。

被修的缺陷：runner 只把 stroke 傳給求解端，執行端沿用案例的
`target_opening_m = 0.200`；而執行端自己的到位判定用案例容差 ±10 mm，
遠寬於 v1 的 P1（±0.5 mm）。**不修改 v1 門檻。**
"""
from __future__ import annotations
import os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
V1 = os.path.join(HERE, 'results/specs/wb_coman_drawer20_criteria_v1.yaml')
CASES = os.path.join(WS, 'src/my_omnibot_description/config/manipulation_cases.yaml')
RUNNER = os.path.join(HERE, 'run_coman_drawer20.sh')
EP = os.path.join(HERE, 'isaac_coman_drawer_sim.py')
from coman_handover_state import HandoverMachine, load_spec                # noqa


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:58s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    v1 = yaml.safe_load(open(V1, encoding='utf-8'))
    case = yaml.safe_load(open(CASES, encoding='utf-8'))['cases']['drawer_open_a_fixed']
    runner = open(RUNNER, encoding='utf-8').read()
    ep = open(EP, encoding='utf-8').read()

    # ---- v1 的門檻：只讀，不改 ----
    p1 = float(v1['P_pull']['P1_final_opening_err_m_max'])
    hold = float(v1['P_pull']['P1_hold_s'])
    target = float(v1['profile']['target_stroke_m'])
    print(f'  v1：target {target*1000:.1f} mm、P1 ±{p1*1000:.1f} mm、保持 {hold:.1f} s')
    print(f'  案例：target {case["drawer"]["target_opening_m"]*1000:.0f} mm、'
          f'容差 ±{case["tolerance"]["opening_m"]*1000:.0f} mm、'
          f'保持 {case["tolerance"]["opening_hold_s"]:.1f} s')
    check('v1 的 P1 未被改動（0.5 mm）', abs(p1 - 0.0005) < 1e-12)
    check('v1 的目標行程未被改動（20 mm）', abs(target - 0.020) < 1e-12)

    # ---- 正式資格確實由 v1 狀態機判定 ----
    spec = load_spec(V1)
    m = HandoverMachine(spec)
    check('狀態機的目標取自 v1 profile.target_stroke_m',
          abs(m.target - target) < 1e-12, f'  {m.target*1000:.1f} mm')
    check('狀態機的到位容差取自 v1 P1（不是案例容差）',
          abs(m.p1 - p1) < 1e-12 and abs(m.p1 - case['tolerance']['opening_m']) > 1e-9,
          f'  ±{m.p1*1000:.2f} mm（案例容差是 ±'
          f'{case["tolerance"]["opening_m"]*1000:.0f} mm）')
    # 落在案例容差內但不在 v1 容差內的開度：正式判定**必須不通過**
    for o_mm, want in ((20.0, True), (20.4, True), (20.6, False), (25.0, False),
                       (15.0, False)):
        ok = abs(o_mm / 1000.0 - m.target) <= m.p1
        check(f'開度 {o_mm:.1f} mm：v1 到位判定 {"通過" if want else "不通過"}',
              ok == want)
    check('25.0 mm 在**案例容差**內卻不在 v1 容差內（兩者確實不同）',
          abs(0.025 - 0.020) <= case['tolerance']['opening_m']
          and abs(0.025 - 0.020) > p1)

    # ---- runner 已把目標傳給執行端 ----
    check('runner 傳 --pull-target-m 給執行端', '--pull-target-m "$STROKE"' in runner)
    stroke = re.search(r'STROKE="\$\{STROKE:-([0-9.]+)\}"', runner).group(1)
    check('runner 的 STROKE 與 v1 目標行程相同',
          abs(float(stroke) - target) < 1e-12, f'  {float(stroke)*1000:.1f} mm')

    # ---- 舊判定已改名並標明不是正式資格 ----
    check('舊到位時間欄名已標為 legacy_case_tol_',
          "'legacy_case_tol_arrived_sim_t'" in ep)
    check('輸出帶有正式資格來源欄位', "'formal_acceptance_source'" in ep)
    check('輸出帶有 legacy 與正式的差異說明',
          "'legacy_vs_formal_note'" in ep)
    check('舊到位的列印已標明非 v1 正式資格', '非 v1 正式資格' in ep)
    check('舊欄名 arrived_sim_t 已不在協同執行端出現',
          "'arrived_sim_t':" not in ep)

    print('開度判準測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
