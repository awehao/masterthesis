"""首測入口核對（**靜態，不開 Isaac**）。

確認「實際啟動的整條鏈，確實使用已核對的配置」：

  A 建置後載入的程式與 src 相同（install 與 src 逐檔比對 sha）
  B runner 傳的每個 -p 參數，都在對應節點**宣告過**
  C 話題接線成對（發布端 ↔ 訂閱端），新增主題都有對應讀取端
  D 診斷欄位一律**按名稱**索引；沒有任何讀取端用硬編碼位置讀新欄位
  E 規格鏈一致：v1 未被改寫、勘誤／R1／R1.1／S1 的狀態與間距逐項相符
  F **模式條件**：runner 選的模式與執行端守衛的期望值相容
  G **任務數值**：目標行程、到位容差與保持時間，兩端與 v1 一致

**A–E 不構成完整驗證** —— 首趟啟動失敗證明了這點：當時 A–E 全過，
但 (F) `--free-base` 與寫死「恰好 1 個固定關節」的守衛相斥、
(G) 執行端仍用案例的 200 mm 目標。F、G 就是為這兩類漏洞補上的。
"""
from __future__ import annotations
import hashlib, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
PKG = os.path.join(WS, 'src/ammr_wholebody_mpc/ammr_wholebody_mpc')
INST = os.path.join(WS, 'install/ammr_wholebody_mpc/lib/python3.12/'
                        'site-packages/ammr_wholebody_mpc')
SPECS = os.path.join(HERE, 'results/specs')
RUNNER = os.path.join(HERE, 'run_coman_drawer20.sh')
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]

# runner 啟動的節點 → 其原始碼
NODE_SRC = {'arm_link_distance': os.path.join(PKG, 'arm_link_distance.py'),
            'wholebody_safety': os.path.join(PKG, 'wholebody_safety_node.py')}


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    runner = open(RUNNER, encoding='utf-8').read()

    # ---------- A 建置後載入的程式 ----------
    print('A 建置後載入的程式與 src 是否相同')
    for f in sorted(os.listdir(PKG)):
        if not f.endswith('.py'):
            continue
        i = os.path.join(INST, f)
        if not os.path.exists(i):
            check(f'{f} 已安裝', False, '  **install 缺檔**')
            continue
        check(f'{f}', sha(os.path.join(PKG, f)) == sha(i),
              '' if sha(os.path.join(PKG, f)) == sha(i) else '  **sha 不同，需重建**')

    # ---------- B runner 的 -p 參數都宣告過 ----------
    print('\nB runner 的每個 -p 參數都在節點宣告過')
    for node, src in NODE_SRC.items():
        body = open(src, encoding='utf-8').read()
        declared = set(re.findall(r"p\(\s*'([A-Za-z0-9_]+)'", body))
        # ROS 內建參數不經節點自行宣告
        declared |= {'use_sim_time', 'start_type_description_service'}
        # 從 runner 取該節點的 spawn 區塊
        m = re.search(r'ros2 run ammr_wholebody_mpc ' + node + r'(.*?)(?=\nspawn |\nsay |\Z)',
                      runner, re.S)
        used = set(re.findall(r'-p ([A-Za-z0-9_]+):=', m.group(1))) if m else set()
        check(f'{node}：取到 spawn 區塊', bool(m))
        missing = sorted(used - declared)
        check(f'{node}：{len(used)} 個參數都已宣告', not missing,
              '' if not missing else f'  **未宣告：{missing}**')

    # ---------- C 話題接線 ----------
    print('\nC 話題接線成對')
    def pubs_subs(path, prefix=''):
        """抓發布／訂閱的主題。主題名不是字面值時（變數、參數預設值），
        額外解析其來源 —— **不能因為抓不到就當成沒接線**。"""
        b = open(path, encoding='utf-8').read()
        pub = set(re.findall(r"create_publisher\([^,]+,\s*'([^']+)'", b))
        sub = set(re.findall(r"create_subscription\([^,]+,\s*'([^']+)'", b))
        # (a) 參數預設值形式的主題（安全節點的 contact_phase_topic）
        for nm, val in re.findall(r"p\('([A-Za-z0-9_]+_topic)',\s*'([^']+)'", b):
            sub.add(val)
        # (b) 以表格驅動訂閱（錄製器的 TOPICS / FIELD_TOPICS）
        for t in re.findall(r"\('(/[^']+)',\s*Float\d+MultiArray", b):
            sub.add(t)
        for t in re.findall(r"'(?:dist|safety)':\s*'(/[^']+)'", b):
            sub.add(t)
        fix = lambda t: (prefix + t[1:]) if t.startswith('~') else t
        return {fix(t) for t in pub}, {fix(t) for t in sub}

    P, S = {}, {}
    for tag, path, pre in (
            ('distance', os.path.join(PKG, 'arm_link_distance.py'),
             '/arm_link_distance'),
            ('safety', os.path.join(PKG, 'wholebody_safety_node.py'),
             '/wholebody_safety'),
            ('solver', os.path.join(HERE, 'coman_pull_solver_node.py'), ''),
            ('endpoint', os.path.join(HERE, 'isaac_coman_drawer_sim.py'), ''),
            ('recorder', os.path.join(HERE, 'coman_diag_record.py'), '')):
        P[tag], S[tag] = pubs_subs(path, pre)
    for topic, src_tag, dst_tags in (
            ('/coman/cmd_meta', 'solver', ('safety',)),
            ('/wholebody_safety/cmd_meta', 'safety', ('endpoint',)),
            ('/wholebody_safety/diag_fields', 'safety',
             ('endpoint', 'recorder')),
            ('/arm_link_distance/diag_fields', 'distance',
             ('endpoint', 'recorder')),
            ('/arm_link_distance/diag', 'distance', ('endpoint', 'recorder')),
            ('/wholebody_safety/diag', 'safety', ('endpoint', 'recorder')),
            ('/coman/contact_phase', 'solver', ('safety',)),
            ('/coman/task_state', 'endpoint', ('solver',)),
            ('/arm_link_distance/obstacle_names', 'distance', ('solver',))):
        check(f'{topic} 由 {src_tag} 發布', topic in P[src_tag])
        for d in dst_tags:
            check(f'  ↳ {d} 訂閱', topic in S[d])
    check('runner 啟動診斷錄製器',
          'coman_diag_record.py' in runner)

    # ---------- D 診斷欄位按名稱索引 ----------
    print('\nD 診斷欄位一律按名稱索引')
    sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
    from ammr_wholebody_mpc.wholebody_safety_node import DIAG_FIELDS
    from ammr_wholebody_mpc.arm_link_distance import (DIAG_FIELDS_BASE,
                                                      DIAG_TIGHT_SUFFIX)
    check('安全 diag：tf_age 與 node_ms 不同欄',
          DIAG_FIELDS.index('tf_age') != DIAG_FIELDS.index('node_ms'),
          f'  {DIAG_FIELDS.index("tf_age")} vs {DIAG_FIELDS.index("node_ms")}')
    ep = open(os.path.join(HERE, 'isaac_coman_drawer_sim.py'),
              encoding='utf-8').read()
    for nm in ('reason', 'filter_ms', 'node_ms', 'src_paired', 'n_rows',
               'node_cycle_ms', 'dup_dropped'):
        check(f"執行端按名稱取 '{nm}'", f"'{nm}'" in ep)
    # 執行端不得再出現對 diag 的硬編碼位置索引
    hard = re.findall(r"\bd\[(1[0-9]|[89])\]", ep)
    check('執行端已無 diag 的硬編碼位置索引', not hard,
          '' if not hard else f'  **殘留 d[{"], d[".join(hard)}]**')
    rec = open(os.path.join(HERE, 'coman_diag_record.py'),
               encoding='utf-8').read()
    check('錄製器的欄位名單來自上游（不自帶硬編碼）',
          'diag_fields' in rec and "'n_occluded'" not in rec)

    # ---------- E 規格鏈 ----------
    print('\nE 規格鏈一致')
    v1 = yaml.safe_load(open(f'{SPECS}/wb_coman_drawer20_criteria_v1.yaml',
                             encoding='utf-8'))
    s1 = yaml.safe_load(open(f'{SPECS}/wb_coman_drawer20_supplement_s1.yaml',
                             encoding='utf-8'))
    r11 = yaml.safe_load(open(f'{SPECS}/coman_r1_1_pair_gaps.yaml',
                              encoding='utf-8'))
    c1 = yaml.safe_load(open(
        f'{SPECS}/wb_coman_drawer20_v1_corrigendum_c1.yaml', encoding='utf-8'))
    check('v1 仍為 frozen 且 sha 未變',
          v1['status'] == 'frozen'
          and sha(f'{SPECS}/wb_coman_drawer20_criteria_v1.yaml')
          == s1['references']['criteria_v1_sha256_16'])
    check('勘誤 C1 已核准且門檻未變',
          c1['status'] == 'approved'
          and c1['correction']['threshold_unchanged'] is True
          and abs(float(c1['correction']['threshold_m'])
                  - float(v1['P_pull']['P6_retreat_clear_m'])) < 1e-12)
    check('S1 已掛載 C1',
          any(x['file'].startswith('wb_coman_drawer20_v1_corrigendum_c1')
              for x in s1['references'].get('v1_corrigenda', [])))
    check('R1.1 已核准且為十五組',
          r11['status'] == 'approved' and len(r11['pairs']) == 15)
    gaps = {p['pair']: float(p['g_pair_m']) for p in r11['pairs']}
    check('S1 的 g_by_pair 與 R1.1 逐項相符',
          {k: float(v) for k, v in
           s1['pair_avoidance']['g_by_pair'].items()} == gaps)
    rg = dict(re.findall(r'([A-Za-z0-9_]+):([A-Za-z0-9_]+):([0-9.]+)',
                         re.search(r'PAIR_GAP="\$\{PAIR_GAP:-([^}]*)\}"',
                                   runner).group(1)) and
              [(f'{a}|{b}', float(c)) for a, b, c in
               re.findall(r'([A-Za-z0-9_]+):([A-Za-z0-9_]+):([0-9.]+)',
                          re.search(r'PAIR_GAP="\$\{PAIR_GAP:-([^}]*)\}"',
                                    runner).group(1))])
    check('runner 的 PAIR_GAP 與 R1.1 逐項相符', rg == gaps,
          '' if rg == gaps else f'  **差異 {sorted(set(rg.items()) ^ set(gaps.items()))}**')
    need = {k.split('|')[0] for k in gaps}
    have = {x.split(':')[0] for x in re.search(
        r'PAIR_ROWS="\$\{PAIR_ROWS:-([^}]*)\}"', runner).group(1).split(',')}
    check('間距被放寬的連桿都有必要配對列', need <= have,
          '' if need <= have else f'  **缺 {sorted(need - have)}**')
    blocking = [i['id'] for i in (s1.get('open_issues') or [])
                if i.get('severity') == 'blocking']
    print(f'\n  S1 status = {s1["status"]}；blocking 未決項 = {blocking or "無"}')

    # ---------- F 模式條件 ----------
    print('\nF 模式條件：runner 的模式與執行端守衛相容')
    ep = open(os.path.join(HERE, 'isaac_coman_drawer_sim.py'),
              encoding='utf-8').read()
    m_rule = re.search(r'_n_expect\s*=\s*(\d+)\s*if\s*a\.free_base\s*else\s*(\d+)',
                       ep)
    check('執行端的固定關節期望值是**模式相依**', m_rule is not None,
          f'  free_base→{m_rule.group(1)}、fix_base→{m_rule.group(2)}'
          if m_rule else '  **仍是寫死值**')
    isaac_blk = re.search(r'spawn isaac(.*?)(?=\nsay )', runner, re.S)
    isaac_args = isaac_blk.group(1) if isaac_blk else ''
    free = '--free-base' in isaac_args
    check('runner 使用 --free-base（開放底盤）', free)
    if m_rule and free:
        check('該模式的期望值為 0 個 world→根固定關節',
              int(m_rule.group(1)) == 0)
    check('守衛不符期望仍會中止（保護未移除）', 'return 10' in ep)
    for flag in ('--cmd-source wb9', '--machine', '--attach-on-handover'):
        check(f'runner 傳 {flag}', flag in isaac_args)

    # ---------- G 任務數值 ----------
    print('\nG 任務數值：兩端與 v1 一致')
    stroke = float(re.search(r'STROKE="\$\{STROKE:-([0-9.]+)\}"',
                             runner).group(1))
    tgt_v1 = float(v1['profile']['target_stroke_m'])
    check('runner 的 STROKE 與 v1 target_stroke_m 相同',
          abs(stroke - tgt_v1) < 1e-12, f'  {stroke*1000:.1f} mm')
    check('runner 把目標傳給**執行端**（不只求解端）',
          '--pull-target-m "$STROKE"' in isaac_args,
          '' if '--pull-target-m "$STROKE"' in isaac_args
          else '  **執行端會沿用案例值**')
    check('runner 把行程傳給求解端', '--stroke-m "$STROKE"' in runner)
    sys.path.insert(0, HERE)
    from coman_handover_state import HandoverMachine, load_spec
    _mm = HandoverMachine(load_spec(
        f'{SPECS}/wb_coman_drawer20_criteria_v1.yaml'))
    check('v1 狀態機的目標與容差取自 v1（正式資格）',
          abs(_mm.target - tgt_v1) < 1e-12
          and abs(_mm.p1 - float(v1['P_pull']['P1_final_opening_err_m_max']))
          < 1e-12,
          f'  {_mm.target*1000:.1f} mm ±{_mm.p1*1000:.2f} mm')
    case = yaml.safe_load(open(
        os.path.join(WS, 'src/my_omnibot_description/config/'
                         'manipulation_cases.yaml'), encoding='utf-8'))
    ctol = float(case['cases']['drawer_open_a_fixed']['tolerance']['opening_m'])
    check('案例容差與 v1 容差不同 ⇒ 舊判定必須標明非正式',
          abs(ctol - _mm.p1) > 1e-9
          and "'legacy_case_tol_arrived_sim_t'" in ep
          and "'formal_acceptance_source'" in ep,
          f'  案例 ±{ctol*1000:.0f} mm vs v1 ±{_mm.p1*1000:.2f} mm')
    check('協同執行端已無未標註的 arrived_sim_t 欄名',
          "'arrived_sim_t':" not in ep)

    print('\n入口核對：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
