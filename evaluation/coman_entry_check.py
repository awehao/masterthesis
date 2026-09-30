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
import hashlib
import math, os, re, sys
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

    # ---------- H 低速框與執行端 E2 相容（L1）----------
    print('\nH 低速框與執行端 E2 相容（L1）')
    l1 = yaml.safe_load(open(f'{SPECS}/coman_low_speed_box_l1.yaml',
                             encoding='utf-8'))
    check('L1 已核准', l1.get('status') == 'approved', f"  {l1.get('status')}")
    # **逐行**解析，兩種形式都要：`NAME="${NAME:-值}"` 與 `NAME=值`。
    # 先前用一條跨行正則，[^:] 會吃掉換行 ⇒ 抓到別的變數的預設值。
    _WANT = ('VMAX_BASE_LIN', 'VMAX_BASE_ANG', 'VMAX_ARM',
             'E2_BASE_LIN', 'E2_BASE_ANG', 'E2_ARM_RATE', 'E2_MAX_CMD_AGE')
    rv = {}
    for _ln in runner.splitlines():
        _m = re.match(r'([A-Z0-9_]+)=(.*)$', _ln.strip())
        if not _m or _m.group(1) not in _WANT:
            continue
        _val = _m.group(2).strip().strip('"')
        _d = re.match(r'^\$\{[A-Z0-9_]+:-([\d.eE+-]+)\}$', _val)
        rv[_m.group(1)] = _d.group(1) if _d else _val
    check('runner 定義了七個低速框／E2 變數', len(rv) == 7, f'  {sorted(rv)}')
    lv = l1['values']
    for k, want in (('VMAX_BASE_LIN', lv['base_lin_per_axis_mps']),
                    ('VMAX_BASE_ANG', lv['base_ang_mps']),
                    ('VMAX_ARM', lv['arm_per_joint_rps'])):
        check(f'runner 的 {k} 與 L1 相同',
              k in rv and abs(float(rv[k]) - float(want)) < 1e-12,
              f"  {rv.get(k)} vs {want}")
    # **執行端的門檻不由 runner 決定** —— runner 的 E2_* 只是比對用的副本，
    # 必須與執行端 argparse 的預設逐項相同，否則比對基準會與實際門檻漂移。
    for k, flag in (('E2_BASE_LIN', 'base-lin-max'),
                    ('E2_BASE_ANG', 'base-ang-max'),
                    ('E2_ARM_RATE', 'arm-rate-max'),
                    ('E2_MAX_CMD_AGE', 'max-cmd-age-s')):
        m = re.search(r"add_argument\('--" + flag
                      + r"',\s*type=float,\s*default=([\d.]+)", ep)
        check(f'runner 的 {k} == 執行端 --{flag} 預設',
              bool(m) and k in rv
              and abs(float(rv[k]) - float(m.group(1))) < 1e-12,
              f"  runner {rv.get(k)} vs 原始碼 {m.group(1) if m else '未找到'}")
    # 逐軸框對**範數**門檻：兩軸同時到頂 ⇒ sqrt(2) 倍
    if {'VMAX_BASE_LIN', 'E2_BASE_LIN'} <= set(rv):
        worst = math.sqrt(2.0) * float(rv['VMAX_BASE_LIN'])
        check('兩軸同時到頂的平面範數 <= E2 範數門檻',
              worst <= float(rv['E2_BASE_LIN']) + 1e-12,
              f"  {worst:.6f} <= {rv['E2_BASE_LIN']}")
    if {'VMAX_BASE_ANG', 'E2_BASE_ANG'} <= set(rv):
        check('角速度框 <= E2 門檻',
              float(rv['VMAX_BASE_ANG']) <= float(rv['E2_BASE_ANG']) + 1e-12)
    if {'VMAX_ARM', 'E2_ARM_RATE'} <= set(rv):
        check('關節速率框 <= E2 門檻',
              float(rv['VMAX_ARM']) <= float(rv['E2_ARM_RATE']) + 1e-12)
    check('runner 把低速框傳給**安全層**',
          '-p vmax_base_lin:="$VMAX_BASE_LIN"' in runner)
    check('runner 把低速框傳給**求解端**',
          '--vmax-base-lin "$VMAX_BASE_LIN"' in runner)
    check('runner 把 E2 門檻傳給求解端做自檢',
          '--e2-base-lin "$E2_BASE_LIN"' in runner)
    check('runner 在起求解端**之前**做讀回比對',
          runner.index('coman_lowspeed_readback.py')
          < runner.index('coman_pull_solver_node.py'))
    check('讀回比對失敗會具名中止（exit 67）', 'exit 67' in runner)
    sol = open(os.path.join(HERE, 'wholebody_pregrasp.py'),
               encoding='utf-8').read()
    check('求解端的框進入 cfg.vmax（**不是**對輸出裁切）',
          'self.cfg.vmax = _vm' in sol)
    check('求解端與讀回比對用**同一個**共用模組',
          'from coman_low_speed_box import' in sol
          and 'from coman_low_speed_box import' in open(
              os.path.join(HERE, 'coman_lowspeed_readback.py'),
              encoding='utf-8').read())
    check('求解端覆寫只縮不放（共用 tighten_vmax）', 'tighten_vmax(' in sol)
    check('求解端自檢呼叫共用 e2_compat_violations',
          'e2_compat_violations(' in sol)
    lsb = open(os.path.join(HERE, 'coman_low_speed_box.py'),
               encoding='utf-8').read()
    check('共用模組的範數上界用 sqrt(2) 倍（不是只比單軸）',
          'SQRT2 * float(vmax_base_lin_per_axis)' in lsb)
    check('求解端自檢失敗即啟動中止（fail closed）',
          'raise SystemExit(' in sol and '啟動中止（fail closed）' in sol)
    # **實際執行**守門判斷，不只比對字串
    sys.path.insert(0, HERE)
    from coman_low_speed_box import e2_compat_violations as _v
    e2 = (float(rv['E2_BASE_LIN']), float(rv['E2_BASE_ANG']),
          float(rv['E2_ARM_RATE']))
    check('守門：L1 值判為相容',
          not _v(float(rv['VMAX_BASE_LIN']), float(rv['VMAX_BASE_ANG']),
                 float(rv['VMAX_ARM']), *e2))
    check('守門：main4 的硬體框判為不相容（三項全中）',
          len(_v(0.2775, 1.1327, 3.141593, *e2)) == 3)
    check('守門：逐軸剛好等於範數門檻仍判為不相容（√2 生效）',
          bool(_v(e2[0], 0.1, 0.5, *e2)))
    # ---- O7 的排程修正 ----
    ps = open(os.path.join(HERE, 'coman_pull_solver_node.py'),
              encoding='utf-8').read()
    check('每輪都在 guard **之前**做有界回呼處理',
          ps.index('self._pump()\n                why = self.guard()')
          < ps.index('_t_solve0'))
    check('_pump 有界（次數與阻塞時間都有上限）',
          'range(max(0, n - 1))' in ps and 'timeout_sec=0.0)' in ps)
    check('超時跳過錯過的 slot（不連續追趕）',
          'slot += _sk' in ps and 'n_deadline_miss' in ps)
    check('deadline miss 有記錄（不以降頻掩蓋）',
          'self.worst_late_ms' in ps and 'missed_slots' in ps
          and '--rate' not in ps)
    check('發布**之前**再驗有效性並可丟棄結果',
          '丟棄本次求解結果（不發布）' in ps
          and ps.index('_drop is not None') < ps.index('self.pub.publish(m)'))
    check('過期判斷用**模擬時鐘**之差（不與牆鐘相減）',
          '_in_age = _ref - _src_sim_t' in ps
          and '_ref = self.sim_now()' in ps)
    check('年齡參考取兩個模擬時鐘來源的較新者（時鐘落後不會把年齡算小）',
          "_ref = max(_ref, float(self.task.get('sim_t', _ref)))" in ps)
    check('in_age 與 solve_ms 在紀錄中各自成欄（不互相比較）',
          'in_age=round(_in_age, 6)' in ps and 'solve_ms=round(_solve_ms, 4)' in ps)
    check('同一來源狀態不重複求解', 'n_same_state_skip' in ps
          and '_last_src_sim_t' in ps)
    check('耗時用單調牆鐘', '_t.perf_counter()' in ps and '_tt.monotonic()' in ps)
    check('runner 以執行端的命令年齡界限傳入 --max-input-age',
          '--max-input-age "$E2_MAX_CMD_AGE"' in runner)
    check('deadline miss 等排程計數會**落盤**',
          'def sched_summary' in ps and 'n_deadline_miss' in ps
          and "sched=_sched" in open(os.path.join(HERE, 'wholebody_pregrasp.py'),
                                     encoding='utf-8').read())
    check('solve_ms 已分段（約束組裝／QP／運動學）',
          all(x in ps for x in ('cons_ms=', 'qp_ms=', 'kin_ms=')))
    dr = open(os.path.join(HERE, 'coman_diag_record.py'), encoding='utf-8').read()
    check('O8：診斷錄製改用 try_shutdown 並吞 ExternalShutdownException',
          'rclpy.try_shutdown()' in dr
          and 'except ExternalShutdownException' in dr
          and 'rclpy.shutdown()' not in dr)

    check('執行端封存三個界限（不只 arm_rate_max）',
          "'coman_low_speed_interface'" in ep
          and "'base_lin_max_mps'" in ep and "'base_ang_max_rps'" in ep
          and "'arm_rate_max_rps'" in ep)

    print('\n入口核對：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
