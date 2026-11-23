"""基線門檻的兩道新檢查，各以**必須被拒絕的反例**驗證。

反例一：**10 秒提早中止的紀錄** ⇒ 不得產生門檻。
        偏低的門檻會讓階段 A 的任何微小數值都看起來超標。
反例二：**不同櫃體位置的門檻檔** ⇒ 階段 A 不得套用。
        門檻只對同一份資產、同一個擺放、同一個步長、同一套開度讀法有效。

反例是真的建出紀錄檔再跑程式，不是只比對原始碼。
"""
import json, os, subprocess, sys, tempfile
import yaml
import numpy as np
sys.path.insert(0, 'evaluation')
import wgmpc_stage_a_drawer_check as DC

OK = []
def chk(c, m):
    OK.append(bool(c)); print(f'{"ok  " if c else "FAIL"}  {m}')

PY_ = sys.executable
RULE = 'evaluation/results/wgmpc_stage_a_baseline_rule.yaml'
ASSET = 'src/my_omnibot_description/config/drawer_unit.yaml'
DERIVE = 'evaluation/wgmpc_stage_a_baseline_derive.py'
CHECKER = 'evaluation/wgmpc_stage_a_drawer_check.py'
SHA = __import__('hashlib').sha256(open(ASSET, 'rb').read()).hexdigest()
DT = 0.01
FORMULA = ('opening = DY0 − drawer_rigid_body_world_y；'
           'DY0 取於設定增益之後、進入物理迴圈之前（與 isaac_drawer_sim 同式）')

# 真實模擬的 log 一直都有 base_x／base_y，夾具也要有 ——
# 少了它們會讓所有舊反例多一條「無法確認底盤未移動」的拒絕理由，
# 那是夾具缺欄，不是被測邏輯的問題。
COLS = ['t', 'integrating', 'base_lin_meas', 'base_x', 'base_y',
        'drawer_opening', 'drawer_vy',
        'drawer_contact_fx', 'drawer_contact_fy', 'drawer_contact_fz',
        'drawer_contact_fmag']


def make_record(sim_s=60.0, stop='sim_limit', pose=(0.0, 1.45),
                opening_peak=2.0e-7, contact_peak=0.0, recv=0,
                read_fail=0, cf_fail=0, nan_opening=0, base_lin=0.0,
                drive_stiff=0.0, dt=DT, sha=SHA, formula=FORMULA,
                schema='drawer_unit/1', name='drawer_unit_a'):
    n = int(round(sim_s / dt))
    rng = np.random.default_rng(0)
    # **隨機漫步，步長與 n 無關**：數值漂移是累積的，窗越長走得越遠。
    # 若步長隨 n 縮放，短窗反而可能得到較大的峰值 —— 那是夾具的假象。
    step = opening_peak / np.sqrt(60.0 / dt)      # 以 60 s 的步數為基準
    op = np.cumsum(rng.normal(0, step, n)) if n else np.zeros(0)
    cf = np.zeros(n)
    if n and contact_peak:
        cf[n // 3] = contact_peak
    for i in range(nan_opening):
        op[i] = float('nan')
    log = [[round(i * dt, 4), 0, base_lin, 0.0, 0.0,
            float(op[i]), 0.0, 0.0, 0.0, 0.0, float(cf[i])] for i in range(n)]
    return {
        'schema': 'wb_sim/1', 'mode': 'solver_drawer',
        'stop_reason': stop, 'sim_time_s': sim_s, 'wall_s': sim_s * 3,
        'cmd_chain': {'received': recv, 'rejected': 0},
        'drawer_read_fail_steps': read_fail,
        'drawer_contact_read_fail_steps': cf_fail,
        'init_arm_q': [0.0, 0.0, 1.43728, 0.0, 0.0, 0.0],
        'drawer': {
            'asset': ASSET, 'asset_sha256': sha, 'schema': schema,
            'name': name, 'pose_xy': list(pose), 'physics_dt_s': dt,
            'sim_limit_s': sim_s, 'dy0': 1.165,
            'opening_formula': formula,
            'authored': {'drive_stiffness': drive_stiff, 'drive_damping': 0.0,
                         'drive_max_force': 0.0, 'mass_kg': 2.0,
                         'linear_damping': 2.0, 'joint_friction': 0.0,
                         'limit_lower': 0.0, 'limit_upper': 0.22,
                         'root': '/World/drawer_unit'}},
        'log_cols': COLS, 'log': log,
    }


def truncate(rec, sim_s, stop='stop_request'):
    """把紀錄截成前 sim_s 秒 —— **提早中止的真實形狀就是截斷**。

    用截斷而不是另外產生一條較短的序列：前綴的最大值在數學上必然 <= 全窗，
    所以「短觀察會壓低門檻」這件事由構造保證，不靠隨機種子碰巧成立。
    """
    import copy
    r = copy.deepcopy(rec)
    ix = r['log_cols'].index('t')
    r['log'] = [row for row in r['log'] if row[ix] <= sim_s + 1e-12]
    r['sim_time_s'] = float(sim_s)
    r['stop_reason'] = stop
    r['drawer']['sim_limit_s'] = float(sim_s)
    return r


# **測試夾具用的位移門檻**，不等於核准定版 —— 正式門檻值 Howard 於
# 2026-10-03 明確不核准，產生器也沒有預設值。這裡給值只是為了讓其他
# 反例能測到各自要測的那一項（I3 刻意不給，驗證「不給即拒絕」）。
DISP = ['--base-disp-limit-mm', '1.0']


def run(args):
    r = subprocess.run([PY_] + args, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


TD = tempfile.mkdtemp(prefix='baseline_guards_')

# ================= 正例：完整 60 s、正常收尾、未受命令 =================
FULL = make_record()
good = os.path.join(TD, 'good.json')
json.dump(FULL, open(good, 'w'))
tol_good = os.path.join(TD, 'tol_good.json')
rc, out = run([DERIVE, good, '--rule', RULE, '--observe-sim-s', '60',
               *DISP, '--out', tol_good])
chk(rc == 0, f'A1 完整基線 ⇒ 產生門檻（rc={rc}）')
if rc == 0:
    b = json.load(open(tol_good))
    _rec = json.load(open(good))
    _ix = _rec['log_cols'].index('drawer_opening')
    _peak = max(abs(r[_ix]) for r in _rec['log'])
    chk(abs(b['opening_threshold']['threshold'] - (_peak + 1e-9)) < 1e-18,
        f'A2 門檻 = 全窗峰值 {_peak:.6g} ＋ 預留 1e-09（得 '
        f'{b["opening_threshold"]["threshold"]:.6g}）')
    chk(b['config_fingerprint']['asset_sha256'] == SHA
        and b['config_fingerprint']['drawer_pose_xy'] == [0.0, 1.45],
        'A3 門檻帶配置指紋（資產內容雜湊 ＋ 擺放）')
    chk(b['baseline_run']['acceptance']['verdict'] == 'pass',
        'A4 驗收證據一併落盤')

# ================= 反例一：10 秒提早中止 =================
print('\n--- 反例一：10 秒提早中止的紀錄 ---')
for sim_s, stop, why in [
        (10.0, 'stop_request', '10 s ＋ 受控停止（運行器逾時那條路徑）'),
        (10.0, 'sim_limit', '10 s 但 stop_reason 寫 sim_limit（只短觀察）'),
        (60.0, 'stop_request', '跑滿 60 s 但收尾是 stop_request'),
        (60.0, 'wall_limit', '跑滿 60 s 但收尾是 wall_limit'),
        (60.0, 'cmd_chain_fail', '命令鏈失效收尾')]:
    p = os.path.join(TD, f'early_{sim_s}_{stop}.json')
    json.dump(truncate(FULL, sim_s, stop), open(p, 'w'))
    o = os.path.join(TD, f'tol_early_{sim_s}_{stop}.json')
    rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
                   *DISP, '--out', o])
    chk(rc != 0, f'B {why} ⇒ **拒絕產生門檻**（rc={rc}）')
    if rc != 0:
        rep = json.load(open(o))
        chk(rep['verdict'] == 'rejected' and rep['reasons'],
            f'  理由已落盤：{rep["reasons"][0][:52]}…')

# 10 s 若未被擋，門檻會偏低多少 —— 說明這道檢查的必要性
p10 = os.path.join(TD, 'd10.json')
json.dump(truncate(FULL, 10.0, 'sim_limit'), open(p10, 'w'))
o10 = os.path.join(TD, 'tol10.json')
rc10, _ = run([DERIVE, p10, '--rule', RULE, '--observe-sim-s', '10',
               *DISP, '--out', o10])
chk(rc10 == 0, 'B6 同一筆 10 s 紀錄在「只要求 10 s」時可產生門檻 '
               '⇒ 擋住它的確實是**觀察時長**，不是別的毛病')
if rc10 == 0 and os.path.exists(tol_good):
    t10 = json.load(open(o10))['opening_threshold']['threshold']
    t60 = json.load(open(tol_good))['opening_threshold']['threshold']
    chk(t10 < t60,
        f'B7 同一趟截到 10 s 的門檻 {t10:.6g} < 完整 60 s 的 {t60:.6g}'
        f'（低 {100*(1-t10/t60):.1f}%）⇒ 提早中止確實會壓低門檻')

# ================= 其他驗收條件 =================
print('\n--- 其他驗收條件 ---')
for kw, why in [
        (dict(recv=5), '命令鏈收到 5 筆命令（不是靜止基線）'),

        (dict(read_fail=3), '有 3 步讀不到抽屜狀態'),
        (dict(cf_fail=2), '有 2 步讀不到接觸力'),
        (dict(nan_opening=4), '開度有 4 筆 NaN（關鍵讀值不完整）'),
        (dict(drive_stiff=1.0), '抽屜 drive_stiffness 不為 0（不是被動件）'),
        (dict(dt=0.005), '物理步長 0.005 不等於規則要求的 0.01')]:
    p = os.path.join(TD, 'x.json')
    json.dump(make_record(**kw), open(p, 'w'))
    rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
                   *DISP, '--out', os.path.join(TD, 'xo.json')])
    chk(rc != 0, f'C {why} ⇒ 拒絕（rc={rc}）')

# 「底盤有動」現在以**位移**表達：速度 1 mm/s 但位置不動的夾具自相矛盾
#（那表示機器人其實沒動），判準改成位移之後它不再是有效反例。
# 真正的反例是 I1（中途跑出 4 mm），在下面的「反例五」。
_mv = np.zeros(6000); _mv[3000:] = 0.002        # 中途起移動 2 mm 且不回來
p = os.path.join(TD, 'moved.json')
json.dump(make_record(), open(p, 'w'))          # 先佔位，下面用 make_base 覆寫

# ================= 反例二：不同櫃體位置的門檻檔 =================
print('\n--- 反例二：不同櫃體位置的門檻檔 ---')
stage = os.path.join(TD, 'stage_a.json')
json.dump(make_record(sim_s=5.0, stop='sim_limit', opening_peak=1.0e-7),
          open(stage, 'w'))
chk(run([CHECKER, stage, '--baseline', tol_good])[0] == 0,
    'D1 同一配置的門檻 ⇒ 可用（判定通過）')

other = os.path.join(TD, 'other_pose.json')
json.dump(make_record(pose=(0.0, 1.60)), open(other, 'w'))
tol_other = os.path.join(TD, 'tol_other.json')
rc, _ = run([DERIVE, other, '--rule', RULE, '--observe-sim-s', '60',
             *DISP, '--out', tol_other])
chk(rc == 0, 'D2 另一個櫃體位置的基線本身是合法的（1.60 而非 1.45）')
rc, out = run([CHECKER, stage, '--baseline', tol_other])
chk(rc == 78, f'D3 **不同櫃體位置的門檻 ⇒ 階段 A 拒用**（rc={rc}，要求 78）')
chk('drawer_pose_xy' in out and '1.6' in out,
    f'D4 拒用理由指名 drawer_pose_xy 與兩邊的值')
chk('不套用另一趟的門檻' in out, 'D5 訊息說明這是拒用而非降級套用')

for kw, field, why in [
        (dict(sha='0' * 64), 'asset_sha256', '資產**內容**改過（檔名相同）'),
        (dict(dt=DT), None, None),
        (dict(schema='drawer_unit/2'), 'drawer_schema', '資產 schema 升版'),
        (dict(name='drawer_unit_b'), 'drawer_name', '資產名稱不同'),
        (dict(formula='opening = joint_position'), 'opening_formula',
         '開度讀法改了')]:
    if field is None:
        continue
    p = os.path.join(TD, 'o2.json')
    json.dump(make_record(**kw), open(p, 'w'))
    to = os.path.join(TD, 'to2.json')
    if run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
            *DISP, '--out', to])[0] != 0:
        chk(False, f'E {why}：基線本身應可產生門檻但失敗'); continue
    rc, out = run([CHECKER, stage, '--baseline', to])
    chk(rc == 78 and field in out,
        f'E {why} ⇒ 拒用，理由指名 {field}（rc={rc}）')

# 物理步長不同：門檻合法但階段 A 的趟次步長不同 ⇒ 拒用
p = os.path.join(TD, 'dt5.json')
json.dump(make_record(sim_s=5.0, stop='sim_limit', dt=0.005), open(p, 'w'))
rc, out = run([CHECKER, p, '--baseline', tol_good])
chk(rc == 78 and 'physics_dt_s' in out,
    f'E6 階段 A 的步長 0.005 與門檻的 0.01 不同 ⇒ 拒用（rc={rc}）')

# 缺指紋的舊版門檻檔 ⇒ 拒用
old = os.path.join(TD, 'tol_old.json')
json.dump({'schema': 'wgmpc_stage_a_baseline_threshold/1',
           'opening_threshold': {'threshold': 1e-6}}, open(old, 'w'))
rc, out = run([CHECKER, stage, '--baseline', old])
chk(rc == 78 and '配置指紋' in out,
    f'E7 舊版門檻檔沒有配置指紋 ⇒ 拒用（rc={rc}）')

# 被拒絕的基線產出的檔 ⇒ 拒用
rej = os.path.join(TD, 'tol_rej.json')
json.dump({'verdict': 'rejected', 'reasons': ['觀察不完整']}, open(rej, 'w'))
rc, out = run([CHECKER, stage, '--baseline', rej])
chk(rc == 78 and '被拒絕的基線' in out,
    f'E8 被拒絕的基線產出的檔 ⇒ 拒用（rc={rc}）')

# ================= 反例三：時間戳不涵蓋全窗 =================
print('\n--- 反例三：筆數達標但時間戳不涵蓋全窗 ---')


def clip_tail(rec, last_t):
    """把 log 截到 last_t，但 sim_time_s 與 stop_reason **維持正常** ——
    這正是「筆數達標、收尾正常，資料卻集中在前段」的情形。"""
    import copy
    r = copy.deepcopy(rec)
    ix = r['log_cols'].index('t')
    r['log'] = [row for row in r['log'] if row[ix] <= last_t + 1e-12]
    return r


def drop_middle(rec, t0, t1):
    """挖掉中間一段（保留首尾）⇒ 製造中間缺口。"""
    import copy
    r = copy.deepcopy(rec)
    ix = r['log_cols'].index('t')
    r['log'] = [row for row in r['log']
                if not (t0 < row[ix] < t1)]
    return r


# 關鍵反例：sim_time_s=60、收尾 sim_limit、筆數 90%，但 log 只到 53.99 s
p = os.path.join(TD, 'short_tail.json')
rec = clip_tail(FULL, 53.99)
json.dump(rec, open(p, 'w'))
_n, _exp = len(rec['log']), 6000
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               *DISP, '--out', os.path.join(TD, 'o3.json')])
chk(rc != 0,
    f'G1 筆數 {_n}（{100*_n/_exp:.1f}%，過九成）、sim_time_s=60、收尾 sim_limit，'
    f'但末時間戳 53.99 s ⇒ **拒絕**（rc={rc}）')
chk('終點未涵蓋' in out and '53.99' in out,
    f'G2 理由指名終點未涵蓋與實際末時間戳')
chk('門檻會系統性偏低' in out, 'G3 理由說明後果')

# 剛好在容差內 ⇒ 通過（容差是兩個物理步）
p = os.path.join(TD, 'tail_ok.json')
json.dump(clip_tail(FULL, 59.99), open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               *DISP, '--out', os.path.join(TD, 'o4.json')])
chk(rc == 0, f'G4 末時間戳 59.99 s（距 60 不到兩個物理步）⇒ 通過（rc={rc}）')

# 起點未涵蓋
p = os.path.join(TD, 'late_start.json')
import copy as _cp
_r = _cp.deepcopy(FULL)
_ixt = _r['log_cols'].index('t')
_r['log'] = [row for row in _r['log'] if row[_ixt] >= 1.0]
json.dump(_r, open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               *DISP, '--out', os.path.join(TD, 'o5.json')])
chk(rc != 0 and '起點未涵蓋' in out, f'G5 第一筆時間戳 1.0 s ⇒ 拒絕（rc={rc}）')

# 中間缺口
p = os.path.join(TD, 'gap.json')
json.dump(drop_middle(FULL, 20.0, 25.0), open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               *DISP, '--out', os.path.join(TD, 'o6.json')])
chk(rc != 0 and '中間有缺口' in out,
    f'G6 挖掉 20–25 s（首尾仍涵蓋、筆數 91.7%）⇒ 拒絕（rc={rc}）')

# 時間戳遞減 / 含 NaN
for mk, why, key in [
        (lambda r: (r['log'].__setitem__(
            3000, r['log'][3000][:1] + r['log'][3000][1:]) or
            r['log'][3000].__setitem__(0, 1.0) or r), '時間戳遞減', '遞減'),
        (lambda r: (r['log'][3000].__setitem__(0, float('nan')) or r),
         '時間戳含 NaN', '非有限值')]:
    _r = _cp.deepcopy(FULL); _r = mk(_r)
    p = os.path.join(TD, 'bad_t.json'); json.dump(_r, open(p, 'w'))
    rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
                   *DISP, '--out', os.path.join(TD, 'o7.json')])
    chk(rc != 0 and key in out, f'G7 {why} ⇒ 拒絕（rc={rc}）')

# ================= 反例四：引用式 heredoc 不得引用 shell 變數 =================
print('\n--- 反例四：腳本的 heredoc ---')
import re as _re


def quoted_heredoc_violations(path):
    """找出**引用式** heredoc 內文裡引用 shell 變數的行。

    掃描器本身要寫對，否則反例會失效：
      * `<<'EOF'` 與 `<<EOF` 要分開認。用反向參照 `\\1` 寫成一條會讓
        未引用的那種完全匹配不到（未匹配群組的 `\\1` 不成立）。
      * heredoc 的**起始行本身**不是內文 —— 那一行出現 $VAR 是合法的
        （例如把路徑當引數傳）。
      * 註解行不是程式碼。
    """
    out, inblk, quoted = [], False, False
    start = 0
    for i, l in enumerate(open(path).read().split('\n'), 1):
        if not inblk:
            if l.lstrip().startswith('#'):
                continue          # 註解裡提到 <<'EOF' 不是區塊起點
            if _re.search(r"<<'EOF'", l):
                inblk, quoted, start = True, True, i
            elif _re.search(r'<<EOF\b', l):
                inblk, quoted, start = True, False, i
            continue              # 起始行本身不算內文
        if l.strip() == 'EOF':
            inblk = False
            continue
        if l.lstrip().startswith('#'):
            continue              # 註解不是程式碼
        if quoted and _re.search(r'\$[A-Za-z_{]', l):
            out.append((i, start, l.strip()[:48]))
    return out


# 掃描器自己要先有反例：故意造一個違規檔，確認它抓得到
_probe = os.path.join(TD, 'probe.sh')
open(_probe, 'w').write("python3 - <<'EOF'\nopen('$DIR/x.json')\nEOF\n")
chk(len(quoted_heredoc_violations(_probe)) == 1,
    'H0 掃描器對刻意造的違規檔抓到 1 處（掃描器本身可信）')
_probe2 = os.path.join(TD, 'probe2.sh')
open(_probe2, 'w').write('python3 - <<EOF\nopen("$DIR/x.json")\nEOF\n')
chk(not quoted_heredoc_violations(_probe2),
    'H0b 未引用式 heredoc 內的 $DIR 會正常展開 ⇒ 不算違規')

_probe3 = os.path.join(TD, 'probe3.sh')
open(_probe3, 'w').write(
    "# 說明：`<<'EOF'` 不會展開 $DIR\n"
    'python3 - "$DIR/x.json" <<\'EOF\'\n'
    'open(sys.argv[1])\n'
    'EOF\n')
chk(not quoted_heredoc_violations(_probe3),
    'H0c 註解提到 heredoc、起始行以引數傳路徑 ⇒ 不算違規'
    '（起始行本身不是內文）')

for _f in ('evaluation/run_wgmpc_stage_a_baseline.sh',
           'evaluation/run_wgmpc_stage_a.sh'):
    _bad = quoted_heredoc_violations(_f)
    chk(not _bad,
        f'H {os.path.basename(_f)}：引用式 heredoc 內沒有 shell 變數'
        + (f'；違規 {_bad}' if _bad else '（`<<\'EOF\'` 不會展開，'
           '引用了就會去開字面路徑）'))

# 門檻檔讀回那一段要檢查退出碼
_B = open('evaluation/run_wgmpc_stage_a_baseline.sh').read()
chk('python3 - "$DIR/drawer_threshold.json"' in _B,
    'H3 門檻檔讀回**以引數傳路徑**')
_seg = _B[_B.index('[4/4]'):]
chk('PIPESTATUS[0]' in _seg and 'exit 81' in _seg,
    'H4 該段檢查退出碼（腳本沒有 set -e，失敗後不得繼續宣稱完成）')
chk(_B.index('WALL_LIMIT=$(python3') < _B.index('isaac_wholebody_sim_e2.py'),
    'H5 牆鐘上限在**起 Isaac 之前**算好')
chk('exit 82' in _B and 'math.isfinite' in _B,
    'H6 牆鐘上限用浮點安全的算法，不合法即不啟動')
_Bcode = '\n'.join(l for l in _B.split('\n')
                   if not l.lstrip().startswith('#'))
chk('$((OBSERVE_SIM_S' not in _Bcode,
    'H7 程式碼裡不再用 shell 整數運算算牆鐘上限'
    '（註解可提及舊寫法；非整數會在 Isaac 起來後才炸）')

# ================= 反例五：底盤位移與 twist 異常 =================
print('\n--- 反例五：底盤判準 ---')
def make_base(n=6000, dt=0.01, path=None, twist=None):
    """在基礎夾具上覆寫 base_x 與 base_lin_meas 的時間序列。"""
    rec = make_record(sim_s=n * dt, dt=dt)
    ibx = COLS.index('base_x')
    ibl = COLS.index('base_lin_meas')
    out = []
    for i, row in enumerate(rec['log'][:n]):
        r = list(row)
        if path is not None:
            r[ibx] = float(path[i])
        if twist is not None:
            r[ibl] = float(twist[i])
        out.append(r)
    rec['log'] = out
    return rec


N = 6000
zeros = np.zeros(N)
# 1) 首末差小但**中途跑出去又回來** ⇒ 全窗最大位移必須抓到
mid = np.zeros(N); mid[2000:3000] = 0.004          # 中途 4 mm，最後回到 0
p = os.path.join(TD, 'mid_excursion.json')
json.dump(make_base(path=mid), open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               '--base-disp-limit-mm', '1.0', '--out', os.path.join(TD, 'b1.json')])
chk(rc != 0 and '全窗最大位移' in out,
    f'I1 首末差 0 但中途 4 mm ⇒ **拒絕**（rc={rc}）—— 首末差抓不到這種')
_e = json.load(open(os.path.join(TD, 'b1.json')))['evidence']
chk(abs(_e['base_disp_max_m'] - 0.004) < 1e-12
    and _e['base_disp_first_to_last_m'] == 0.0,
    f'I2 證據同時記下最大位移 {_e["base_disp_max_m"]*1e3:.3f} mm 與'
    f'首末差 {_e["base_disp_first_to_last_m"]*1e3:.3f} mm')

# 2) 門檻未給 ⇒ 拒絕並報出實測值（門檻尚未核准定版）
p = os.path.join(TD, 'nolimit.json')
json.dump(make_base(path=zeros), open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               '--out', os.path.join(TD, 'b2.json')])   # **刻意不帶 DISP**
chk(rc == 0, f'I3 不給 --base-disp-limit-mm ⇒ 取規則檔**已核定**的 1 mm'
             f'（Howard 2026-10-03）（rc={rc}）')
if rc == 0:
    _b2 = json.load(open(os.path.join(TD, 'b2.json')))
    chk(_b2['base_disp_limit_source'] == '規則檔（已核定）',
        'I3b 門檻檔記下來源是規則檔的已核定值，不是命令列')
    chk(abs(_b2['baseline_run']['acceptance']['evidence'][
            'base_disp_limit_m'] - 1.0e-3) < 1e-12,
        'I3c 用的就是 1.0 mm')

# 3) twist 與位移矛盾**但有停更簽章** ⇒ 刻畫後可通過
tw = np.full(N, 1.658e-3); tw[:41] = np.linspace(0.7e-3, 1.658e-3, 41)
pth = np.zeros(N); pth[:41] = np.linspace(0, 1.63e-4, 41); pth[41:] = 1.63e-4
p = os.path.join(TD, 'freeze.json')
json.dump(make_base(path=pth, twist=tw), open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               '--base-disp-limit-mm', '1.0', '--out', os.path.join(TD, 'b3.json')])
chk(rc == 0, f'I4 twist 矛盾但有停更簽章 ⇒ 通過（rc={rc}）')
if rc == 0:
    _b = json.load(open(os.path.join(TD, 'b3.json')))
    _e = _b['baseline_run']['acceptance']['evidence']
    chk(_e['freeze_signature'] is True and _e['base_twist_vs_disp_ratio'] > 10,
        f'I5 簽章與比值 {_e["base_twist_vs_disp_ratio"]:.1f} 倍都寫進門檻檔'
        f' —— 異常被**刻畫**而非移除')
    chk('機制未證實' in _e['freeze_note'],
        'I6 證據明標機制未證實（不宣稱已解釋成 PhysX 睡眠）')

# 4) twist 與位移矛盾**而且沒有簽章** ⇒ 未解釋的異常，拒絕
tw2 = np.full(N, 1.658e-3)
rng2 = np.random.default_rng(7)
tw2 += rng2.normal(0, 1e-5, N)          # 一直在變 ⇒ 沒有凍結
p = os.path.join(TD, 'nofreeze.json')
json.dump(make_base(path=pth, twist=tw2), open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               '--base-disp-limit-mm', '1.0', '--out', os.path.join(TD, 'b4.json')])
chk(rc != 0 and '未解釋的量測異常' in out,
    f'I7 twist 矛盾且**沒有**停更簽章 ⇒ 拒絕（rc={rc}）'
    f' —— 異常沒有被悄悄移除')

# 5) twist 與位移一致 ⇒ 不觸發簽章檢查
tw3 = np.zeros(N)
p = os.path.join(TD, 'consistent.json')
json.dump(make_base(path=zeros, twist=tw3), open(p, 'w'))
rc, out = run([DERIVE, p, '--rule', RULE, '--observe-sim-s', '60',
               '--base-disp-limit-mm', '1.0', '--out', os.path.join(TD, 'b5.json')])
chk(rc == 0, f'I8 twist 與位移一致 ⇒ 通過，不需簽章（rc={rc}）')

# ================= 反例六：規則版本紀錄 =================
print('\n--- 反例六：規則版本紀錄 ---')
_R = yaml.safe_load(open(RULE))
chk(_R['schema'].endswith('/2'), f'J1 規則 schema 已升版（{_R["schema"]}）')
_h = _R.get('修訂史')
chk(_h is not None and 'v1' in _h and 'v2' in _h, 'J2 有修訂史，v1／v2 分開')
chk(_h['v1']['性質'].startswith('執行前定版'), 'J3 v1 標為執行前定版')
_ph = [it for it in _h['v2']['逐條'] if 'post-hoc' in str(it['性質'])]
chk(len(_ph) == 2,
    f'J4 兩條 post-hoc 修訂逐條標示（{[it["條目"] for it in _ph]}）')
_unch = [it for it in _h['v2']['逐條'] if it['變更'] == '未變']
chk(any('取值規則' in it['條目'] for it in _unch),
    'J5 取值規則標為未變（它確實沒動）')
_st = _R['可覆寫參數的約束']['start_temp_c']
chk(_st.get('enforced') is False and len(_st['修訂紀錄']) == 2,
    'J6 起跑線已解除且修訂紀錄 2 筆')
_sa = _R['基線趟次的驗收條件（**門檻產生前必須全部成立**）']
_srec = _sa['時間戳涵蓋全窗']['起點涵蓋的修訂紀錄'][0]
chk('post-hoc' in _srec['修訂性質'] and '原判準' in _srec,
    'J7 起點修訂標為 post-hoc 且保留原判準')
chk('不宣稱原判準未變' in _srec['明確不宣稱'],
    'J8 明確不宣稱原判準未變（先前寫「取值規則本身未動」是誤導）')
_brec = _sa['未受命令驅動的修訂紀錄'][0]
chk('已核定為 1.0 mm' in _brec['門檻值的狀態']
    and '先前一度標為不核准' in _brec['門檻值的狀態'],
    'J9 底盤位移門檻已核定為 1.0 mm，且保留「先前一度不核准」的歷程')
chk('未證實' in _brec['機制（**假設，未證實**）'],
    'J10 停更機制標為假設未證實')
chk(len(_h['尚未產生門檻的趟次']) == 2,
    f'J11 兩趟未產生門檻的紀錄都在（{[x["run"] for x in _h["尚未產生門檻的趟次"]]}）')

# ================= 反例七：措辭限制（Howard 2026-10-03 收緊） =================
print('\n--- 反例七：措辭限制 ---')
_RR = yaml.safe_load(open(RULE))
_w = _RR['判定措辭（**本輪明確收準**）']
_cl = _w['接觸力讀取的能力與限制']
chk('能成功讀取' in _cl['已證明'], 'K1 規則明寫接觸力 API 已證明能讀取')
_ns = _cl['不成立的宣稱']
chk(any('不證明沒有接觸' in x for x in _ns),
    'K2 明寫淨合力為零**不證明沒有接觸**')
chk(any('不證明有接觸時感測器一定會回報' in x for x in _ns),
    'K3 明寫淨合力為零**不證明有接觸時感測器一定會回報**')
chk('正向對照' in _cl['還需要什麼'], 'K4 明寫還需要已知接觸的正向對照')
_rp = _w['可重現性的措辭']
chk('所測趟次可重現' in _rp['可以說'], 'K5 可說「所測趟次可重現」')
chk(any('所有條件下都是決定性' in x for x in _rp['不能說']),
    'K6 **不可**擴寫為「所有條件下都是決定性的」')
_ot = [x for x in _RR if x.startswith('開度與接觸力門檻')][0]
_oc = _RR[_ot]
chk('已核定' in _oc['狀態'], f'K7 開度／接觸力門檻已核定（{_oc["狀態"][:16]}…）')
chk(_oc['核定依據']['核定值']['開度_tol_m'] == 1e-9
    and _oc['核定依據']['核定值']['接觸力_tol_n'] == 1e-9,
    'K8 核定值為 1e-9 m / 1e-9 N')
chk(len(_oc['核定依據']['實際達成的趟次']) == 4,
    f'K9 四趟達成條件（{len(_oc["核定依據"]["實際達成的趟次"])} 筆）')
chk(all(x['開度峰值_m'] == 0.0 and x['接觸力峰值_n'] == 0.0
        for x in _oc['核定依據']['實際達成的趟次']),
    'K10 四趟的開度與接觸力峰值皆為 0（核定條件）')
chk(_oc['名稱限定'].startswith('它是「紀錄變化偵測門檻」'),
    'K11 門檻名稱限定為「紀錄變化偵測門檻」，不是物理漂移上界')
chk(_oc['超標只能說'] == '讀值超過靜止基線', 'K12 超標只能說「讀值超過靜止基線」')
chk(set(_oc['超標不能說的']) == {'確定有接觸', '確定抽屜被推動'},
    'K13 超標不能稱確定接觸或推動')

# 判定器的輸出也要帶這些限制（不只寫在規則檔）
_r = DC.classify([0.0] * 3, tol_m=1e-9, contact_fmag=[0.0] * 3,
                 contact_force_tol_n=1e-9)
chk('紀錄解析度' in _r['displacement']['threshold_is'],
    'K14 判定輸出說明門檻量的是紀錄解析度')
_sl = _r['contact']['sensing_limits']
chk('不證明沒有接觸' in _sl['zero_does_not_prove_no_contact']
    and '不證明有接觸時感測器一定會回報'
    in _sl['zero_does_not_prove_sensor_would_report']
    and '正向對照' in _sl['needs'],
    'K15 判定輸出自帶接觸偵測的三條能力限制')

# 底盤位移門檻已核定 ⇒ 不給命令列旗標也能產生門檻
_p = os.path.join(TD, 'ratified.json')
json.dump(make_base(path=zeros), open(_p, 'w'))
_rc, _out = run([DERIVE, _p, '--rule', RULE, '--observe-sim-s', '60',
                 '--out', os.path.join(TD, 'k.json')])
chk(_rc == 0, f'K16 位移門檻已核定 ⇒ 不給旗標也能產生門檻（rc={_rc}）')
if _rc == 0:
    _b = json.load(open(os.path.join(TD, 'k.json')))
    chk(_b['base_disp_limit_source'] == '規則檔（已核定）',
        f'K17 門檻檔記下位移門檻的來源（{_b["base_disp_limit_source"]}）')
    chk(_b['wording']['over_threshold_means'] == '讀值超過靜止基線',
        'K18 門檻檔帶措辭限制')

# ================= 參數閘門：起跑線不得調高 =================
print('\n--- 參數閘門 ---')
GATE = 'evaluation/wgmpc_stage_a_baseline_gate.py'
BASE = [GATE, '--rule', RULE, '--stage-config',
        'src/ammr_wholebody_mpc/config/wgmpc_stage_a.yaml',
        '--drawer-asset', ASSET]
def gate(**kw):
    d = dict(start_temp_c='42', observe_sim_s='60', physics_dt_s='0.01',
             mode='solver_drawer', joint_margin='0.05',
             drawer_pose='0.0,1.45')
    d.update(kw)
    return run(BASE + [f'--{k.replace("_", "-")}' for _ in [0] for k in []]
               + sum([[f'--{k.replace("_", "-")}', v] for k, v in d.items()],
                     []))
chk(gate()[0] == 0, 'F1 符合規則 ⇒ 通過')
chk(gate(start_temp_c='30')[0] == 0, 'F2 起跑線調低到 30 ⇒ 允許')
# **起跑線已解除**（規則檔 enforced: false，Howard 2026-10-03 授權），
# 所以它不再是啟動條件；任何值都通過，解除本身由 J6 與下面兩條核對。
for _v in ('60', '43', '999'):
    chk(gate(start_temp_c=_v)[0] == 0,
        f'F-解除 起跑溫度 {_v} ⇒ 通過（該條已解除為啟動條件）')
chk(yaml.safe_load(open(RULE))['可覆寫參數的約束'][
        'start_temp_c'].get('enforced') is False,
    'F-解除 規則檔明寫 enforced: false（解除有紀錄，不是閘門壞了）')
for kw, why in [
        (dict(observe_sim_s='10'), '觀察縮短到 10 s'),
        (dict(physics_dt_s='0.005'), '物理步長改成 0.005'),
        (dict(mode='solver_freespace'), '模式改成 solver_freespace'),
        (dict(joint_margin='0.0'), 'joint_margin 改成 0'),
        (dict(drawer_pose='0.0,1.60'), '櫃體位置與階段 A 不同')]:
    rc, out = gate(**kw)
    chk(rc == 80, f'F {why} ⇒ 不啟動（rc={rc}）')
chk(gate(observe_sim_s='120')[0] == 0, 'F10 觀察加長到 120 s ⇒ 允許')

# **非有限數值**：NaN／inf 與任何界的比較都是 False ⇒ 不先擋住會零違規通過
for kw, why in [
        (dict(observe_sim_s='NaN'), '觀察時長 NaN'),
        (dict(observe_sim_s='inf'), '觀察時長 inf'),


        (dict(physics_dt_s='NaN'), '物理步長 NaN'),
        (dict(joint_margin='NaN'), 'joint_margin NaN'),
        (dict(observe_sim_s='-5'), '觀察時長 −5（必須為正）'),
        (dict(physics_dt_s='0'), '物理步長 0'),
        (dict(drawer_pose='0.0,NaN'), '櫃體位姿含 NaN')]:
    rc, out = gate(**kw)
    chk(rc == 80, f'G8 {why} ⇒ 不啟動（rc={rc}）')

# 合法的非整數加長要**全程**可用：閘門過，而且牆鐘上限算得出整數
# 負號開頭的值 argparse 會當成選項（rc=2），要用 `=` 形式才真的傳進閘門 ——
# 測的要是**閘門的判定**，不是 argparse 的參數解析。
# 起跑溫度已解除 ⇒ 連非有限值都不再判定（它不被任何條件使用）。
# 其他**仍在生效**的數值欄位的非有限值檢查由上面的 G8 群組覆蓋。
for _v in ('-inf', 'NaN'):
    _r = run(BASE + ['--start-temp-c=' + _v, '--observe-sim-s', '60',
                     '--physics-dt-s', '0.01', '--mode', 'solver_drawer',
                     '--joint-margin', '0.05', '--drawer-pose', '0.0,1.45'])
    chk(_r[0] == 0,
        f'G8b 起跑溫度 {_v} ⇒ 通過（該條已解除，值不被使用）（rc={_r[0]}）')

chk(gate(observe_sim_s='90.5')[0] == 0,
    'G9 合法的非整數加長 90.5 s ⇒ 閘門通過')
_w = subprocess.run(
    [PY_, '-c',
     'import math,sys;v=float(sys.argv[1]);'
     'print(int(math.ceil(v*20.0+300.0)))', '90.5'],
    capture_output=True, text=True)
chk(_w.returncode == 0 and _w.stdout.strip().isdigit(),
    f'G10 90.5 s 的牆鐘上限算得出正整數 {_w.stdout.strip()}'
    f'（先前 shell 的 $(( )) 會在 Isaac 起來後才語法錯誤）')

import shutil
shutil.rmtree(TD, ignore_errors=True)
print(f'\n{sum(OK)}/{len(OK)} 通過')
sys.exit(0 if all(OK) else 1)
