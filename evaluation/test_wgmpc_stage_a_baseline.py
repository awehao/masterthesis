"""基線趟次：抽屜接線、取值規則、門檻產生、以及**措辭的界線**。

為什麼要測「抽屜真的被建出來」：E2 模擬先前只在場景守衛裡**放行**
/World/drawer，卻沒有任何程式建立它 —— 而 build_usd 的 root 是
/World/drawer_unit，路徑分段比對下前者根本不匹配後者。那種組合會讓
`--mode solver_drawer` 靜默跑出一個**沒有抽屜**的場景，守衛照樣通過
（放行子樹 0 個碰撞體），距離節點則對著不存在的抽屜算屏障。
"""
import ast, json, os, re, sys
import numpy as np
import yaml
sys.path.insert(0, 'evaluation')
sys.path.insert(0, 'src/ammr_wholebody_mpc')

OK = []
def chk(c, m):
    OK.append(bool(c)); print(f'{"ok  " if c else "FAIL"}  {m}')

SIM = open('evaluation/isaac_wholebody_sim_e2.py').read()
CODE = '\n'.join(l for l in SIM.split('\n') if not l.lstrip().startswith('#'))

# ---------- A. 抽屜真的被建出來，且路徑與放行子樹一致 ----------
import drawer_asset as DA
_src = open('evaluation/drawer_asset.py').read()
root_default = re.search(r"def build_usd\(stage, spec: dict, pose, "
                         r"root: str = '([^']+)'\)", _src).group(1)
subtree = re.search(r"^DRAWER_SUBTREE = '([^']+)'", SIM, re.M).group(1)
chk(subtree == root_default,
    f'A1 DRAWER_SUBTREE {subtree!r} == build_usd 的 root 預設 {root_default!r}')
chk('DA.build_usd(stage, _dspec, _dpose, root=DRAWER_SUBTREE)' in CODE,
    'A2 solver_drawer 模式真的呼叫 build_usd 建抽屜')
chk(CODE.index('build_usd') < CODE.index('foreign = ['),
    'A3 **先建抽屜再跑場景守衛**（否則核的是還沒有抽屜的場景）')
chk("_dauth['root'] != DRAWER_SUBTREE" in CODE,
    'A4 建出來的 root 與放行子樹不符即中止')
for k in ('drive_stiffness', 'drive_damping', 'drive_max_force'):
    chk(k in CODE, f'A5 讀回核對 {k}（非零即中止；抽屜必須是被動件）')
chk("return 14" in CODE, 'A6 抽屜前提不成立時以非零碼中止')

# 子樹比對的語意：/World/drawer 不該匹配 /World/drawer_unit/...
# 用 AST 取整個函式定義，不用 regex 抓到空行就停（docstring 裡就有空行）
_tree = ast.parse(SIM)
_fn = next(n for n in _tree.body
           if isinstance(n, ast.FunctionDef) and n.name == '_under')
_u = {}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), '<u>', 'exec'), _u)
under = _u['_under']
chk(not under('/World/drawer_unit/drawer', ('/World/drawer',)),
    'A7 舊常數 /World/drawer **不會**匹配 /World/drawer_unit/... '
    '（這正是先前的缺陷）')
chk(under('/World/drawer_unit/drawer/handle_bar', (subtree,)),
    'A8 新常數匹配抽屜子樹')

# ---------- B. 開度與接觸力分開記錄 ----------
cols = re.findall(r"'(drawer_[a-z_]+)'", SIM)
for c in ('drawer_opening', 'drawer_vy', 'drawer_contact_fx',
          'drawer_contact_fy', 'drawer_contact_fz', 'drawer_contact_fmag'):
    chk(c in cols, f'B1 log 欄位有 {c}')
chk('track_contact_forces=True' in CODE,
    'B2 抽屜剛體開啟接觸力追蹤（接觸的獨立證據）')
chk('get_contact_force_matrix' in CODE, 'B3 逐步讀接觸力')
chk('DY0 - float(_dp[0][1])' in CODE,
    'B4 開度 = DY0 − 抽屜剛體世界 y（與 isaac_drawer_sim 同式）')
chk('drawer_v=None, DY0=None' in CODE and 'drawer_v, DY0)' in CODE,
    'B5 drawer_v 與 DY0 **顯式傳入** loop（不經 globals；否則 NameError）')
chk("_dop = _dvy = float('nan')" in CODE,
    'B6 讀不到留 NaN，**不補零**（補零會把「沒量到」偽裝成「沒有接觸」）')
chk('--init-arm-q' in SIM and 'INIT_ARM_Q' in CODE,
    'B7 支援 --init-arm-q（基線要停在階段 A 的起始姿態）')

# ---------- C. 取值規則**執行前定版** ----------
RULE = yaml.safe_load(open('evaluation/results/wgmpc_stage_a_baseline_rule.yaml'))
rk = [k for k in RULE if k.startswith('取值規則')][0]
R = RULE[rk]
chk(R['規則名'] == 'max_abs_plus_resolution', f'C1 規則名 {R["規則名"]}')
chk('全窗' in R['開度門檻']['式子'] and '末值' in R['開度門檻']['全窗而不是末值'],
    'C2 規則明寫取全窗最大值，不取末值（漂移不保證單調）')
chk(float(R['開度門檻']['解析度預留量_m']) > 0, 'C3 開度有解析度預留量')
chk(float(R['接觸力門檻']['解析度預留量_n']) > 0, 'C4 接觸力有解析度預留量')
chk(RULE['狀態'].startswith('執行前定版'), f'C5 狀態標為執行前定版')
chk('工程偵測門檻' in RULE['門檻的範圍（措辭限制）']['成立'],
    'C6 範圍明寫是「這次模擬配置的工程偵測門檻」')
chk(any('漂移上界' in x for x in RULE['門檻的範圍（措辭限制）']['不成立']),
    'C7 明寫**不是**所有情況的漂移上界')
chk(any('交棒' in x for x in RULE['基線趟次**不**驗證什麼']),
    'C8 明寫基線不驗證交棒流程')
chk(int(RULE['熱況條件']['熱中止線_c']) == 92
    and int(RULE['熱況條件']['起跑溫度門檻_c']) == 42,
    'C9 熱中止線 92（不得提高）、起跑門檻 42')

# ---------- D. 門檻產生：套用規則，不事後調整 ----------
from wgmpc_stage_a_drawer_check import derive_threshold, classify
tol, meta = derive_threshold([0.0, 1e-6, -3e-6, 2e-6], resolution=1e-9,
                             label='開度')
chk(abs(tol - (3e-6 + 1e-9)) < 1e-18,
    f'D1 門檻 = 全窗最大絕對值 3e-06 ＋ 預留 1e-09（得 {tol:.12g}）')
chk(meta['baseline_peak_abs'] == 3e-6 and meta['window'].startswith('全窗'),
    'D2 依據記下全窗峰值與取窗方式')
chk('工程偵測門檻' in meta['scope'], 'D3 依據記下門檻的範圍')
for bad, why in [(dict(series=[float('nan')] * 3, resolution=1e-9),
                  '基線全為 NaN'),
                 (dict(series=[0.0, 1e-6], resolution=None),
                  '沒給解析度預留量'),
                 (dict(series=[0.0, 1e-6], resolution=float('nan')),
                  '預留量非有限')]:
    try:
        derive_threshold(**bad); chk(False, f'D4 {why} 應拒絕但沒有')
    except ValueError as e:
        chk(True, f'D4 {why} 正確拒絕：{str(e)[:30]}…')
try:
    derive_threshold([0.0], rule='whatever', resolution=1e-9)
    chk(False, 'D5 未知規則應拒絕')
except ValueError:
    chk(True, 'D5 未知規則正確拒絕（不接受臨時發明的規則）')

# ---------- E. 判定措辭：位移與接觸**獨立** ----------
r = classify([0.0, 0.004, 0.0], tol_m=0.001)
chk(r['displacement']['over_baseline'] is True,
    'E1 開度 4 mm > 門檻 1 mm ⇒ 位移超過靜止基線')
chk(r['displacement']['claim_supported'] == '抽屜位移超過靜止基線',
    f'E2 可支持的敘述只有「抽屜位移超過靜止基線」（得 '
    f'{r["displacement"]["claim_supported"]}）')
chk('不能' in r['displacement']['claim_NOT_supported']
    and '碰到' in r['displacement']['claim_NOT_supported'],
    'E3 同時寫明**不能**單憑開度證明碰到或推了抽屜')
chk(r['contact']['verdict'] == 'undetermined',
    f'E4 沒有接觸證據時接觸**無法判定**（得 {r["contact"]["verdict"]}）'
    ' —— 不因位移超標就說碰到了')
chk('undetermined' in r['verdict'] and 'displacement_over_baseline' in r['verdict'],
    f'E5 合併判定同時說出兩軸：{r["verdict"]}')
for k in ('moved', 'touched', 'pushed', '推動'):
    chk(k not in json.dumps(r, ensure_ascii=False),
        f'E6 判定輸出不含會被誤讀為接觸的字眼 {k!r}')

# 有接觸力證據時
r = classify([0.0] * 3, tol_m=0.001, contact_fmag=[0.0, 0.5, 0.0],
             contact_force_tol_n=1e-9)
chk(not r['displacement']['over_baseline']
    and r['contact']['verdict'] == 'contact_evidence',
    'E7 位移未超標但接觸力超標 ⇒ 位移未超過基線 ＋ 有接觸證據（兩軸獨立）')
r = classify([0.0] * 3, tol_m=0.001, contact_fmag=[0.0] * 3,
             contact_force_tol_n=1e-9)
chk(r['contact']['verdict'] == 'no_contact_evidence',
    'E8 接觸力全為 0 ⇒ 可用證據未顯示接觸')
r = classify([0.0] * 3, tol_m=0.001, contact_fmag=[0.0] * 3,
             contact_force_tol_n=None)
chk(r['contact']['verdict'] == 'undetermined'
    and '沒有門檻' in r['contact']['force']['note'],
    'E9 有接觸力紀錄但**沒有門檻** ⇒ 不判定（噪訊底線沒量過不猜）')
r = classify([0.0] * 3, tol_m=0.001, d_min=[0.1, -0.001, 0.1])
chk(r['contact']['verdict'] == 'contact_evidence',
    'E10 真值最短距離 <= 0 也是接觸證據')
try:
    classify([0.0] * 3, tol_m=None); chk(False, 'E11 沒給開度門檻應拒絕')
except ValueError:
    chk(True, 'E11 沒給開度門檻正確拒絕')
try:
    classify([float('nan')] * 3, tol_m=0.001)
    chk(False, 'E12 開度全為 NaN 應拒絕')
except ValueError:
    chk(True, 'E12 開度全為 NaN 正確拒絕（缺資料不等於沒有位移）')

# ---------- F. 基線腳本的不變量 ----------
B = open('evaluation/run_wgmpc_stage_a_baseline.sh').read()
BC = '\n'.join(l for l in B.split('\n') if not l.lstrip().startswith('#'))
chk('wgmpc_wg2_node.py' not in BC and 'wgmpc_prepos_node.py' not in BC
    and 'arm_vel_adapter.py' not in BC,
    'F1 基線**不起任何命令來源**（無 W-GMPC、無前置調姿、無 adapter）')
chk('--init-arm-q "$INIT_ARM_Q"' in BC, 'F2 機器人停在指定起始姿態')
chk('--physics-dt "$PHYSICS_DT"' in BC, 'F3 物理步長顯式傳入（與階段 A 對齊）')
chk('--drawer-asset "$ASSET"' in BC and '--drawer-pose "$DRAWER_POSE"' in BC,
    'F4 抽屜資產與位姿顯式傳入')
chk('MODE="solver_drawer"' in BC, 'F5 模式與階段 A 相同')
chk('START_TEMP_C' in BC and 'exit 78' in BC,
    'F6 起跑溫度門檻不過即不啟動')
chk('92' in BC and '不得提高' in B, 'F7 熱中止線 92 且標明不得提高')
chk('wgmpc_stage_a_baseline_derive.py' in BC,
    'F8 跑完依規則產生門檻')
chk(BC.index('isaac_wholebody_sim_e2.py') < BC.index('baseline_derive'),
    'F9 先跑趟次再產生門檻（規則在趟次之前就定版，門檻在趟次之後才算）')
chk('不驗證' in B, 'F10 腳本自己寫明不驗證哪些事')
m = re.search(r'OBSERVE_SIM_S="\$\{OBSERVE_SIM_S:-(\d+)\}"', BC)
chk(m is not None and int(m.group(1)) == 60,
    f'F11 觀察時長預設 60 s（與階段 A 的模擬時間預算相同）')

print(f'\n{sum(OK)}/{len(OK)} 通過')
sys.exit(0 if all(OK) else 1)
