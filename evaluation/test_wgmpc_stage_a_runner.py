"""階段 A 運行器的不變量，以及六維目標的姿態驗證。

為什麼要測運行器本身：這些前提不是註解，是啟動中止條件。只要有一行被改掉
（例如從自由空間趟次複製 `freespace_confirmed:=true`、或加回
`--assume-initial-rest`），趟次仍會跑完、仍會產出數字，但衡量的是另一件事。
"""
import ast, os, re, sys
import numpy as np
sys.path.insert(0, 'evaluation')
sys.path.insert(0, 'src/ammr_wholebody_mpc')

OK = []
def chk(c, m):
    OK.append(bool(c)); print(f'{"ok  " if c else "FAIL"}  {m}')

R = open('evaluation/run_wgmpc_stage_a.sh').read()
# 去掉註解行，避免把說明文字當成程式
CODE = '\n'.join(l for l in R.split('\n') if not l.lstrip().startswith('#'))
# **再剝掉 python heredoc 區塊**：那裡面的字串是落盤內容，不是傳給節點的旗標。
# 不剝的話，run_config.json 裡寫「不傳 --assume-initial-rest」會被誤判成有傳。
SH_ONLY, _keep = [], True
for _l in CODE.split('\n'):
    if re.match(r"^python3 - .*<<'?EOF'?", _l) or re.match(r'^\w+=\$\(python3 - ', _l):
        _keep = False
    if _keep:
        SH_ONLY.append(_l)
    if _l.strip() == 'EOF':
        _keep = True
SH_ONLY = '\n'.join(SH_ONLY)

# ---------- A. 模式與餘裕 ----------
chk('--mode "$MODE"' in CODE and re.search(r'^MODE="?solver_drawer"?$', CODE, re.M),
    'A1 模式鎖為 solver_drawer（不可由環境變數覆寫）')
chk(re.search(r'^JOINT_MARGIN="0\.05"$', CODE, re.M),
    'A2 joint_margin 鎖為 0.05（不可由環境變數覆寫）')
chk('--joint-margin "$JOINT_MARGIN"' in CODE,
    'A3 執行端收到 --joint-margin')
chk('check_margin_consistency' in CODE,
    'A4 啟動時核對求解器與執行端的 joint_margin 一致')

# ---------- B. freespace_confirmed ----------
chk('-p freespace_confirmed:=false' in CODE,
    'B1 安全層明確設 freespace_confirmed:=false')
chk('freespace_confirmed:=true' not in CODE,
    'B2 腳本裡沒有任何 freespace_confirmed:=true')
chk('wgmpc_stage_a_entry.py' in CODE,
    'B3 執行期讀回核對有被呼叫（不只靠傳對參數）')

# ---------- C. 真值模式與具名幾何由產生器輸出 ----------
chk('gen_drawer_obstacle_params.py' in CODE,
    'C1 距離節點參數由產生器輸出（幾何單一來源）')
chk('DIST_ARGS' in CODE and 'arm_link_distance' in CODE
    and CODE.index('mapfile -t DIST_ARGS') < CODE.index('arm_link_distance'),
    'C2 產生器輸出真的傳進距離節點（先產生再起節點）')
chk(not re.search(r'-p obstacles:=\[', CODE),
    'C3 腳本裡沒有手抄的 obstacles 字串')
chk(not re.search(r'-p scene_truth_approved:=\[', CODE),
    'C4 腳本裡沒有手抄的核准清單')

# ---------- C2. 參數逐引數傳遞，不經 eval ----------
chk('mapfile -t DIST_ARGS' in CODE,
    'C2-1 參數讀成 bash 陣列（mapfile），不是文字 blob')
chk('eval spawn dist' not in CODE and 'eval ' not in SH_ONLY,
    'C2-2 腳本裡沒有 eval —— eval 會吃掉 ros2 參數陣列需要的引號')
chk('"${DIST_ARGS[@]}"' in CODE,
    'C2-3 以 "${DIST_ARGS[@]}" 逐引數展開（每個引數原樣傳遞）')
chk('"${#DIST_ARGS[@]}" -eq 8' in CODE,
    'C2-4 引數數自核（8 個）⇒ 產生器輸出變形時啟動就擋住')
import subprocess, yaml
_o = subprocess.run([sys.executable, 'evaluation/gen_drawer_obstacle_params.py',
                     '--pose', '0.0,1.45', '--argv'],
                    capture_output=True, text=True)
_args = _o.stdout.strip().split('\n')
chk(len(_args) == 8, f'C2-5 --argv 輸出 8 個引數（得 {len(_args)}）')
chk(all(_args[i] == '-p' for i in range(0, len(_args), 2)),
    'C2-6 偶數位都是 -p（引數成對）')
_parsed = {}
for i in range(0, len(_args), 2):
    _n, _, _v = _args[i + 1].partition(':=')
    _parsed[_n] = yaml.safe_load(_v)       # ros2 CLI 也是 YAML 解析
chk(isinstance(_parsed.get('obstacles'), list)
    and len(_parsed['obstacles']) == 14
    and all(isinstance(x, str) for x in _parsed['obstacles']),
    'C2-7 obstacles 以 ros2 的 YAML 解析得到 14 個**字串**'
    f'（得 {type(_parsed.get("obstacles")).__name__}）')
chk(isinstance(_parsed.get('scene_truth_approved'), list)
    and len(_parsed['scene_truth_approved']) == 14,
    'C2-8 scene_truth_approved 解析得到 14 項')
chk(_parsed.get('scene_truth') is True,
    f'C2-9 scene_truth 解析為布林真（得 {_parsed.get("scene_truth")!r}）')
chk(':' in _parsed['obstacles'][0] and ',' in _parsed['obstacles'][0],
    'C2-10 設定字串內含 : 與 , 仍被當成單一字串（引號有留在詞元裡）')

# ---------- 起始路徑切分：spawn（生成）／move（前置調姿） ----------
# 運行器現在有兩條起始路徑，不變量各自不同，必須分開核，不能用全文搜尋
# ——否則 spawn 分支的 --assume-initial-rest 會被誤讀成 move 分支也傳了。
# 腳本裡有**數個** START_MODE 分支（起跑訊息、[A] 段、[B] 段的標題），
# 只取 [A] 段那一個：它的 spawn 分支裡有起始構型核對的啟動指令。
_IF = 'if [ "$START_MODE" = "spawn" ]; then'
SPAWN = MOVE = None
_at = -1
while True:
    _at = SH_ONLY.find(_IF, _at + 1)
    if _at < 0:
        break
    _a1 = SH_ONLY.find('\nelse\n', _at)
    _a2 = SH_ONLY.find('\nfi\n', _a1)
    if _a1 < 0 or _a2 < 0:
        continue
    _sp, _mv = SH_ONLY[_at:_a1], SH_ONLY[_a1:_a2]
    if 'wgmpc_stage_a_start_check.py' in _sp:
        SPAWN, MOVE = _sp, _mv
        break
chk(SPAWN is not None and MOVE is not None,
    'D0 找得到 [A] 段的兩條起始路徑分支')
if SPAWN is None:
    raise SystemExit('**切不出 [A] 段分支，後續不變量無法核**')
chk('wgmpc_stage_a_start_check.py' in SPAWN
    and 'wgmpc_stage_a_handover.py' not in SPAWN,
    'D0a spawn 分支走起始構型核對，不走交棒判定')
chk('wgmpc_stage_a_handover.py' in MOVE
    and 'wgmpc_stage_a_start_check.py' not in MOVE,
    'D0b move 分支走交棒判定，不走起始構型核對')
chk('wgmpc_prepos_node.py' not in SPAWN,
    'D0c spawn 分支**不起前置調姿節點** ⇒ 起始位置不去調 j3')
chk('wgmpc_prepos_node.py' in MOVE, 'D0d move 分支才起前置調姿節點')

# ---------- D. u_prev 的來源在兩條路徑上各自正確 ----------
chk('--assume-initial-rest' not in MOVE,
    'D1 move 分支**沒有**傳 --assume-initial-rest ⇒ 第一輪 u_prev 只能取自'
    f'套用回報（move 分支 {MOVE.count("--assume-initial-rest")} 次）')
chk('UPREV_ARGS=()' in MOVE,
    'D1b move 分支把 UPREV_ARGS 清空 ⇒ 展開後不會多帶旗標')
chk('--assume-initial-rest' not in SPAWN and 'UPREV_ARGS=()' in SPAWN,
    'D1c spawn 分支**也不**傳 --assume-initial-rest：adapter 持續送零命令，'
    'u_prev 取得到實際套用回報')
chk('--rest-tol "$REST_TOL"' in SPAWN,
    'D1e spawn 閘門核「手臂確實靜止」（交棒判定不核這條）')
chk('--u-prev-policy strict "${UPREV_ARGS[@]}"' in SH_ONLY,
    'D1d 旗標以陣列展開傳給 W-GMPC ⇒ 空陣列時不會留下空字串引數')
chk('--u-prev-policy strict' in CODE, 'D2 u_prev 政策為 strict')
# spawn 路徑的 u_prev 是**假設**，不是量到的；這點必須落盤。
_SS = open('evaluation/wgmpc_stage_a_start_check.py').read()
chk('assumed_initial_rest' not in _SS,
    'D3 起始核對**不再**使用初始靜止假設（實測否證該前提）')
chk('修訂史' in _SS,
    'D3b 起始核對保留修訂史：原前提是什麼、被什麼實測否證')
chk('not_a_handover' in _SS,
    'D4 起始核對明確標示自己**不是**交棒判定（判準不可混用）')

# ---------- E. 兩段分開、不同時在線 ----------
# 比**啟動指令**的位置，不是檔名首次出現的位置（版本雜湊清單也會提到檔名）。
_RUN_WG2 = 'python3 -u evaluation/wgmpc_wg2_node.py'
_RUN_HO = 'python3 evaluation/wgmpc_stage_a_handover.py'
_RUN_SS = 'python3 evaluation/wgmpc_stage_a_start_check.py'
chk(MOVE.index('kill_named prepos') < MOVE.index(_RUN_HO),
    'E1 move 分支在交棒判定之前先終止前置來源')
chk(_RUN_WG2 in SH_ONLY and _RUN_HO in SH_ONLY and _RUN_SS in SH_ONLY,
    'E2a 三個啟動指令都在 shell 層（不是只出現在版本清單裡）')
chk(SH_ONLY.index(_RUN_HO) < SH_ONLY.index(_RUN_WG2)
    and SH_ONLY.index(_RUN_SS) < SH_ONLY.index(_RUN_WG2),
    'E2 兩條路徑的起始閘門都在 W-GMPC 之前')
chk(MOVE.index('wgmpc_prepos_node.py') < MOVE.index(_RUN_HO),
    'E3 前置調姿在交棒判定之前')
# ---------- E6. 生成式起始構型：Isaac 與閘門必須核同一組角度 ----------
chk('--init-arm-q "$INIT_ARM_Q"' in SH_ONLY,
    'E6 生成角度以 $INIT_ARM_Q 傳給 Isaac')
chk('--start-q "$INIT_ARM_Q"' in SPAWN,
    'E7 起始核對用**同一個** $INIT_ARM_Q ⇒ 核的是實際要求的構型')
chk('START_MODE="${START_MODE:-spawn}"' in SH_ONLY,
    'E8 預設起始模式為 spawn ⇒ 起始位置不去調 j3')
# 紀錄標籤也要分路徑：spawn 趟次沒有前置調姿也沒有交棒，
# 沿用 move 的標籤會讓 wb_run.json 描述一件沒發生的事。
chk('--run-label "$RUN_LABEL"' in SH_ONLY,
    'E9 Isaac 的 run-label 由變數帶入（不是寫死 move 路徑的描述）')
_m = re.search(r'RUN_LABEL="([^"]*)"\s*\nelse\s*\n\s*RUN_LABEL="([^"]*)"',
               SH_ONLY)
chk(_m is not None and '不調 j3' in _m.group(1)
    and '前置調姿' in _m.group(2) and '前置調姿' not in _m.group(1),
    'E10 spawn 的標籤不提前置調姿與交棒，move 的才提')
chk('PREPOS_RUN_ID' in CODE and '--run-id "$PREPOS_RUN_ID"' in CODE,
    'E4 前置調姿用自己的 run_id ⇒ 封裝紀錄能分辨命令來自誰')
chk('--run-id "$RUN_ID"' in CODE,
    'E5 W-GMPC 用趟次的 run_id')
m = re.search(r'HO_RC=\$\{PIPESTATUS\[0\]\}(.*?)fi', CODE, re.S)
chk(m is not None and 'exit 72' in m.group(1),
    'E6 交棒失敗 ⇒ exit（不讓 W-GMPC 上線）')
chk(m is not None and 'stop_request' in m.group(1),
    'E7 交棒失敗時請執行端走受控停止')

# ---------- F. 六維目標 ----------
chk('--target $TARGET --target-rot $TARGET_ROT' in CODE,
    'F1 目標同時給位置與姿態（六維）')
chk('compute_pre_contact_pause' in CODE,
    'F2 目標由當步抽屜位姿算出')
chk('check_target' in CODE, 'F3 核對實際使用的目標 == 當步位姿推得的')
chk("target.json" in CODE, 'F4 目標的來源與時間落盤')
# 第一次出現是 [0/9] 的離線預覽（在交棒之前，本來就該在）。
# 要核的是**目標區塊**那一次 —— 取最後一次出現。
chk(CODE.rindex('compute_pre_contact_pause') > CODE.index('handover.json'),
    'F5 實際算目標的那一次在交棒之後（用的是交棒時刻的抽屜位姿）')
chk(CODE.index('compute_pre_contact_pause') < CODE.index('isaac_wholebody_sim_e2'),
    'F5b 另有一次離線預覽在起 Isaac 之前（前提先核完才啟動）')

# ---------- G. 前置調姿的配置 ----------
chk('--vel-max "$PREPOS_VEL"' in CODE and '--acc-max "$PREPOS_ACC"' in CODE,
    'G1 前置調姿的速度與加速度上限可追溯')
chk(re.search(r'PREPOS_VEL="\$\{PREPOS_VEL:-0\.35\}"', CODE),
    'G2 速度上限預設 0.35（與凍結的預抓取軌跡同值）')
chk(re.search(r'PREPOS_ACC="\$\{PREPOS_ACC:-0\.7\}"', CODE),
    'G3 加速度上限預設 0.7')
chk('--tail-s "$PREPOS_TAIL_S"' in CODE,
    'G4 軌跡後有零速率尾段（讓設定點停穩）')

# ---------- H. 交棒設定的自洽 ----------
qs = float(re.search(r'QUIET_S="\$\{QUIET_S:-([0-9.]+)\}"', CODE).group(1))
from ammr_wholebody_mpc.wgmpc_handover import HandoverConfig
LO = (-6.283185, -2.356194, -0.061087, -6.283185, -2.233612, -6.283185)
HI = (6.283185, 2.356194, 2.935644, 6.283185, 2.233612, 6.283185)
c = HandoverConfig(max_cmd_age_s=0.2, quiet_s=qs, joint_lower=LO, joint_upper=HI)
c.validate()
chk(qs >= 0.2, f'H1 QUIET_S 預設 {qs} >= max_cmd_age_s 0.2（設定自洽）')

# ---------- I. 抽屜判定接線（判定邏輯由 test_wgmpc_stage_a_baseline 詳測） ----------
chk('wgmpc_stage_a_drawer_check.py' in CODE, 'I1 收尾有抽屜判定')
chk('DRAWER_BASELINE' in CODE and 'DRAWER_TOL_M' in CODE
    and 'DRAWER_CONTACT_TOL_N' in CODE,
    'I2 門檻由外部給（量過才判定），腳本不內建猜測值')
chk('--drawer-asset "$ASSET"' in CODE and '--drawer-pose "$DRAWER_POSE"' in CODE,
    'I3 抽屜場景參數傳給 Isaac（否則場景裡沒有抽屜）')
import wgmpc_stage_a_drawer_check as DC
try:
    DC.classify([0.0] * 5, tol_m=None); chk(False, 'I4 沒給門檻應拒絕')
except ValueError as e:
    chk(True, f'I4 沒給門檻正確拒絕：{str(e)[:30]}…')
_r = DC.classify([0.0, 0.0001, 0.0], tol_m=0.001,
                 contact_fmag=[0.0] * 3, contact_force_tol_n=1e-9)
chk(not _r['displacement']['over_baseline']
    and _r['contact']['verdict'] == 'no_contact_evidence',
    'I5 位移未超基線且接觸力未超門檻')
chk('不能' in _r['displacement']['claim_NOT_supported'],
    'I6 判定輸出自帶措辭限制（開度不能證明接觸）')
chk('wgmpc_stage_a_baseline_rule.yaml' in CODE
    or 'DRAWER_BASELINE' in CODE,
    'I7 門檻的來源可追溯到基線趟次')

# ---------- J. 六維目標的姿態驗證（真的呼叫，不只看原始碼） ----------
from wgmpc_wg2_node import validate_target_rot
from wgmpc_stage_a_entry import R_DES
Rv = validate_target_rot(R_DES.reshape(-1))
chk(np.allclose(Rv, R_DES), 'J1 R_DES 通過驗證且原值回傳')
for v, why in [
        ([1, 0, 0, 0, 1, 0, 0, 0], '只有 8 個值'),
        ([1, 0, 0, 0, 1, 0, 0, 0, float('nan')], '含 NaN'),
        ([1, 0, 0, 0, 1, 0, 0, 0, -1], '行列式 −1（反射）'),
        ([1, 0.1, 0, 0, 1, 0, 0, 0, 1], '非正交'),
        ([2, 0, 0, 0, 2, 0, 0, 0, 2], '含縮放')]:
    try:
        validate_target_rot(v); chk(False, f'J {why} 應拒絕但沒有')
    except ValueError as e:
        chk(True, f'J {why} 正確拒絕：{str(e)[:38]}…')
nsrc = open('evaluation/wgmpc_wg2_node.py').read()
ncode = '\n'.join(l for l in nsrc.split('\n') if not l.lstrip().startswith('#'))
chk('target_rot_deg' in ncode and '_bail' in ncode
    and re.search(r'target_rot_deg\)\) > 0\.0', ncode),
    'J7 --target-rot-deg 非零即中止（先前宣告卻從未被讀取 ⇒ 靜默無效）')
chk('validate_target_rot(a.target_rot)' in ncode,
    'J8 main() 走的是同一個驗證函式')

print(f'\n{sum(OK)}/{len(OK)} 通過')
sys.exit(0 if all(OK) else 1)
