"""場景真值遮蔽來源：每一條回退原因都要有反例，不只測成功路徑。

為什麼要逐條測：這個模式的價值完全在它的**前提**。只測「條件齊備時給 0」
等於沒測 —— 一個永遠回傳 0 的函式也會通過。
"""
import sys, numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_link_distance import (
    parse_obstacles, parse_scene_truth, scene_truth_occ,
    TRUTH_GIVEN, TRUTH_OFF, TRUTH_NOT_APPROVED, TRUTH_MODEL_MISMATCH,
    TRUTH_NO_POSE, TRUTH_POSE_STALE, TRUTH_REASON)

OK = []


def chk(cond, msg):
    OK.append(bool(cond))
    print(f'{"ok  " if cond else "FAIL"}  {msg}')


# ---------- A. 核准清單的解析 ----------
ap = parse_scene_truth(['cabinet/top:', 'handle/bar:drawer_unit', ''])
chk(ap == {'cabinet/top': '', 'handle/bar': 'drawer_unit'},
    f'A1 解析 name:model，空 model = 宣告靜態：{ap}')
for bad, why in [(['nocolon'], '缺冒號'),
                 ([':model'], '名稱為空'),
                 (['a:m1', 'a:m2'], '同名兩個模型')]:
    try:
        parse_scene_truth(bad); chk(False, f'A2 {why} 應拋出但沒有')
    except ValueError:
        chk(True, f'A2 {why} 正確拋出')

# ---------- B. 條件齊備 ----------
stat, = parse_obstacles(['cabinet/top::box:0.6,0.45,0.02:0,1.45,0.89:'])
dyn, = parse_obstacles(['handle/bar:drawer_unit:cylinder:0.005,0.2:0,-0.285,0.55:'])
dyn.T_world_link = np.eye(4)
AP = {'cabinet/top': '', 'handle/bar': 'drawer_unit'}
STAMP = {'drawer_unit': 100.0}

occ, c = scene_truth_occ(stat, AP, 100.0, STAMP, 0.5, True)
chk(occ == 0.0 and c == TRUTH_GIVEN, f'B1 靜態宣告＋設定給位姿 ⇒ 給真值 occ={occ}')
occ, c = scene_truth_occ(dyn, AP, 100.2, STAMP, 0.5, True)
chk(occ == 0.0 and c == TRUTH_GIVEN, f'B2 動態＋位姿新鮮（0.2 < 0.5）⇒ 給真值 occ={occ}')

# ---------- C. 逐條回退 ----------
occ, c = scene_truth_occ(dyn, AP, 100.2, STAMP, 0.5, False)
chk(occ is None and c == TRUTH_OFF, 'C1 模式關閉 ⇒ 不給真值')

occ, c = scene_truth_occ(None, AP, 100.2, STAMP, 0.5, True)
chk(occ is None and c == TRUTH_NOT_APPROVED, 'C2 障礙物為 None ⇒ 不給真值')

other, = parse_obstacles(['wall_12::box:1,1,1:5,5,0.5:'])
occ, c = scene_truth_occ(other, AP, 100.2, STAMP, 0.5, True)
chk(occ is None and c == TRUTH_NOT_APPROVED, 'C3 名稱不在核准清單 ⇒ 不給真值')

mism, = parse_obstacles(['handle/bar:other_model:cylinder:0.005,0.2:0,0,0:'])
mism.T_world_link = np.eye(4)
occ, c = scene_truth_occ(mism, AP, 100.2, STAMP, 0.5, True)
chk(occ is None and c == TRUTH_MODEL_MISMATCH, 'C4 模型身分不符 ⇒ 不給真值')

nop, = parse_obstacles(['handle/bar:drawer_unit:cylinder:0.005,0.2:0,0,0:'])
chk(nop.T_world_link is None, 'C5a 動態物件在收到位姿前 T_world_link 為 None')
occ, c = scene_truth_occ(nop, AP, 100.2, STAMP, 0.5, True)
chk(occ is None and c == TRUTH_NO_POSE, 'C5b 沒有位姿 ⇒ 不給真值')

occ, c = scene_truth_occ(dyn, AP, 100.2, {}, 0.5, True)
chk(occ is None and c == TRUTH_NO_POSE, 'C6 沒有時戳 ⇒ 不給真值（不當成 0 秒前）')

occ, c = scene_truth_occ(dyn, AP, 100.6, STAMP, 0.5, True)
chk(occ is None and c == TRUTH_POSE_STALE, 'C7 位姿過期（0.6 > 0.5）⇒ 不給真值')

# 邊界：剛好等於 pose_timeout 不算過期（> 才算）
occ, c = scene_truth_occ(dyn, AP, 100.5, STAMP, 0.5, True)
chk(occ == 0.0 and c == TRUTH_GIVEN, 'C8 age 恰等於 pose_timeout ⇒ 仍給真值')

# ---------- D. 靜態物件不受時戳影響，但仍要有位姿 ----------
occ, c = scene_truth_occ(stat, AP, 1e9, {}, 0.5, True)
chk(occ == 0.0 and c == TRUTH_GIVEN,
    'D1 宣告為靜態者由設定給定世界位姿，不套 pose_timeout')
stat2, = parse_obstacles(['cabinet/top::box:0.6,0.45,0.02:0,1.45,0.89:'])
stat2.T_world_link = None
occ, c = scene_truth_occ(stat2, AP, 100.0, STAMP, 0.5, True)
chk(occ is None and c == TRUTH_NO_POSE, 'D2 靜態但位姿被清掉 ⇒ 仍然不給真值')

# ---------- E. 回傳碼都有說明 ----------
codes = [TRUTH_GIVEN, TRUTH_OFF, TRUTH_NOT_APPROVED, TRUTH_MODEL_MISMATCH,
         TRUTH_NO_POSE, TRUTH_POSE_STALE]
chk(len(set(codes)) == len(codes), 'E1 回傳碼互不相同')
chk(all(k in TRUTH_REASON for k in codes), 'E2 每個回傳碼都有文字說明')

# ---------- F. 真值**只**改遮蔽，不得改距離或狀態 ----------
import inspect
src = inspect.getsource(scene_truth_occ)
body = '\n'.join(l for l in src.split('\n') if not l.strip().startswith('#'))
chk('status' not in body and '.d' not in body.replace('T_world_link', ''),
    'F1 函式不碰 status 與距離：真值只回答遮蔽，不冒充距離資料')


# ---------- G. 啟動前提核對（與節點共用同一份） ----------
from ammr_wholebody_mpc.arm_link_distance import validate_scene_truth
OBS = parse_obstacles(['cabinet/top::box:0.6,0.45,0.02:0,1.45,0.89:',
                       'handle/bar:drawer_unit:cylinder:0.005,0.2:0,0,0:'])
GOOD = {'cabinet/top': '', 'handle/bar': 'drawer_unit'}

validate_scene_truth(OBS, GOOD, True, 'links')
chk(True, 'G1 名單涵蓋全部具名幾何且身分一致 ⇒ 通過')

validate_scene_truth(OBS, {}, False, 'points')
chk(True, 'G2 模式關閉時不核對（行為完全不變）')

for ap, geo, why in [
        ({}, 'links', '核准清單為空'),
        ({'cabinet/top': ''}, 'links', '少列了 handle/bar'),
        (dict(GOOD, **{'wall_12': ''}), 'links', '多列了設定裡沒有的障礙物'),
        ({'cabinet/top': '', 'handle/bar': 'other'}, 'links', '模型身分不符'),
        (GOOD, 'points', 'geometry 不是 links')]:
    try:
        validate_scene_truth(OBS, ap, True, geo)
        chk(False, f'G3 {why} 應拋出但沒有')
    except RuntimeError as e:
        chk(True, f'G3 {why} 正確拒絕：{str(e)[:38]}…')

# ---------- H. 節點的遮蔽解析真的呼叫了真值函式，且配對列有綁名稱 ----------
import re
nsrc = open('src/ammr_wholebody_mpc/ammr_wholebody_mpc/'
            'arm_link_distance.py').read()
code = '\n'.join(l for l in nsrc.split('\n') if not l.strip().startswith('#'))
chk('def _occ_resolve(' in code and 'scene_truth_occ(' in code,
    'H1 列建構走統一的 _occ_resolve，且它呼叫 scene_truth_occ')
chk(code.count('self._occluded(') == 2,
    f'H2 原本的遮蔽路徑仍存在（_rows_points 與 _occ_resolve 各一處），'
    f'回退不是改成放行：{code.count("self._occluded(")} 處')
chk('_occ_of_for(_obn)' in code,
    'H3 必要配對列把**該配對的障礙物名稱**綁進閉包，不是靠預設參數')
chk("'scene_truth_on'" in nsrc and "'n_truth'" in nsrc,
    'H4 診斷發布模式旗標與真值列數 ⇒ 趟次能標明使用場景真值')
# H5 用 AST 核，不用字串：三處 freespace_confirmed 都只在說明文字裡，
# 字串比對會把文件誤判成程式碼。要問的是「有沒有真的去設它」。
import ast as _ast
_tree = _ast.parse(nsrc)
_touch = []
for _nd in _ast.walk(_tree):
    if isinstance(_nd, _ast.Attribute) and _nd.attr == 'freespace_confirmed':
        _touch.append(f'屬性存取 行{_nd.lineno}')
    if isinstance(_nd, _ast.Constant) and _nd.value == 'freespace_confirmed':
        _touch.append(f'字串常值 行{_nd.lineno}')   # p('freespace_confirmed', …)
    if isinstance(_nd, _ast.Name) and _nd.id == 'freespace_confirmed':
        _touch.append(f'名稱 行{_nd.lineno}')
chk(not _touch, f'H5 距離節點沒有任何一處去設或讀 freespace_confirmed：{_touch}')

print(f'\n{sum(OK)}/{len(OK)} 通過')
sys.exit(0 if all(OK) else 1)
