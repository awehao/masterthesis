"""抽屜場景 → 距離節點的參數字串。**幾何只有一份來源：資產檔。**

為什麼要產生器，而不是把 14 行設定字串抄進趟次腳本
----------------------------------------------------
抄過去就有了第二份幾何。資產檔改一個尺寸、趟次腳本不會跟著改，而距離節點
用的是趟次腳本那一份 —— 屏障會以錯誤的幾何計算，而且不會有任何錯誤訊息。

抽屜與把手**不得宣告為靜態**：它有被動滑動自由度，位姿要由模型話題提供。
宣告靜態等於宣稱它不會動，而整個任務的目的就是要讓它動。

用法：
    python3 evaluation/gen_drawer_obstacle_params.py            # 人看的
    python3 evaluation/gen_drawer_obstacle_params.py --ros-args # 可直接貼進命令列
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
import drawer_asset as DA

ap = argparse.ArgumentParser()
ap.add_argument('--asset', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--pose', default='0.0,1.45', help='櫃體底面中心 x,y')
ap.add_argument('--opening', type=float, default=0.0, help='開度 m')
ap.add_argument('--model', default='drawer_unit',
                help='抽屜與把手的模型名稱（位姿話題 /model/<名稱>/pose）')
# **沿用專案既有的凍結設定**（run_coman_drawer20.sh 的 PAIR_ROWS），不自創一組。
# 為什麼只有遠端六連桿：強制列的作用是在「該連桿的最近障礙物不是最危險的那個」
# 時補上列 —— 那主要發生在接觸例外把最近列濾掉的時候。近端連桿與底盤仍有
# 各自的最近障礙物帶選列，不會沒有列可約束。
# 離線核算也支持這個選擇：把強制列擴到全部連桿 × 全部 14 個障礙物
#（每週期 1938–2853 列，對照帶選列的 170–201 列）後，**最小餘量、綁定列與
# 安全層修改週期數完全相同**（見 wgmpc_wg2_pregrasp_barrier.yaml）。
ap.add_argument('--links',
                default='link4,link5,link6,uflite_finger1,uflite_finger2,'
                        'uflite_gripper_link',
                help='要為**全部具名障礙物**補強制列的連桿')
ap.add_argument('--ros-args', action='store_true', help='輸出可貼的 --ros-args 片段')
# **逐引數輸出**（每行一個）。供 shell 以 `mapfile -t A < <(... --argv)` 讀成陣列，
# 再以 "${A[@]}" 傳給 ros2。
# 為什麼不用 --ros-args 的文字再 eval：ros2 的參數陣列需要**引號留在詞元裡**
# （YAML 解析），而 eval 會把引號吃掉；續行符也會讓後續的 `-p` 帶上前導空白、
# 變成一個畸形詞元。實測 `eval spawn ... $DIST_ARGS` 正是如此。
ap.add_argument('--argv', action='store_true',
                help='每行一個引數，供 mapfile 讀成 bash 陣列')
a = ap.parse_args()

spec = DA.load(a.asset)
pose = tuple(float(v) for v in a.pose.split(','))
shapes = DA.shapes_world(spec, pose, a.opening)

obstacles, approved = [], []
for item in shapes:
    nm = item[0]
    # 櫃體是靜態（model 留空 ⇒ xyz/rpy 即世界位姿）；抽屜與把手是動態
    md = '' if nm.startswith('cabinet/') else a.model
    if item[1] == 'box':
        c, s = item[2], item[3]
        obstacles.append(f'{nm}:{md}:box:{s[0]},{s[1]},{s[2]}:'
                         f'{c[0]},{c[1]},{c[2]}:')
    else:
        c, r, L = item[2], item[3], item[4]
        # 生產端圓柱沿**本地 z**，把手沿 x ⇒ rpy = (0, pi/2, 0) 把 z 轉到 x
        obstacles.append(f'{nm}:{md}:cylinder:{r},{L}:'
                         f'{c[0]},{c[1]},{c[2]}:0,{np.pi/2},0')
    approved.append(f'{nm}:{md}')

# **必要配對列**：補齊「每個連桿對全部具名障礙物」。只留最近障礙物時，
# 較遠但較危險的物件可能完全沒有列可約束。
pair_rows = [f'{lk}:*' for lk in a.links.split(',') if lk.strip()]


def ros_list(xs):
    return '[' + ','.join(f'"{x}"' for x in xs) + ']'


ARGS = [
    '-p', f'obstacles:={ros_list(obstacles)}',
    '-p', f'pair_rows:={ros_list(pair_rows)}',
    '-p', 'scene_truth:=true',
    '-p', f'scene_truth_approved:={ros_list(approved)}',
]

if a.argv:
    for x in ARGS:
        print(x)
elif a.ros_args:
    for i in range(0, len(ARGS), 2):
        cont = ' \\' if i + 2 < len(ARGS) else ''
        print(f'  {ARGS[i]} {ARGS[i+1]}{cont}')
else:
    print(f'資產 {os.path.relpath(a.asset, WS)}  位姿 {pose}  開度 {a.opening}')
    print(f'\nobstacles（{len(obstacles)} 個具名幾何，'
          f'靜態 {sum(1 for x in approved if x.endswith(":"))}、'
          f'動態 {sum(1 for x in approved if not x.endswith(":"))}）：')
    for x in obstacles:
        print(f'  {x}')
    print(f'\nscene_truth_approved（{len(approved)}）：')
    for x in approved:
        print(f'  {x}')
    print(f'\npair_rows（{len(pair_rows)}）：')
    for x in pair_rows:
        print(f'  {x}')
    print('\n**安全層不得設 freespace_confirmed:=true** —— 場景裡有櫃體與抽屜。')
