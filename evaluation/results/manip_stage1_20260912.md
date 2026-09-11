# 操作階段第一步：模型統一與手臂位置命令入口（2026-09-12）

範圍：**固定底盤**的手臂短測試。不動導航（`isaac_bigarena_sim.py` 與導航跑批
完全未改，實驗維持凍結），不改預抓取姿態，不開導航精調，不加抓取接觸，
不啟動會另發底盤命令的 `arm_vel_gate`。

## 1. 已建立且驗證通過

### 模型統一 —— **執行期已確認**

`/tmp/omni_bot_wb.urdf`（Isaac 載入的那份）在 repo 裡沒有產生器。用**逐字元
比對**還原出它的展開參數：

    use_arm:=true add_gripper:=true add_arm_camera:=true use_camera:=true

（去註解、正規化空白後與該檔完全相同。）同一時間
`omni_bot_dynamic.launch.py` 給 `robot_state_publisher` 的是
`use_camera:=false use_arm:=true`，`add_gripper` 取預設 `false` ——
**兩邊差兩個參數**，不只夾爪，底盤相機也不在 TF 的模型裡。

現由 `evaluation/expand_wholebody_urdf.sh` 產生單一份
`evaluation/models/omni_bot_manip.urdf`（檔頭記參數與來源 sha）。

離線差異（`check_model_consistency.py`）：

| | 操作模型 | 導航 TF 模型 |
|---|---|---|
| 非固定關節 | 8（含 `finger_joint1/2`） | 6 |
| link | 126 | 112 |
| `link_tcp` | link6 → `joint_eef` → `gripper_fix` → `joint_tcp`，**純平移 +0.08360 m** | **不存在** |
| 相機框架 | 13 | 3 |

**執行期確認**（`check_model_runtime.py`，取回 `robot_state_publisher` 實際的
`robot_description` 參數再比對結構）：

```
非固定關節  檔案 8  執行中 8
link 數     檔案 126  執行中 126
檔案的 link_tcp：link6 → joint_eef → gripper_fix → joint_tcp   累計平移 0.08360 m
執行中的 link_tcp：link6 → joint_eef → gripper_fix → joint_tcp   累計平移 0.08360 m
--- 結果：執行中的模型一致 ---
```

`base_footprint → link_tcp` 的 TF 查詢在執行中成功。
**到這裡才能說「執行中的模型已統一」**，離線檔案相同不足以支持這句話。

附帶發現：Isaac **本來就發布全部 8 個關節**（含手指），是 TF 端模型沒有它們
才被丟掉；Isaac 影像用的 `base_camera_color_optical_frame` 也只存在於操作模型。

### 端到端幾何交叉驗證

Isaac 初始化後底盤在 **(5.0030, 6.6710)、yaw 90.000°**（案例要求 5.003 / 6.671 / 90°）。
`test_start` 的 TCP 在底盤座標為 (0.197, 0, 0.400)（FK 算得），yaw 90° 下
手算世界座標應為 (5.003, 6.868, 0.400)；Isaac 由 `link_tcp` prim 直接讀出
**(5.0032, 6.8677, 0.3999)** —— **吻合到 0.3 mm**。
FK、停放位姿與 TCP 框架三者一致。

底盤在 60 s 模擬時間內位移 **0.219 mm**（容許 10 mm），確認固定底盤成立。

### 獨立操作案例

`src/my_omnibot_description/config/manipulation_cases.yaml`，**與
`bigarena_poses.csv` 無關**。把 18 個 `known_obs` × 4 個面全部算過再選：
`known_obs_12` 南面，停放 (5.003, 6.671) / yaw 90°，
**淨距 = 底盤圓盤表面到箱體表面**（底盤中心到箱面 0.689 m 減機器人半徑 0.30 m）
對目標箱 **0.390 m**、對其他靜態箱 **2.328 m**。

TCP 目標以**底盤座標**定義 (0.450, 0, 0.550)、工具朝 +x；
`pregrasp_reference` 的關節角可用，是因為它的 FK 結果與本案例的底盤相對目標
相同 —— **核對過的等價，不是照搬世界座標**。

### 沿途碰撞檢查

`check_arm_path.py` 檢查的是**實際要執行的那條**梯形速度軌跡
（227 點、4.52 s、v ≤ 0.35 rad/s、a ≤ 0.7 rad/s²、50 Hz），
產生器 `evaluation/arm_traj.py` 由**檢查端與播放端共用 import**，不各自實作。

* 沿途最小自碰餘裕 **8.71 mm**（link4/link6，第 211/226 點），全程 8.7–13.3 mm
* 沿途最小環境餘裕 **242.49 mm**（`uflite_finger1` 對 `known_obs_12`）——
  與案例設定的站距 0.24 m 相符，是幾何的交叉驗證
* 幾何與算法：URDF `<collision>` 的 mesh 頂點與基本形取樣點雲，每 link ≤900 點，
  以控制器自己的 FK 轉到世界座標後做 KD-tree 最近鄰
* **227 個離散取樣點，不是連續碰撞證明**；取樣間隔 20.0 ms

**門檻更正**：先前報告寫的「10 mm 警告線」是我在新腳本裡自設的預設值，
不是專案標準。專案既有的兩個數字是 `verify_self_collision.py` 的
`MARGIN_WARN = 0.020`（**20 mm** 舒適線）與 §10.9 記載的 **0.005 m**
規劃器自碰拒絕門檻。8.71 mm **通過 5 mm 門檻、低於 20 mm 舒適線**。
兩者都不是「軌跡驗收」的正式門檻；本檔只如實標示落點。

## 2. 尚未完成：軌跡沒有實際播出去

手臂命令入口 `/arm/joint_position_cmd`（`Float64MultiArray`, 6）已實作於
`evaluation/isaac_manip_sim.py`，Isaac 端 `收到關節命令 0 則`——
播放端在等訂閱者時逾時，所以**手臂仍停在 `test_start`，沒有運動**。
`arm_final_err_max = 1408.602 mrad` 就是「完全沒動」（等於 joint5 的整段行程）。

### 已排除的原因

* Isaac 有進主迴圈、有建立訂閱（log 明確印出「進入主迴圈；等待
  /arm/joint_position_cmd」）
* 把節點改到背景執行緒用 `SingleThreadedExecutor`（與 `isaac_bigarena_sim.py`
  相同寫法）**沒有改變症狀**，所以不是 spin 方式
* Isaac 的端點確實在線上：某一趟 `endpoint_check` 解析出
  `/clock`、`/joint_states`、`/manip/tcp_pose` 的發布者就是 `/isaac_manip_sim`

### 已定位的現象（尚未修好）

節點名在圖上的解析**有競賽**，且方向會變：

* 第 3–5 趟：Isaac 顯示為 `_NODE_NAME_UNKNOWN_`，`robot_state_publisher` 與
  guard 正常
* 第 6 趟：**反過來** —— `/isaac_manip_sim` 正常，換成 guard 是
  `_NODE_NAME_UNKNOWN_`，導致「`/cmd_vel` 唯一發布者」誤判為不符而中止

`ros2` CLI（走 ROS 2 daemon）在同 domain 下 `topic list` 只看到
`/parameter_events` 與 `/rosout`、`node list` 為空，而同一趟的 rclpy 腳本
看得到圖 —— 兩者不一致。

**下一步要修的是這個，不是手臂控制本身**：`endpoint_check.py` 需要先等節點名
解析完成再判定，否則會產生偽失敗；播放端的放行條件也不該只靠圖查詢
（已改為「等一段時間後照發，由接收端計數判定」，但該版尚未跑到播放階段）。

## 3. 本輪未觸碰

導航鏈與 `isaac_bigarena_sim.py`、`pregrasp_reference` 的關節角、
導航到達容差、`arm_vel_gate`、抓取接觸、相機感知。

## 4. 待定

* **8.71 mm 的自碰餘裕**：通過 5 mm 規劃門檻、低於 20 mm 舒適線。
  依裁決先保留姿態在模擬器中驗證，不因低於舒適線就改姿態，
  也不因為正值就宣稱安全。
* **停放容差 ±0.05 m / ±3°**：案例檔中標為**操作端需求**，
  並註明現有導航到達容差 0.30 m（實測停在 0.29–0.31 m）**達不到**，
  需要另一段停放對準行為。尚未確認在該誤差範圍內手臂沿途仍可行、
  TCP 相對箱體仍合格 —— 這要等能實際動起來之後才有意義。
* 操作完成判準應對照**箱體／世界座標**的預抓取目標，不能只看 TCP 是否回到
  指定的底盤相對位姿。目前 `manip_run.json` 已記 TCP 世界座標，
  但尚未寫入對箱體的判準。
