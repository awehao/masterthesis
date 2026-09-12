# 凍結：固定底盤預抓取版本（2026-09-12）

達成條件：**227/227 軌跡設定點全部收到並套用**，未放寬任何停止門檻，
終點位置與工具軸方向達標，時間與追蹤皆可核對。基準趟 `manip_121910`。

## 版本（sha256 前 16 碼）

| 檔案 | sha |
|---|---|
| `evaluation/isaac_manip_sim.py` | `624162138b054a36` |
| `evaluation/play_arm_traj.py` | `e1c1f322491d7e25` |
| `evaluation/arm_traj.py` | `8d5d85b8d1f4704d` |
| `evaluation/models/omni_bot_manip.urdf` | `fd3252f41bd3d6ef` |
| `src/my_omnibot_description/config/manipulation_cases.yaml` | `abc7edf943c8a2fd` |
| `src/my_omnibot_description/config/arm_initial_pose.yaml` | `685ee536c51f898d` |
| `evaluation/check_arm_path.py` | `4f22c591b8e3b7ef` |
| `evaluation/obstacle_geometry.py` | `afbd1001adfb4885` |

## 設定

```
physics_dt      0.01 s          rendering_dt 同值
rtf 節流        1.0（每步 sleep 到 10 ms，同時讓出 GIL）
命令介面        /arm/joint_position_cmd  Float64MultiArray [seq, kind, t_sched, q1..q6]
命令排程        模擬時間，50 Hz
軌跡限制        v ≤ 0.35 rad/s，a ≤ 0.7 rad/s²
關節增益        kp 1e5，kd 1e4
停止門檻        追蹤誤差 0.20 rad、關節限位餘裕 0.02 rad、底盤位移 0.010 m
QoS             RELIABLE / VOLATILE / KEEP_LAST / depth 10
```

## 基準結果（`manip_121910`）

| 項目 | 值 |
|---|---|
| 軌跡點 發布 → 進回呼 → 曾被套用 | 227 → 227 → **227** |
| 收到未套用 | **0** |
| 接收間隔 中位／最大 | 0.0200 / **0.0200 s** |
| 軌跡套用區間 | **4.530 s**（標稱 4.52） |
| 到位時間（③ − ①） | **4.440 s**，容差 ±0.50 s |
| TCP 位置誤差 | **0.0037 m**（容差 0.0050） |
| **工具軸方向**誤差 | **0.131°**（容差 2.0°） |
| 追蹤誤差 (a) 對已套用設定點 | p50 31.71 / max **35.04 mrad** |
| 追蹤誤差 (b) 對 `q_ref(t)` | p50 **42.04 mrad** |
| 實際路徑 最小自碰／環境餘裕 | 8.72 mm / 242.57 mm |

**停止原因為 `sim_limit`，不是完成訊號** —— 完整軌跡已執行、離線確認達標，
程序最後由模擬時限結束。

## 界線

* 「工具軸方向」不等於完整三維姿態；繞工具軸的旋轉未受判準約束。
* 幾何為離散取樣（1952 個姿態），非連續無碰撞證明；未讀模擬器接觸事件。
* 35 mrad 的持續落後**成因未診斷**，不可當成已證明的位置驅動特性。
* 節流同時把 RTF 由約 1.5 降到 1.0，**該因素與 executor 排程未分離**；
  不宣稱 executor 是先前遺失的根因。
* 底盤由初始化直接放置並每步歸零，**不是導航停放或 guard 自然保持的證據**。
