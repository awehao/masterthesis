# door_k3a_1 判定結果分窗註記（Codex 20261021 裁定；door_k3a_check.json 原檔保留不改）

`door_k3a_check.json` 的下列兩個旗標來自**中止後尾段**（ABORTED，22.04 s 起：保留夾持、門續轉到 1.70°）的相位歸類，
不能解讀為 OPEN 前已有這些違規：
- `door_angle_pre_open_over_1deg`（1.70°）：OPEN 前（≤ 20.22 s）門角實際 ≤ 0.022°。
- `finger_bar_outside_allowed_phase`（10.3 N）：中止前手指—橫桿接觸只出現在允許相位（ENGAGE～OPEN）。

本趟封存定位：**K3-A 執行後因單指接觸不足而中止，功能確認未通過**（OPEN 起 1.8 s；finger1 於門 ≈ 0.57° 低於 0.5 N；中止時門 ≈ 0.74°，q_ref 1.19°，未達 8°）。
中止後門續轉到 1.70°，不算受控開門達成量；零速率命令不等於物體瞬間停止。
