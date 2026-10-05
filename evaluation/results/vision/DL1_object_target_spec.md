# DL1：物體局部座標目標生成——規格（draft-1，送審；核可前不實作）

2026-10-06。依第二階段計畫 §9.13.3 與 Codex 裁定（DL0 收尾審查）。**這一步不用 DL**，只把目標生成從世界軸假設改成物體座標；
不改控制限制、不發布控制命令、不接節點、不宣稱物理可達或視覺操作完成。

## 0　現況核對（要移除的隱藏假設）

`evaluation/drawer_task_node.py::target_at(standoff, H)`：

```
p = 把手位姿 H 的平移（實測或凍結）
T[:3,:3] = R_grasp                         # 設計抓取姿態在停車位姿的 FK（世界旋轉，固定）
T[:3, 3] = (p_x, p_y − (tcp_offset + standoff), p_z)   # 沿世界 −y 退讓
```

隱藏假設：(1) 接近／退讓軸＝世界 −y；(2) 抓取旋轉＝固定世界旋轉，**把手位姿 H 的旋轉完全未使用**；(3) 抓取深度 `tcp_offset`（實測 12.26 mm）沿世界 y。
`drawer_target.py::DrawerTarget`（夾持後）已用 `G_T_H` 相對關係，但滑軌方向仍是世界向量 `axis_world=(0,−1,0)`。

## 1　介面（新檔 `evaluation/object_target_geometry.py`，純函式）

依計畫 §4.4：$T_{WE}T_{EG}=T_{WH}T_{HG}$ ⇒ 工具目標 $T_{WE}=T_{WH}\,T_{HG}\,T_{EG}^{-1}$。接觸前位姿在**把手座標**沿接近軸退讓：

$$T_{WE}^{pre}(s)=T_{WH}\;\mathrm{Trans}(s\,\hat a_H)\;T_{HG}\;T_{EG}^{-1}$$

| 輸入 | 意義 | 驗證 |
|---|---|---|
| `T_WH` (4×4) | 把手座標在世界的位姿（實測、凍結或之後的視覺估計） | 剛體（det = +1、正交、有限） |
| `T_HG` (4×4) | 把手座標 → 抓取座標（含抓取深度與抓取旋轉） | 剛體；平移範數上限（防 mm 當 m） |
| `T_EG` (4×4) | 末端工具座標 → 抓取座標（目前 TCP 即抓取座標 ⇒ I） | 剛體 |
| `a_H` (3) | 接近／退讓軸，**在把手座標**表示（單位向量；指向離開物體的一側） | 有限、非零、正規化 |
| `s` (m) | 退讓距離（≥ 0） | 有限、0 ≤ s ≤ s_max（預設 0.30 m，防 mm 誤用） |
| `mech` | 機構模型：滑軌軸 `u_H`（把手座標）或鉸鏈（軸與軸上一點，皆在物體座標） | 有限、單位向量 |

輸出：`T_WE` 與診斷 dict（所用參數、各輸入來源標籤、拒絕原因）。**函式內不出現任何世界軸常數或固定旋轉。**

另提供：
- `handle_pose_from_partial(center_W, axis_W, normal_prior_W | None)`：視覺只給中心＋軸線時，第三軸（繞橫桿軸的滾轉）**不可觀測**；
  有明示先驗（例如已知櫃體前板法向）才組出 `T_WH` 並標 `prior_used`，否則回傳拒絕 `underdetermined_roll`。橫桿軸正負號不定 ⇒ 以先驗選號並標示。
- `opened_handle_pose(T_WH0, mech, q)`：機構開度 q 下的把手位姿（滑軌 $T_{WH0}\mathrm{Trans}(q\,u_H)$；鉸鏈為繞軸旋轉）——物體與其機構一起變換，不只旋轉 TCP 目標。
- `symmetric_equivalents(T_HG, sym)`：對稱把手的等價抓取（例如繞橫桿軸 180°）；選擇規則明示（取與參考姿態旋轉距離最小者），不默默換抓取。

## 2　基準相容（不換既有接觸幾何）

既有配置：櫃體未旋轉、`R_WH = I`。取
`T_HG = [R_grasp | (0, −tcp_offset, 0)]`、`a_H = (0, −1, 0)`、`T_EG = I` ⇒ $T_{WE}^{pre}(s)$ 與 `target_at(s)` **逐元素一致**（容差 1e-12）。
原成功配置與抓取深度 12.26 mm 保留為基線；本包不改 `drawer_task_node.py`（接線另立、另送審）。

## 3　離線測試（`test_object_target_geometry.py`，不跑模擬）

1. 基準相容：多組 s 與既有 `target_at` 一致；另以既有實錄（例如 phf／d1s4b 的把手位姿與任務參數）重算接觸前目標，與紀錄中的目標比對。
2. 等變性：任意剛體 G 作用於 `T_WH`（及機構）⇒ `T_WE` 變為 `G·T_WE`；平移、繞 z、任意 SO(3) 各測。
3. 往返：由 `T_WE`、`T_HG`、`T_EG`、`a_H`、`s` 反推 `T_WH` 回到原值。
4. 退讓軸符號：s 增加時 TCP 沿 `R_WH a_H` 遠離把手；`a_H` 反號被偵測（與機構開啟方向不一致時拒絕或警示，依規格選拒絕）。
5. SO(3) 合法性：非正交、det = −1、NaN／inf 輸入一律拒絕。
6. 單位誤用：平移或 s 以 mm 數值輸入（例如 12.26、300）被上限攔下。
7. 部分觀測：只有中心＋軸線而無先驗 ⇒ `underdetermined_roll` 拒絕；有先驗 ⇒ 組出合法 `T_WH` 並標 `prior_used`。
8. 對稱等價：繞橫桿軸 180° 的等價抓取被列出，選擇規則可重現。
9. 機構：滑軌開度 q 下把手沿 `R_WH u_H` 平移；物體整體旋轉後滑軌方向一起旋轉。
10. 物體身分跳變：輸入的物體 id 與鎖定抓取時不同 ⇒ 拒絕沿用鎖定的抓取關係。

## 4　不做／不宣稱

不改 `drawer_task_node.py`、控制器、限制或接觸幾何；不發布 `/drawer/tcp_target` 或任何命令；不宣稱旋轉後櫃體的物理可達、碰撞安全或夾爪相容性
（計畫 §9.13.3 第 3 點的幾何契約——夾爪閉合指距、指墊空隙、碰撞模型——列為後續子包，本包只留介面欄位與明示「未核對」）；不宣稱視覺操作。

## 5　待審

1. 介面與公式（退讓在把手座標、`T_EG = I` 的現況假設）是否足夠。
2. 部分觀測的處理（無先驗即拒絕、軸號由先驗選定並標示）。
3. 基準相容的核對方式（逐元素一致＋既有實錄目標比對）。
4. 幾何契約（夾爪相容、碰撞模型一起轉換）是否必須納入本包，或可列後續子包。

---

# draft-2（依 Codex DL1 規格審查四項修正；修正後實作純函式與離線測試，不接節點、不開模擬）

### 1　基準抓取深度以實錄參數為準
- 實錄 `task.json → args.grasp_depth_m`：`phf_01_M`、`ph21_func_H`、`d1s4b_M` 為 **0.0068 m**；早期 `motm_200_075305` 為 **0.01226 m**；四趟 `tcp_offset_m = −grasp_depth_m`。
  `drawer_task_node.py` 的來源說明字串仍寫 12.26 mm，**不據以判定實際配置**。兩種配置都保留並分別核對。
- 由 `target_at`：TCP y＝p_y − (tcp_offset + s)＝p_y + grasp_depth − s ⇒ 基準 **`T_HG = [R_grasp | (0, +grasp_depth, 0)]`**、退讓沿把手局部 **−y**（`a_H = (0, −1, 0)`）。
- 容差：純函式相容 1e-12；實錄比對用求解節點紀錄的每輪目標 `solve_in.T_cyc`（float64），在 ENGAGE_WAIT（`target_at(0, H_frozen)`、開度 0）核對
  旋轉＝停車位姿＋`q_grasp` 的 FK、平移＝凍結把手位姿＋(0, +grasp_depth, 0)；容差依紀錄精度另訂並寫明（把手位姿以 USD 真值、抽屜關閉時的 prim 位姿重建）。

### 2　座標與部分觀測契約
- 記號：`T_AB` 把 **B 座標表示的點轉到 A**（p_A = T_AB · p_B）。剛體驗證：4×4、有限、底列 `[0,0,0,1]`、旋轉 det = +1 且正交。
- `handle_pose_from_partial(center_W, axis_W, normal_W, axis_ref_W, center_is_center)`：
  - 前板法向只補**繞橫桿軸的滾轉**，**不決定橫桿軸正負號**；必須另給明示的軸向參考 `axis_ref_W`（或參考姿態）。
  - 軸號由 `axis_ref_W` 決定；`|axis·axis_ref| < cos(60°)` 視為無法消除歧義 ⇒ 拒絕 `axis_sign_ambiguous`（或在 `return_candidates=True` 時輸出兩個等價候選，不默默選號）。
  - 法向與橫桿軸近平行（`|axis·normal| > cos(15°)`）⇒ 拒絕 `normal_parallel_axis`。
  - `center_is_center=False`（只是軸線上的一點）⇒ 拒絕 `point_not_center`，不冒充把手中心。
- `T_EG = I` 是本次定義；測試另以至少一個**非單位**工具轉換核對公式。

### 3　機構與退讓方向
- 滑軌：$T_{WH}(q)=T_{WH0}\,\mathrm{Trans}(q\,u_H)$；`T_WH0`＝**零開度**（抽屜關閉）時的把手位姿；q 單位公尺、合法範圍 `[q_min, q_max]`（由機構模型給定，越界拒絕）。
- **不把「退讓軸＝開啟方向」當通用規則**。反號核對依明示的**外側方向契約** `outward_H`（物體外側、接近側在把手座標的方向）：
  `a_H · outward_H ≤ 0` ⇒ 拒絕 `retreat_not_outward`；未提供 `outward_H` ⇒ 不判定（診斷標 `retreat_side_unchecked`），函式不自行推斷是否遠離物體。
- 鉸鏈列後續子包（本包不實作）。

### 4　身分與對稱候選
- `object_target(..., object_id, locked_object_id=None)`：`locked_object_id` 給定且 ≠ `object_id` ⇒ 拒絕 `object_identity_changed`（薄契約，不另建相位機）。
- `symmetric_equivalents(T_HG, sym_axis_H, order)`：依給定對稱模型產生**幾何等價候選**（不是已驗證可抓取）；
  選擇＝與參考旋轉的測地距離最小，**同分（差 ≤ 1e-9 rad）取候選索引最小者**（固定、可重現）。

### 範圍
夾爪相容與碰撞模型一起轉換留後續子包；輸出一律標 `geometry_contract: 'unchecked'`，不能據此放行控制或宣稱可達。

---

# 實作審查 r1 → r2 修正（Codex 四項必修；規格增訂）

1. **把手座標唯一約定**＝資產 prim 座標：x 沿橫桿、y 指向物體內側（−外側法向）、z = x×y。`handle_pose_from_partial` 的 `normal_W` 為**外側**法向，組姿採同一約定（bar26 未旋轉 ⇒ R_WH = I）。
   若上游使用其他把手座標 H'：`reparam_grasp(T_HG, a_H, C)`，C = T_{H H'}，T_{H'G} = C⁻¹T_{HG}、a_{H'} = R_Cᵀ a_H；不得直接共用參數（混用反例：46.4 mm）。
2. **嚴格容差**：剛體底列純絕對容差 1e-12；測試的 ≤1e-12 宣稱以最大絕對誤差判定。
3. **拒絕介面**：所有公開函式回 dict（ok／why），不拋原生例外；中心、候選、參考旋轉皆驗證。`symmetric_equivalents` 改回 dict（`candidates`）；`choose_candidate` 改回 dict（`index`、`distances`）。
4. **同分規則**：先算全體最小 d_min，取 d ≤ d_min + 1e-9 rad 的第一個候選。
