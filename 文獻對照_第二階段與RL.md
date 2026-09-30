# 文獻對照:第二階段(手臂/全身)與 RL 殘差學習

> 從 `文獻對照.md` 分出。**這些不屬於第一階段最終版**:手臂與全身是第二階段,殘差學習是第四次進度的 RL 工作,主實驗 `chain_three100` 並未使用。保留待各自階段需要時再用。
>
> ⚠️ **兩個使用前提**
>
> 1. **內部的 §1 / §4 / §12 / §19 / §22 等交叉引用指向舊版 `文獻對照.md` 的編號**,
>    該檔已重寫為第一階段專用並重新編號,對應關係已不成立。讀到交叉引用時請依上下文
>    判斷,不要直接跳去現在的同號小節。本檔自身的 §18–§23 編號亦是舊編號,僅為保留
>    原樣未改。
> 2. **本檔的文獻與主張尚未按第一階段那輪的收準標準逐條複查**。搬進正式論文前,
>    至少要重查:是否把參數對照寫成方法等價、是否借用了機制不同的方法名稱、是否把
>    工程防護寫成已證明的保證。判準見 `文獻對照.md` 的「驗證標記」與「行動清單 E」。

---

# 第三部分:手臂與全身(第二階段)

## 18. 全身運動學與雅可比

**程式碼**:[wholebody_kinematics.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/wholebody_kinematics.py)

```
q = [x, y, θ, j1..j6] ∈ SE(2) × R⁶          9 個廣義座標
鏈:world → base_x → base_y → base_theta → base_link → link_base → joint1..6 → link_eef → gripper → link_tcp
T = fk(q, 'link_tcp')  ∈ R^{4×4}
J = jacobian(q, 'link_tcp') ∈ R^{6×9}        一個雅可比同時涵蓋底盤與手臂
```
底盤三個平面自由度**當成同一運動鏈上的關節**,不是浮動基座。

**文獻**
- ○ Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation", *IEEE J. Robotics and Automation* **3**(1):43–53, 1987 — 任務空間控制原始
- ○ Siciliano & Slotine, "A general framework for managing multiple tasks in highly redundant robotic systems", ICAR 1991 — 冗餘度解析、任務優先序
- ○ Bayle, Fourquet, Renaud, "Manipulability of wheeled mobile manipulators: Application to motion generation", *IJRR* **22**(7-8):565–581, 2003 — **輪式行動機械臂把底盤與手臂合成單一雅可比的經典。你這節的直接前作**
- ○ Sentis & Khatib, "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives", *Int. J. Humanoid Robotics* 2(4), 2005
- ✓ [An Efficient Representation of Whole-body MPC for Online Compliant Dual-arm Mobile Manipulation](https://arxiv.org/abs/2410.22910), 2024 — 近期 whole-body MPC 於行動機械臂
- ✓ [Safe Expeditious Whole-Body Control of Mobile Manipulators for Collision Avoidance](https://arxiv.org/abs/2409.14775), 2024 — **低維底盤 + 高維手臂的反應式避障,與你 Phase 2 目標高度重疊,必讀**
- ○ Siciliano, Sciavicco, Villani, Oriolo, *Robotics: Modelling, Planning and Control*, Springer, 2009, ch. 3 — 幾何雅可比標準推導

**差異**:合成雅可比是 Bayle 2003 就有的,**不要當貢獻**。你的特點是工程嚴謹性:
- **URDF 是唯一幾何來源**,不硬編任何 link offset → 改掛載點或夾具不會讓控制器規劃一台已不存在的機器人
- 有 `verify_wholebody_kinematics.py` 與視覺化庫交叉驗證(「控制器需要自己的運動學,而且必須證明兩者一致」)

這是 methodology 的 validation 小節材料,不是新方法。

---

## 19. 連桿級速度安全濾波 ⚠️ 必引文獻

**程式碼**:[wholebody_safety_filter.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/wholebody_safety_filter.py)、[arm_link_distance.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_link_distance.py)

```
n_iᵀ J_{p_i}(q) v ≤ α_i (d_i − d_stop,i)        每個偵測點一列
加上關節速度 box 與關節位置限制
```

**文獻 —— 這個不等式有明確出處,程式碼裡目前沒引**
- ✓ **Faverjon & Tournassoud, "A local based approach for path planning of manipulators with a high number of degrees of freedom", ICRA 1987, pp. 1152–1159** — **「velocity damper」的原始出處**。論文把避碰「透過 velocity damper 與 tangent separating planes 的方法,轉譯成非常簡單的線性約束」,並明確主張「把任務的描述與避碰的約束分開」。原始形式:
  ```
  ḋ ≥ −ξ (d − d_s)/(d_i − d_s)     當 d ≤ d_i
  ```
  其中 `ḋ = nᵀ J q̇`。**你的式子是它的直接變體**(把 `d_stop` 放在右手邊、`α_i` 當增益)
- ✓ **Marinho, Adorno, Harada, Mitsuishi 等, ["Dynamic Active Constraints for Surgical Robots Using Vector Field Inequalities"](https://arxiv.org/abs/1906.07322) / "Whole-Body Control with (Self) Collision Avoidance using Vector Field Inequalities"** — **VFI**,`nᵀ J v ≤ α d` 的現代形式與完整推導(含自碰撞、多幾何基元)。**這是你這層最該引的一篇**
- ○ Kanoun, Lamiraux, Wieber, "Kinematic control of redundant manipulators: Generalizing the task-priority framework to inequality task", *IEEE T-RO* **27**(4):785–792, 2011 — 不等式任務納入優先序框架
- ✓ [Safe Expeditious Whole-Body Control…](https://arxiv.org/abs/2409.14775)(§18)— 多 primitive shape 的安全集
- ○ Schulman, Duan, Ho, Lee, Awwal, Bradlow, Pan, Patil, Goldberg, Abbeel, "Motion planning with sequential convex optimization and convex collision checking", *IJRR* **33**(9):1251–1270, 2014 — 凸碰撞檢查、capsule/swept volume 距離
- ○ Flacco, Kröger, De Luca, Khatib, "A depth space approach to human-robot collision avoidance", ICRA 2012 — 由感測距離直接構造速度約束

> **務必引用,不要重新命名。** 這個不等式已有 40 年歷史(Faverjon-Tournassoud 1987)與現代形式(VFI)。自己命名會被認為不熟文獻。

**你的加值(文獻很少寫的部分)—— 退化處理**

| 狀態 | 你的處理 | 論證 |
|---|---|---|
| `OK` | 直接用 `d_i` | — |
| `STALE` | 按障礙物在該時間內**可能移動的距離**縮減 `d_i`,並對整個命令限速 | 「忽略 stale 列而繼續接近,是唯一絕對不能發生的行為」 |
| `NODATA` | 沒有距離,無法約束任何東西 → 只能全域限速 | — |
| `occluded=1` | **保留該列不變**,但另加一個更緊的、僅針對該方向的上限 | 「有場景模型時距離不依賴視線,所以距離仍有效。丟掉該列是丟掉一個好的約束;把該方向視為淨空是發明從未量到的餘裕」 |

這段 docstring 的推理品質很高,可以直接擴寫成論文小節。文獻(VFI、Faverjon)假設距離量測總是可得,**沒有處理 stale/occluded/nodata**。這是你可主張的貢獻:**把 velocity damper 從「假設完美感知」擴充到「感知會退化」的情形,並對每種退化給出有論證的行為**。

補引建議:○ Lasota, Fong, Shah, "A survey of methods for safe human-robot interaction", *Foundations and Trends in Robotics*, 2017 — 安全監督與退化模式;○ ISO/TS 15066 speed-and-separation monitoring — 工業標準裡的 stale 資料處理。

**`d_stop` 的非線性問題**:`d_stop` 透過煞停項依賴命令速度 `v`,使約束對 `v` 非線性。你的處理與 Phase 1 一致 —— **從輸入命令算一次然後固定**,每週期解線性可行性問題,**殘差事後量測而非假設消失**。這個做法要寫,並引 ○ Diehl, Bock, Schlöder 的 real-time iteration 概念作為類比(一次線性化 + 事後驗證)。

---

## 20. 全連桿幾何:覆蓋半徑與 1-Lipschitz 論證

**程式碼**:[arm_link_geometry.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_link_geometry.py)(831 行,手臂側最大的模組)

**這一節是你手臂部分最有技術深度的地方。**

問題:barrier 列原本建在 12 個固定偏移的偵測點上,**那只約束了那 12 個點**。實測(60 個合法構型):碰撞幾何最遠**超出最近偵測點 0.1062 m**(link_base;link4 0.0823,link2 0.0655),而零速煞停距離是 **0.08 m** —— 所以**每一列都可以報告「已滿足」而連桿其實已經接觸**。5B 實驗中差距穩定為 **0.0416 m**,正好是偵測點報告的 0.0710 m 與網格實際 0.0373 m 之差。

**你的兩段論證**

第一段,膨脹距離可以修「距離」,因為距離場是 1-Lipschitz:
```
d_link ≥ d(p_i) − ρ_i
```

第二段(關鍵),**但修不了「速度」**:同連桿上另一點 `x` 的速度是
```
v_x = v_{p_i} + ω_i × (x − p_i)
```
所以距 `ρ_i` 遠的點可以比 `p_i` 接近得更快,最多快 `|ω_i|·ρ_i`。**保留 `J_{p_i}` 而只膨脹距離,仍然無法支撐關於整條連桿的主張。**

你的解法:每週期找出**真正最接近障礙物的取樣表面點**,距離、法向、雅可比**全都取在那個點**。剩下的誤差只有離散化(真實表面極小值可能落在取樣之間),由取樣覆蓋半徑 `ρ_sample` 界定,而 `ρ_sample` 是**量測的而非假設的**。

且**比原方案更不保守**:`ρ_i` 必須覆蓋整條連桿的最壞情形,`ρ_sample` 只需覆蓋相鄰取樣點的間隙。

**文獻**
- ✓ VFI(§19)— 現代 VFI 用幾何基元(球、capsule、平面)而非取樣點,對凸基元可得解析最近點。**你要說明為何選取樣而非基元**(Lite 6 網格非凸?取樣可處理任意網格?)
- ○ Schulman et al. 2014(§19)— convex-convex 最近距離、swept volume;凸分解後的精確距離
- ○ Gilbert, Johnson, Keerthi, "A fast procedure for computing the distance between complex objects in three-dimensional space" (GJK), *IEEE J. RA* 4(2), 1988 — 凸體精確距離
- ○ Pan, Chitta, Manocha, "FCL: A general purpose library for collision and proximity queries", ICRA 2012 — MoveIt 用的碰撞庫,可作為比較基準
- ○ Lin & Canny / ○ Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2004 — capsule/OBB 距離
- ○ **1-Lipschitz 距離場**:○ Osher & Fedkiw, *Level Set Methods*,或 SDF 文獻;用來支撐 `d_link ≥ d(p_i) − ρ_i` 這一步。也可引 ○ Koptev, Figueroa, Billard, "Neural joint space implicit signed distance functions for reactive robot manipulator control", *RA-L* 2023 — 機器人 SDF 的近期工作

**可主張的貢獻**:「固定偵測點的 barrier 無法支撐關於整條連桿的安全主張,而且膨脹距離只修一半(修距離不修速度)」這個論證 + 0.1062 m 的量化證據。我在 VFI 文獻裡沒看到把「距離膨脹修不了速度項」明確寫出來的。這一節可以獨立成小節,配一張「偵測點 vs 真實最近點」的圖。

⚠️ 但要誠實:VFI 用解析基元本來就沒有這個問題(基元覆蓋整個連桿)。所以你的論證是針對「固定取樣點」這個特定做法,**不是針對 VFI 本身**。措辭要精確,否則會被指為攻擊稻草人。建議寫成:「以固定取樣點實作 velocity damper 時(如我們 Phase 1 的做法與 [arm_detection_points 的 ROS 1 前身]),會有以下問題……解法之一是解析基元(VFI),我們選擇取樣表面點的理由是……」

---

## 21. 手臂運動包絡與力矩

**程式碼**:[arm_limits.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_limits.py)、[arm_payload_limits.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_payload_limits.py)、[arm_dynamics.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_dynamics.py)

**靜態重力力矩**:
```
τ(q) = − Σ_j J_{v,com_j}(q)ᵀ m_j g
```
雅可比取在每個連桿的**質心**而非原點(link2 差距達 18 cm)。

**逆動力學**(用已驗證的雅可比組裝 Newton-Euler,非手寫遞迴):
```
τ = M(q)q̈ + C(q,q̇)q̇ + g(q) + JᵀF_ext
τ = Σ_j J_{v,j}ᵀ(m_j a_{c,j}) + J_{w,j}ᵀ(I_j α_j + ω_j × I_j ω_j)
a_{c,j} = J_{v,j}q̈ + J̇_{v,j}q̇ − g
ω_j = J_{w,j}q̇ ,  α_j = J_{w,j}q̈ + J̇_{w,j}q̇
```
`J̇` 用沿 `q̇` 的中央差分(二階精確,重用同一份雅可比程式碼)。

**文獻**
- ○ Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008 — RNEA/ABA 標準參考
- ○ Luh, Walker, Paul, "On-line computational scheme for mechanical manipulators", *J. Dynamic Systems, Measurement, and Control* **102**(2):69–76, 1980 — **遞迴 Newton-Euler 原始**
- ○ Craig, *Introduction to Robotics: Mechanics and Control*, 4th ed., Pearson, 2017, ch. 6 — 你用的「雅可比組裝」形式(Lagrangian/雅可比法)標準推導
- ○ Siciliano, Sciavicco, Villani, Oriolo, *Robotics*, Springer, 2009, ch. 7 — 動力學模型與 `M, C, g` 的性質
- ○ Bowling & Khatib, "The dynamic capability equations: A new tool for analyzing robotic manipulator performance", *IEEE T-RO* 21(1), 2005 — 力矩受限的動態能力分析;**對應你的「負載能力隨姿態變化」**
- ○ Chiacchio, Chiaverini, Sciavicco, Siciliano, "Influence of gravity on the manipulability ellipsoid for robot arms", *J. Dynamic Systems* 114(4), 1992 — 重力對可操作性的影響

**你的方法論選擇值得寫**:遞迴 Newton-Euler 先寫過然後**捨棄** —— 它與已驗證的重力模型在靜止時差 **13 N·m**,錯在遞迴的 frame 合成而非物理。你改用的形式**結構上不可能有那類 bug**:`q̇ = q̈ = 0` 時它自動退化成 `τ = −Σ J_{v,j}ᵀ m_j g`,即重力模型本身 —— **by construction,不是 by coincidence**。代價是 O(n²) 而非 O(n),六軸無所謂。

這是很好的「可驗證性優於效率」的工程論證,論文的 implementation/validation 節值得寫一段。引 Featherstone 說明標準做法是 O(n) 遞迴,然後說明你為何在 n=6 時選擇 O(n²) 的可驗證形式。

**⚠️ 資料來源誠信 —— 這部分你寫得非常好,論文一定要照抄這個標準**

[arm_limits.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_limits.py) 明確區分:

| 數值 | 來源 | 狀態 |
|---|---|---|
| 關節範圍、速度 180°/s、加速度 1145°/s²、jerk 28647°/s³ | UFACTORY "Lite 6 Hardware Manual V2.6.0", Preface | **已驗證** |
| 負載 600 g、最大伸距 440 mm | "Lite 6 User Manual V2.3.0", Appendix 6 §1.8, model LI1000 | **已驗證** |
| 關節力矩 [50,50,32,32,32,20] N·m | **本專案作者實驗測定**(2026-09-12);與 UFACTORY URDF 的 `<limit effort>` 吻合但非轉抄 | **非原廠公布值** |

而且明言:Hardware Manual V2.6.0 **完全沒有**關節力矩表,User Manual V2.3.0 也沒有;兩者唯一的 N·m 數字是底座螺栓的 20 N·m 緊固力矩,與此無關。

`arm_payload_limits.py` 的免責也很完整:這是**靜態**項,加速會疊加慣性力矩,所以靜止通過的姿態運動中仍可能過載;報告的餘裕是「可用於加速的上界」而非「該姿態任何軌跡可行的證明」。連桿質量與質心來自 URDF(UFACTORY 公布模型)**而非實機量測**;摩擦、齒輪箱效率、負載偏心全部缺席。「當作帶餘裕的篩選工具,不是力矩預測,更不是硬體承載能力的證據。」

> 論文寫硬體參數時照這個格式做一張表(數值/來源/狀態)。這種明確標示「哪個數字有原廠依據、哪個是自己測的、哪個是推論」的做法,在論文裡很少見,而且直接回應口委最常問的「這個數字哪來的」。

---

## 22. 手臂自我濾除與偵測點

**程式碼**:[arm_scan_self_filter.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_scan_self_filter.py)、[arm_detection_points.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_detection_points.py)

**文獻**
- ○ MoveIt `robot_self_filter` / ○ Chitta, Sucan, Cousins, "MoveIt!", *IEEE Robotics & Automation Magazine* 19(1), 2012 — 自我濾除的標準實作
- ○ Pan, Chitta, Manocha, "FCL", ICRA 2012(§20)
- ○ Flacco et al. 2012(§19)— depth space 的自我遮罩

**關鍵論點(與 §19 相連)**:手臂遮擋 LiDAR 之處,移除自身回波後**留下的是「沒有量測」而不是「淨空」**。所以最近表面落在被遮擋方位的點會被標為 `occluded`,**永不標為 clear**。同樣邏輯適用於檢查本身無法執行時:感測器變換缺失 → 可觀測性未知 → 報告為 occluded 而非預設淨空。

這是 §12 同一個哲學(缺少 ≠ 安全)在手臂側的應用。論文可以把「**缺少資料不等於沒有危險**」提升為貫穿全文的設計原則,§12(缺少 track ≠ 沒有障礙)、§19(stale ≠ 可忽略)、§22(遮擋 ≠ 淨空)都是它的實例。**這個統一原則本身就是論文的主張之一。**

⚠️ [arm_detection_points.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_detection_points.py) 的 12 點 `PolygonStamped` 佈局**不能改**,因為它是某個手臂避障策略訓練時的觀測向量,點序與每點自有 frame 的慣例必須與訓練環境完全一致。無障礙時發布 **NaN 而非大數**,讓消費者能區分「附近沒東西」與「有東西在 10 m 外」—— 也是訓練時的約定。論文若提到這個節點,要說明它是 ROS 1 `collision_detection_points_node.cpp` 的移植,且介面受既有訓練策略約束(這解釋了為何 §19 另開一個 `arm_link_distance` 介面而不是改這個)。

---

# 第四部分:學習層

## 23. 殘差強化學習(平滑度)

**程式碼**:[train_sac.py](rl_smoothness/train_sac.py)、[rl_env.py](rl_smoothness/rl_env.py)、[cbf_filter.py](rl_smoothness/cbf_filter.py)、[env_real.py](rl_smoothness/env_real.py)

```
u_gmpc = GMPC(x, ref, obstacles)
u_nom  = u_gmpc + Δu_RL              ‖Δu_RL‖ ≤ ACT_LIM(小)
u_safe = CBF_filter(u_nom)            硬安全投影

r = c_prog·(d_prev − d_now)                     potential-based 進度
  − w_res·‖Δu_RL‖²                              殘差要小
  − w_yaw·|ω|                                   yaw rate(擺動)
  − w_h·max(0, h_danger − min_h)²               ← 關鍵項:到 CBF 邊界的深度
  ± collision / reached                         終端訊號
```

**文獻**
- ✓ Johannink, Bahl, Nair, Luo, Kumar, Loskyll, Ojea, Solowjow, Levine, ["Residual Reinforcement Learning for Robot Control"](https://www.semanticscholar.org/paper/ae4d32f05cf40e4cc01c69d7787149a258c95eda), ICRA 2019
- ✓ Silver, Allen, Tenenbaum, Kaelbling, ["Residual Policy Learning"](https://arxiv.org/abs/1812.06298), arXiv:1812.06298, 2018 — 同期獨立提出;「在不可微的既有控制器上學殘差」
- ○ Haarnoja, Zhou, Abbeel, Levine, "Soft Actor-Critic: Off-policy maximum entropy deep RL with a stochastic actor", ICML 2018 — **SAC**
- ○ **Ng, Harada, Russell, "Policy invariance under reward transformations: Theory and application to reward shaping", ICML 1999** — **potential-based reward shaping 的最佳性保證**
- ✓ Cheng, Orosz, Murray, Burdick, ["End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks"](http://www.cds.caltech.edu/~murray/preprints/comb19-aiaa.pdf), AAAI 2019 — RL + CBF,「RL 管性能、CBF 保命」原型
- ○ Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu, "Safe reinforcement learning via shielding", AAAI 2018 — shielding
- ○ Dalal, Dvijotham, Vecerik, Hester, Paduraru, Tassa, "Safe exploration in continuous action spaces", arXiv:1801.08757, 2018 — action projection layer
- ✓ [CBF-RL: Safety Filtering RL in Training with CBFs](https://arxiv.org/abs/2510.14959) — **訓練時就套 CBF(你的做法)vs 只在部署時套**,這篇正好討論這個區別
- ○ Emam, Fiore, Bowman, Notomista, Egerstedt, "Safe model-based reinforcement learning using robust control barrier functions", 2021 — RL + robust CBF
- ○ Zanon & Gros, "Safe reinforcement learning using robust MPC", *IEEE T-AC* 66(8), 2021 — RL 與 MPC 結合的安全性分析;**與你「CBF 在 MPC 內」的架構最相關**

**差異(這是你可主張的方法論貢獻,而且論證很紮實)**

**(1) 你指出文獻標準項在你的架構下恆為零。** 殘差 RL + CBF 文獻用的獎勵項是「CBF intervention penalty」`‖u_safe − u_nom‖`。那個形式**隱含假設 CBF 只存在於「安全無知的 nominal controller」之後的 shield**,所以 `‖u_safe − u_nom‖` 很大、帶梯度。

但你的 full-horizon CBF **已經在 GMPC QP 內部**,`u_opt` 在 shield 看到它之前就已可行 ⟹ 零殘差時 `‖u_safe − u_nom‖` **恆等於零**,該項對學習毫無貢獻。

**(2) 你的替代項與量化依據。** 量測真實 intervention `‖u_with_CBF − u_without_CBF‖`:CBF 確實有 **60.5%** 時間在塑形命令(中位數 0.005,p95 0.069)—— 但取得它要**每步多解一次 QP**。而 `min_h` 求解器已經回傳,且有 **63.9%** 時間低於 0.4 危險門檻。所以用 `max(0, h_danger − min_h)²` **免費**拿到同樣的「讓 barrier 保持不活躍」壓力。

**(3) potential-based 進度 —— 引 Ng 1999 讓失敗分析升級。** 你的第一版獎勵投影速度到目標方向,**每步都給付且無上界**:400 步片段累積約 **+80**,而所有平滑項都在 5 以下 ⟹ 策略學會「只要還在往前挪就盡量橫向甩」。實測 heading change 從 **135 deg/m → 932 deg/m(+589%)**。

> **這正是 Ng et al. 1999 定理預測的失敗模式**:非 potential-based 的 shaping 會改變最佳策略。你的 telescoping 形式 `Σ(d_prev − d_now)` 加總等於實際走過的距離(約 4 m/片段),**不可能跑贏 shaping 項**。引這篇,你的失敗分析就從「調參心得」變成「理論預期的實證」。

**(4) 論文措辭建議**(避免過度主張):
> 「既有的 intervention-penalty formulation 隱含了 CBF 位於 nominal controller 下游的架構假設。當 CBF 內嵌於 MPC 的 horizon 約束時該項退化為零,因此我們提出以 barrier 深度 `max(0, h_danger − min_h)²` 作為替代,並量測兩者在本系統上的統計特性以支持此替換。」

比「我們提出新獎勵函數」站得住太多。

**(5) OSQP 固定模式加速**(§4)在此也重要:RL rollout 需要大量 QP 求解,`setup()` 一次 + `update()` 的 10–30× 加速是可行性關鍵。未使用的槽位填遠端虛擬障礙(約束不活躍),保持約束矩陣模式與梯度非零皆穩定。

⚠️ **訓練/評估環境落差必須交代**(你的記憶 `project_fourth_progress_rl` 記錄的策略):在輕量 2D sim 訓練(CPU 熱限制),在 gz 評估。論文要明寫 sim-to-sim gap,並引 ○ Tobin et al., "Domain randomization", IROS 2017 或 ○ Peng et al., "Sim-to-real transfer of robotic control with dynamics randomization", ICRA 2018 說明為何可接受(或為何是 limitation)。

---

