#!/usr/bin/env python3
"""門機構約束開門：執行層（Codex 20261112 綁定冒煙 3 裁定的五點）。**機構約束離線規劃＋閉鏈誤差回授追蹤**——不是在線機構約束 MPC。
不沿用任何既有開門控制／時間排程／關節序列。綁定（理想抓持）成立後才啟用。

輸入
  plan      door_wbopt 的密集可執行參考 Xd（底盤 3、手臂 6、門角 1；間隔 h），起點＝綁定瞬間實測狀態。
  T_WH_bind 綁定瞬間的把手位姿、phi_bind 綁定瞬間門角、T_HE 綁定瞬間實測抓持變換（之後固定不變）。
  handover  從插入控制器原樣承接：手臂設定點 s、8 步在途命令管線、上一命令、手臂模型偏置。不重置、不清管線。
每控制拍（0.05 s）
  1 進度調節（明確公式）：參考相位 σ 的前進率 g＝clip((LEAD_MAX −(φ_ref(σ)−φ_實測))/LEAD_BAND, 0, 1)，σ ← σ＋dt·g。
    參考超前實測門角越多就越慢，超前到 LEAD_MAX 時停住等門；不讓參考與機構脫節後硬拉。g＝0 持續 STALL_S ⇒ 停止 door_not_following。
  2 預測視窗（k＝0…N）：σ_k＝σ_生效＋k·dt·g。
      底盤參考 X_ref[k]＝b_ref(σ_k)、本體速度前饋＝g·R(θ_ref)ᵀ·db_ref/dσ；
      預測門角 φ̂_k＝φ_ref(σ_k)；TCP 目標 T_seq[k]＝Rz[−(φ̂_k−φ_bind)]·T_WH_bind·T_HE（世界座標）；
      手臂線性化名目＝參考關節經設定點模型反算（同 jnom；求解器回傳的就是完整命令，不再另加前饋）。
  3 **同一個全身問題**（gmpc_wb.GMPC，底盤不固定＋手臂 active）同時輸出底盤與手臂命令；輪級耦合在求解器內。arm_mode='qp'（r5 預設）求解後不再改命令。
    arm_mode='closure'（r6，A；Codex 20261113_003032）：**底盤＝QP 提案**，手臂命令改為「命令生效時刻預測底盤＋預測門角」的閉鏈 IK 追蹤（替換 QP 手臂輸出）；
    替換後的命令有自己的限制驗收（手臂速度、相鄰命令加速度、預測設定點與 IK 構型限位、可操作度下限），不過 ⇒ 候選失敗＝closure_cmd_invalid（模擬中止）；
    只能稱「機構約束離線規劃＋QP 底盤提案＋閉鏈 IK 手臂追蹤」，不能稱最終命令仍是同一全身 QP 的最佳解。
  4 提交前核停止路徑（含**轉動中的門**）：對「現在停」（求解前）與「候選命令後下一拍停／再晚一拍停」各預測整段機器人軌跡（真管線、真減速），
    每個取樣點由把手點位置反算當時門角 φ_i，核：預測閉鏈殘差（徑向／高度／姿態；綁定下它代表內應力而非可見誤差）與全身對 φ_i 門扇的淨空（閉合手指開度）。
    違反 ⇒ 候選不提交、受控停止。
  5 監看（鎖存停止）：實測閉鏈殘差、設定點閉鏈殘差（手臂被命令去的位置對機構允許位置＝內應力指標）、實測淨空、進度停滯、求解失敗、外部超力旗標。
停止（r3，依 Codex 20261112_223736）：
  path    只用於 PATH_OK 類原因（外部停止、完成、未達、門不跟、設定點模型偏差、候選／停止掃掠被拒）：g 0.5 s 斜降、同一全身求解繼續追蹤，
          每拍新候選照樣經提交前核對、監看持續；任何一項不過或惡化 ⇒ abort。
  abort   **模擬中止**：其餘所有原因（超力 40 N〔模擬工程中止值，傳遞力範數，每物理步〕、求解失敗、非有限、限位、實測淨空／閉鏈、path 中被拒或惡化、path 逾時）。
          鎖存失敗、回 abort＝True；執行器必須在下一次 world.step 前停止物理步進。不 hold、不自動解綁、不退出；停止有效性標「未驗證」，不叫受控停止成功。
  解綁    只在正常完成（open_done 或 path 停止到 settled）且停止有效（管線清空、底盤／手臂／門停穩）之後，由執行器解除連接並讀回；張爪期間以兩指實際開度核幾何。
（r2 的 hold／release 與 release 路徑預測保留在 versions/door_open_exec_ctrl_r2.py，不再使用：release 會使夾爪殼碰橫桿，hold 有約 22 mm 閉鏈不相容且無反力模型。）
停止有效性分核：管線清空、底盤、手臂、門角速度。
完成：σ 到參考終點、實測門角 ≥ φ_goal−END_PHI_TOL、各速度低於門檻連續 END_N 拍。
數值皆為事前登錄的工程設定；未校準的傳遞力只作模擬保護。
r8（G2，Codex 20261113 時間帳裁定）：g2＝dict(gravity, ik_fc, ik_warm, sweep_cache)，預設全關＝r7 逐值相同。
  gravity     閉鏈錨定的 URDF 重力改單趟計算（door_fast_kin.FastGravity；與 door_body_model.arm_gravity 逐構型核對）。
  ik_fc       閉鏈 IK 每次迭代 FK＋Jacobian 一趟（FastChain；同迭代、同種子、同停止條件）。
  ik_warm     IK 種子改用上一拍同一預測時刻的解（上一拍節點 k＋1）；分支跳躍防護仍對原鏈式種子（上一節點）量，判準不變。
  sweep_cache 停止掃掠的淨空：輸入完全相同（構型位元組＋手指開度＋PHI0＋把手位姿）才重用、每拍清空 ⇒ 結果與重算相同；樣本數與核對項目不變。
r9（G3，Codex 20261113_120619）：g2['gravity2']＝重力改單趟串鏈後綴和（door_fast_kin.FastGravity2；與原式捨入級差 ≤9e-15 N·m，非逐位元）。
  g2['pool']＝工作程序數（0＝r8）。停止掃掠的樣本照原順序準備（停止軌跡、構型反算門角、預測閉鏈偏差），
  淨空分給預建工作程序（door_sweep_pool），結果依原順序判定 ⇒ 第一個違反、紀錄與 r8 相同。「現在停」在求解前派工、與求解同時計算，
  求解後才收：現在停不過 ⇒ 求解結果丟棄（同 r8 的停止處理；求解的熱啟動副作用保留，只影響初值）。候選掃掠未過不提交。
  逾時（POOL_TIMEOUT_S）、工作程序死亡／例外、結果缺漏、版本不符 ⇒ stop_sweep_unavailable／candidate_sweep_unavailable（不在 PATH_OK ⇒ 模擬中止），不當通過。
r10（Codex 20261113_124213 §4）：g2['prep_fc']＝掃掠準備的 TCP FK 改 FastChain（與 K.fk 捨入級差）；g2['deadline_ms']＝控制期限（None＝不檢查＝r9）：
  每拍從拍首計時，到「提交候選」那一刻若已超過期限 ⇒ 候選不提交（晚到結果不放行）。本拍「現在停」已驗過 ⇒ 改走已驗證的有界減速停法
  （stop_mode＝settled，等同現在停掃掠預測的停法；不是沿用上一命令）；本拍現在停沒驗到 ⇒ 模擬中止。池逾時 0.5 s 是資源失效判定，與此分開。
r11（Codex 20261113_130916 §1）：期限備援的「已驗證」改為本拍現在停預測同時滿足——淨空門檻、每個取樣的預測閉鏈位置／姿態偏差 ≤ CL_TOL
  （r3 起只計數不拒絕；備援要用就必須拒絕）、在步數上限內停穩（命令本身由有界減速構成、不超限）。路徑停止中晚到也要本拍重驗同樣條件
  （每拍另算現在停預測）；不滿足 ⇒ 模擬中止，不先標 settled。測試掛鉤：fb_closure_inject（放大備援評估用的預測閉鏈偏差）、deadline_inject（強制本拍晚到）。
r12（P2 執行層整合，Codex 20261113_124213 §3）：extend_reference(Xd_new, σ_s, final)——參考在 σ_s 起換成新段的密集參考（新段起點＝舊參考在 σ_s 的狀態
  與速度 ⇒ C1 連續、不跳參考）；σ_s 必須還沒到（σ ≤ σ_s）。p2_open＝True 時參考走到盡頭（段尾停穩）且門未到目標：不判 end_not_reached，
  等延伸（靜止重啟由外部在實測停穩後提出）；等超過 P2_REST_WAIT_S ⇒ p2_no_extension 停止。final＝True（末段）後回到原 END 規則。
r13（Codex 20261113_133147 §1）：新參考先暫存（stage_reference：非有限、版本不新、σ 已過、接縫不合 ⇒ 拒），接縫用執行層實際的 ref／dref 語意核
  （接縫兩側分段斜率之差／h 的本體加速度、輪速、手臂速率與加速度、設定點速率、門角速率與加速度，界限同 door_wbopt 驗收；以 g＝1 為最壞）。
  staged_step（每拍首、step 之前）：新參考最早會影響命令的那一拍（σ＋(D_STEPS·PHYS＋N·DT) ≥ σ_s）以複本試跑一拍——當下實測狀態、設定點、
  真管線、拍內斜坡——候選命令、閉鏈手臂驗收、提交前停止掃掠、預測停止閉鏈 ≤ CL_TOL、期限備援都過才接上本體；否則拒接，舊段與其停止尾段不動。
r17（Codex 20261115_131250）：停止升級含 cstop；停法物件共用且每次取用設成當下參考（P2 換段）、參考版本號綁結果；候選核對與試跑複本改用同一停法（提交一拍後停）；結果缺失≠核過（中止）；候選模擬提早派工。
r16（Codex 20261115_005224）：g2['cstop']＝door_cstop.CStopWorker ⇒ 綁定中的期限備援改「閉鏈一致停法」（door_cstop：沿參考把 g 降到 0、底盤前饋×g、
  手臂閉鏈 IK；不依賴 QP）。每拍把狀態快照送獨立程序以同一停法向前模擬（與求解並行），取樣構型再經計算池核淨空、主程序核閉鏈 ≤CL_TOL；
  全過才算備援可用（取代 r11 以各軸減速預測評估的備援——motm_3：該停法在綁定下留下設定點殘差、力升到 36 N、手臂停不穩）。
  晚到 ⇒ stop_mode＝'cstop'：每拍 step_cmd（不過 ⇒ 模擬中止）、監看照常（違反 ⇒ 中止）、4 s 內未停穩 ⇒ 中止；不退回各軸減速、不解綁。
r14：g2['pool'] 可傳外部共用的計算池（執行器在啟動模擬器前 fork）；只關自己建立的池。
r15：複本（_light_copy／warmup）深拷貝前暫時把 g2['pool'] 換成 0、拷貝後還原並讓複本共用同一池（池物件含程序認證字串不可拷貝；
  名目趟 m8house_door_motm_1 在交接試跑時因此例外中斷）；試跑複本不檢查控制期限（期限只對本體真實拍）。"""
import json
import math
import os
import sys
import time

import numpy as np
import door_fric_ff as FF
import door_cstop_def as DCSM
import door_preload_release as PR

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_wbopt as W                                             # noqa: E402
import door_limits as LIM                                          # noqa: E402
import gmpc_wb_ramp as GR                                          # noqa: E402
OPEN_FB_TICKS = int(os.environ.get('D2_OPEN_FB_TICKS', '2'))           # 開門段「底盤後處理改動」可連續執行備援命令的拍數（0＝原行為：一律停）
import dataclasses                                                 # noqa: E402
import door_body_model as BM                                       # noqa: E402
import door_fast_kin as DFK                                        # noqa: E402

DA = W.DA
GW, K, C0, B, C21, S = DA.GW, DA.K, DA.C0, DA.B, DA.C21, DA.S
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE               # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta                    # noqa: E402

PHYS, NPH, D_STEPS, N, DT = C0.PHYS, C0.NPH, C0.D_STEPS, B.N, B.DT
assert D_STEPS == DCSM.D_STEPS                                      # 停法共用定義的管線長度須一致
LEAD_MAX, LEAD_BAND, STALL_S = math.radians(3.0), math.radians(1.0), 2.0
CL_TOL = dict(pos_m=0.010, rot_deg=5.0)                            # 實測／預測閉鏈殘差上限
STRESS_TOL = dict(pos_m=0.030, rot_deg=8.0)                        # 設定點閉鏈殘差上限（內應力指標）
END = dict(phi_tol=math.radians(0.5), n_ticks=10, wait_s=4.0, arm_rps=0.02, base_mps=0.005, yaw_rps=0.01, door_rps=0.01)
T_STOP, PATH_STOP_MAX_S = 0.5, 3.0
QP_RESERVE_MS = float(os.environ.get('DOOR_QP_RESERVE_MS', '70'))   # 整拍期限中保留給候選核對、通訊的時間；QP 流程預算＝期限−已用−保留
G_GOV = os.environ.get('DOOR_G_GOV', '0') == '1'                  # Codex 191426：約束感知在線進度調節（只縮放 g；σ 仍只在候選＋尾段通過並提交時前進）
G_GOV_BAND = dict(env=0.010, base=0.030, self=0.010, shell_bar=0.006)   # 各項自身單位的預警帶：餘裕 ≥ 門檻＋帶寬 ⇒ 不減速；各項先正規化到 [0,1] 再取最小
G_GOV_MIN = float(os.environ.get('DOOR_G_GOV_MIN', '0.15'))      # 下限：避免 g→0 觸發 door_not_following；不假設越慢越安全                                # 路徑停止：g 由 1 降到 0 的時間、最長等待
PATH_OK = ('forced_stop', 'open_done', 'end_not_reached', 'door_not_following', 'setpoint_model_deviation', 'candidate_stop_violation', 'stop_sweep_violation')
# 其餘原因一律 abort（模擬中止）
FORCE_LIMIT_N = 40.0                                                # 模擬工程中止值（未校準傳遞力範數；不是安全上限）
D_DOOR = 2.0                                                        # 解除後門的簡化阻尼（1/s；預測用）
FINGER_OPEN = 0.0089
STOP_STILL = dict(arm_rps=2e-3, base_mps=0.003, yaw_rps=0.003, door_rps=0.005, hold_s=0.3, timeout_s=4.0)
KEYS = ('env', 'base', 'self', 'shell_bar')
START_LAT_ALIGN = True                                              # r5：參考時鐘從第一筆命令到達受控對象才起算（σ₀＝−D_STEPS·PHYS；底盤與手臂共用同一條管線延遲）
P2_REST_WAIT_S = 8.0                                               # r12：參考盡頭等延伸的上限
POOL_TIMEOUT_S = 0.5                                                # r9：平行掃掠收集逾時（偵測卡死用；即時性另以每拍耗時統計）
STREAM_CHUNK = int(os.environ.get('CAND_STREAM_CHUNK', '6'))         # A1：每批取樣點數
_POOL_LOCKS = {}                                                     # A1：同一計算池同一時間只由一個執行緒使用（現在停背景執行緒、候選串流轉送執行緒）；以池物件 id 對應（複本共用同池＝同鎖）


def _pool_lock(pool):
    import threading
    return _POOL_LOCKS.setdefault(id(pool), threading.Lock())
CL_ROWS = None if os.environ.get('DOOR_CL_ROWS', '1') != '1' else dict(e=float(os.environ.get('DOOR_CL_E', '0.006')), e_rot=math.radians(float(os.environ.get('DOOR_CL_EROT_DEG', '3.0'))), n=int(os.environ.get('DOOR_CL_N', '4')))
BASE_KEEP = float(os.environ['DOOR_BASE_KEEP_M']) if os.environ.get('DOOR_BASE_KEEP_M') else None   # 共同 QP 內底盤—門扇淨空下界（m；None＝不加）
LENIENT_SHELL = os.environ.get('D2_LENIENT', '0') == '1'
LENIENT_SHELL_FLOOR = 0.003
def _lenient_shell_ok(b_):
    """寬鬆展示（D2_LENIENT）：預測停止路徑只有外殼—橫桿低於門檻且 ≥3 mm ⇒ 只記錄、不擋（實測淨空與其他項照舊）。"""
    bad_ = set((b_ or {}).get('clear_bad') or [])
    return bool(LENIENT_SHELL and bad_ and bad_ <= {'shell_bar'} and float(((b_ or {}).get('at') or {}).get('shell_bar', 0.0)) >= LENIENT_SHELL_FLOOR)


WHEEL_FRAC = tuple(float(x) for x in os.environ.get('DOOR_WHEEL_FRAC', '1.0,1.0').split(','))   # 控制器端輪速／輪加速度上限比例（預設 1＝原值）
BASE_RAMP = True                                                    # r4：底盤命令在一個控制拍內逐物理步線性過渡（管線、預測、停止掃掠同一語意）


def ramp_cmd(u_from_base, u_tick, j):
    """第 j 個物理步（0…NPH−1）實際送出的命令：底盤分量由上一筆線性過渡到本拍命令，手臂分量＝本拍命令（設定點積分本身已平滑）。"""
    a = min(j + 1, NPH) / NPH if BASE_RAMP else 1.0
    return np.r_[u_from_base + a * (u_tick[:3] - u_from_base), u_tick[3:]]


def stop_traj_ramp(q, s, pipe, u_last, u_from_base, j0, cfg, extra_ticks=0):
    """C21.stop_traj 的斜坡版（同一個設定點／底盤受控對象 S.plant_phys_step）：每控制拍有界減速，底盤分量在拍內逐物理步過渡。
    j0＝起點在目前控制拍內已過的物理步數（拍首呼叫＝0，u_from_base 為本拍過渡起點）。回每 SAMPLE_EVERY 物理步的 q。"""
    qp, sp = np.asarray(q, float).copy(), np.asarray(s, float).copy()
    pp = [np.asarray(u, float).copy() for u in pipe]
    u, uf, j = np.asarray(u_last, float).copy(), np.asarray(u_from_base, float).copy(), int(j0)
    da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * cfg.dt
    out, k, quiet = [qp.copy()], 0, 0
    while quiet < C21.SETTLE_STEPS and k < 2000:
        if j >= NPH:
            uf, j = ramp_cmd(uf, u, NPH - 1)[:3], 0
            if k >= extra_ticks * NPH:
                u = u + np.clip(-u, -da, da)
        cmd = ramp_cmd(uf, u, j)
        pp.append(cmd)
        qp, sp = S.plant_phys_step(qp, sp, pp.pop(0), cfg, 1)
        k += 1
        j += 1
        quiet = quiet + 1 if (not np.any(cmd) and not any(np.any(x) for x in pp)) else 0
        if k % C21.SAMPLE_EVERY == 0:
            out.append(qp.copy())
    return np.array(out)


def rot_deg(Ta, Tb):
    Re = Ta[:3, :3].T @ Tb[:3, :3]
    v = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return math.degrees(math.atan2(float(np.linalg.norm(v)), 0.5 * (float(np.trace(Re)) - 1.0)))


class DoorOpenExec:
    def __init__(self, ident, Xd, h, T_WH_bind, phi_bind, T_HE, phi_goal, handover, finger_q, sweep_stride=2, arm_mode='qp', joint_kp=600.0, g2=None):
        self.g2 = dict(gravity=False, ik_fc=False, ik_warm=False, sweep_cache=False, pool=0, gravity2=False, gravity3=False, prep_fc=False, deadline_ms=None, cstop=None, fric_ff=None, preload_release=False)
        self.n_deadline_miss = 0
        self.p2_open, self.p2_log = False, []                         # r12
        self.staged, self._last_ver = None, 0                         # r13
        self._cs_box, self._cs_th, self._cs_out, self.cs_log, self._xd_ver = [None], None, None, [], 0   # r16；r17：停法物件放共用盒（試跑複本共用）、參考版本號
        self._ident = ident
        self._fb_ok, self._fb_info = False, None                      # r11：本拍期限備援是否可用＋依據
        self.fb_closure_inject, self.deadline_inject = None, None      # r11：測試掛鉤
        self._cand_tr, self._sn_tr = None, None
        self.rec_cand, self._cand_out = False, None
        self.stream_cand = False                                       # A1：候選停法模擬分批送淨空池（非阻塞模式才開；判定不變）
        self.tail_cert_mode = False                                    # A2（Codex 20261117_003351）：非阻塞「尾段證書」模式——晚到／失敗時執行器只執行上一段已核尾段，
        #   故不重算從不會被執行的「現在停」軌跡；接受條件改為「本拍候選＋其完整停止尾段已核」（_cand_out 非空）。監看（力、閉鏈、門角、限位）照舊。只在非阻塞執行器明確設定時開啟。                    # 非阻塞執行（11/17）：候選核對的停止模擬一併回逐物理步輸出（＝執行器停止尾段）；只記錄，判定不變
        self._early_cb, self._hc_e = None, None                        # r17：候選核對提早派工
        self.g_inject = None                                           # r17：測試掛鉤（改寫本拍重算出的 g）
        self.cs_inject = None                                          # r17：測試掛鉤（閉鏈一致停法核對結果改為 'missing'／'infeasible'）
        self.fb_log = []
        for k_ in (g2 or {}):
            if k_ not in self.g2:
                raise ValueError(k_)
        self.g2.update(g2 or {})
        self._grav = DFK.FastGravity3(K, BM.arm_bodies(K)) if self.g2.get('gravity3') else DFK.FastGravity2(K, BM.arm_bodies(K)) if self.g2['gravity2'] else (DFK.FastGravity(K, BM.arm_bodies(K)) if self.g2['gravity'] else None)
        self._ccache = DFK.ClearCache(W.clear_door) if self.g2['sweep_cache'] else None
        self._csw = self.g2['cstop']                                 # r16（須在 g2 更新之後讀）
        self._pool, self._pool_owned = None, False
        if not isinstance(self.g2['pool'], (int, float)):            # r14：外部共用的計算池（執行器在啟動模擬器前 fork），本控制器不關它
            self._pool = self.g2['pool']
        elif int(self.g2['pool']) > 0:
            import door_sweep_pool as DSP
            self._pool, self._pool_owned = DSP.SweepPool(int(self.g2['pool'])), True
        self.Q_cl_prev_full = None
        self._sn = None
        self._sn_verified = False
        self.pool_ms, self.prep_ms = [], []
        self.mon_ms, self.cl_ms, self.cand_ms, self.sn_ms = [], [], [], []   # r18：逐段計時（監看、閉鏈手臂 IK＋重力、候選核對等待、現在停收結果）；都在整拍期限之內
        self.cfg = C0.make_cfg(ident)
        self.cfg.base_fixed = False
        self.cfg.arm_model.bias = np.asarray(handover['bias'], float).copy()
        self._bias0 = self.cfg.arm_model.bias.copy()
        nc = B.nav_cfg()
        rc = GR.GMPCConfig(**{f.name: getattr(nc, f.name) for f in dataclasses.fields(nc)})
        rc.cmd_ramp_w = (0.4, 0.6) if BASE_RAMP else None             # r5：GMPC 底盤預測與 ramp_cmd 同語意（20…100% ⇒ 平均 0.4·u_{k−1}＋0.6·u_k）
        rc.wheel_w_max, rc.wheel_a_max = rc.wheel_w_max * WHEEL_FRAC[0], rc.wheel_a_max * WHEEL_FRAC[1]   # 控制器端輪級餘裕（GUARD 不動；c38：GUARD 依自身收件間隔算加速度，模擬時鐘抖動下縮放命令 ⇒ 放行≠提案）
        self.mpc = GR.GMPC(rc)
        self.arm_mode = str(arm_mode)                                # r6：'qp'＝全身 QP 的手臂輸出（r5 預設）；'closure'＝A 閉鏈一致設定點（命令生效時刻的預測底盤＋預測門角解 IK）
        assert self.arm_mode in ('qp', 'closure')
        self.cl_stats = dict(n=0, rate_clip=0, ik_fail=0, sigma_min=np.inf, ik_resid_max=0.0, s0_jump_max=0.0, rate_max=0.0, acc_max=0.0, fails=[])
        self._cl_fail = None
        self.q_cl_prev = None
        self.joint_kp = float(joint_kp)                              # A：重力偏置換算用的關節 PD 剛度（執行器 --kp；登錄值）
        self.off_cl0, self.G0, self.arm_bodies, self.anchor = None, None, BM.arm_bodies(K), None
        self.Xd, self.h = np.asarray(Xd, float), float(h)
        self.T_WH_bind, self.phi_bind, self.T_HE, self.phi_goal = np.asarray(T_WH_bind, float), float(phi_bind), np.asarray(T_HE, float), float(phi_goal)
        W.set_model(phi0=self.phi_bind, finger_q=finger_q)
        self.T_EH = np.linalg.inv(self.T_HE)
        pH = self.T_WH_bind[:3, 3]
        self.ang_bind, self.r_handle, self.z_handle = math.atan2(pH[1], pH[0]), float(np.hypot(pH[0], pH[1])), float(pH[2])
        self.s = np.asarray(handover['s'], float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in handover['pipe']]
        assert len(self.pipe) == D_STEPS
        self.u_last, self.u_prev = np.asarray(handover['u_last'], float).copy(), np.asarray(handover['u_prev'], float).copy()
        self.lim_lo, self.lim_hi = np.asarray(LITE6_SAFE.lower, float) + 0.05, np.asarray(LITE6_SAFE.upper, float) - 0.05
        self.sig_end = (len(self.Xd) - 1) * self.h
        self.sigma, self.g, self.t, self.n_step = (-D_STEPS * PHYS if START_LAT_ALIGN else 0.0), 0.0, 0.0, 0
        self._gov_m, self._gov_g_prev, self.gov_log, self.first_fail = None, None, [], None
        self._se_end = self.sigma + D_STEPS * PHYS                     # r17：上一個真正已提交命令的參考終點（只在提交時更新）；起始＝第一拍求解的起點
        self._se_cand = self._se_end
        self._hold = None                                             # r20（B1）：閉鏈一致停法的底盤保持錨點（世界 x, y, yaw）；None＝未啟用
        self.hold_log = []
        self._hold_t0 = None
        self._pr_delta, self._w_ext, self.pr_log = np.zeros(6), None, []   # r22（B2）：已提交的設定點偏置 δ、最新外力旋量 (F, M@TCP, 世界)、紀錄
        self._pr_cand, self._pr_commit_tick = None, False             # r23：本拍候選 δ（連同候選命令與停止預測通過才提交；被拒或晚到即丟棄）
        self._pr_inf, self.pr_rejected = {}, 0
        self.qp_log = []                                              # T3：每拍求解分項（t、是否已停止、前置 ms、mpc.solve ms、組裝 ms、GI ms、OSQP ms、求解器 0=gi／1=osqp）
        self.pr_ts = []                                               # 紀錄（每控制拍）：t、事件(1 提交／−1 候選被拒或晚到／0 無候選)、δ 候選 6、δ 已提交 6、δ 已生效 6、輸入旋量 F3＋M3
        self._pr_cand_last = None
        self._pr_u, self._pr_pipe, self._pr_s = np.zeros(6), [np.zeros(6) for _ in range(D_STEPS)], np.zeros(6)   # δ 增量對應的命令分量、其在途管線、已實際進入設定點的 δ（含管線延遲）
        self.stop, self.done, self.stopped_t, self.stop_ok, self._still = None, False, None, None, 0
        self.events, self.rec, self.fail_inject = [], [], None
        self.solve_ms, self.sweep_ms, self.tick_ms = [], [], []
        self.q_tick, self.phi_tick, self.rate = None, None, dict(base=0.0, yaw=0.0, arm=0.0, door=0.0)
        self.q_last, self.phi_last, self._stall, self._n_end, self.t_ref_end = None, None, 0.0, 0, None
        self.sweep_stride = int(sweep_stride)
        self.mins = dict(act={k: np.inf for k in KEYS}, stop={k: np.inf for k in KEYS}, n_sweep=0, stop_closure_pos=0.0, stop_closure_rot=0.0)
        self.overforce = None
        self.stop_mode, self.request_unbind, self.request_finger_open, self.t_path_stop, self.n_emerg_closure = None, False, False, None, 0
        self.bound, self.unbind_t, self.force_max, self.force_n = True, None, 0.0, None
        self.abort = None
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), NPH   # 斜坡狀態：本拍底盤過渡起點、本拍已過物理步數
        # r7（MotM 非零狀態交接）：handover['moving']＝True 表示承接時底盤／手臂在動、管線非空、可能在拍中途。
        #   off_static：上游觀測器（或模型分離）給的靜態偏置 q−s（不含動態追蹤落後）；closure 錨定用它，不取當下 q−s。
        #   斜坡預測的「上一命令」改為實際最後一個物理步送出的底盤命令（拍中途承接時與 u_last 不同）。
        self.moving_handover = bool(handover.get('moving', False))
        self.off_static = None if handover.get('off_static') is None else np.asarray(handover['off_static'], float).copy()
        if self.moving_handover:
            self.u_prev = self.u_prev.copy()
            self.u_prev[:3] = np.asarray(self.pipe[-1][:3], float)
        self.cmd_hist = []                                           # 每物理步實際送進管線的命令（驗收用）
        self.step_log, self.pipe_log = [], []                        # r5 紀錄：逐物理步（提交／生效命令、設定點、σ、生效參考時刻）、逐控制拍管線快照；只記錄，不影響行為
        self.sigma0 = float(self.sigma)
        self.sweep_inject, self.clear_inject, self.closure_inject = None, None, None   # 測試用故障注入
        self.rel_min, self.n_release_viol, self.mon_prev = {k: np.inf for k in KEYS}, 0, None
        self.finger_q_closed = float(finger_q)
        self._ev('start', sigma_end=round(self.sig_end, 2), phi_bind_deg=round(math.degrees(self.phi_bind), 3), pipe_nonzero=int(sum(bool(np.any(u)) for u in self.pipe)))

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(float(self.t), 3), what=what, **kw))

    def _stop(self, why, **kw):
        mode = 'path' if (why in PATH_OK and (self.stop is None or self.stop_mode == 'path')) else 'abort'
        if self.stop is None:
            self.stop = dict(t=round(float(self.t), 3), why=why, sigma=round(self.sigma, 3), mode=mode, **kw)
            self.t_path_stop = self.t
            self._ev('stop', why=why, mode=mode, **kw)
            self.stop_mode = mode
        elif self.stop_mode in ('path', 'settled', 'cstop') and mode == 'abort':   # 正常停止途中再出錯 ⇒ 模擬中止（不降級）；r17：含閉鏈一致備援
            self.stop.setdefault('escalations', []).append(dict(t=round(float(self.t), 3), why=why, to='abort', **kw))
            self._ev('stop_escalated', why=why, to='abort', **kw)
            self.stop_mode = mode
        if self.stop_mode == 'abort' and self.abort is None:
            self.abort = dict(t=round(float(self.t), 3), why=why, **kw)
            self.stop_ok = dict(t=round(float(self.t), 3), ok=None, status='unverified（模擬中止，不是受控停止）')

    def set_finger(self, q_finger):
        """張爪期間由執行器以兩指實際開度更新淨空模型（不因已下張爪命令就換成全開模型）。"""
        W.set_model(phi0=self.phi_bind, finger_q=float(q_finger))

    def confirm_unbind(self, ok, detail=None):
        """執行器在解除連接（並讀回 jointEnabled＝False）後呼叫；之後門角由執行器獨立量測傳入。"""
        if ok and self.bound:
            self.bound, self.unbind_t = False, self.t
            self._ev('unbind_confirmed', detail=detail)
        elif not ok:
            self._ev('unbind_not_confirmed', detail=detail)

    # ---- 參考
    def ref(self, sig):
        x = min(max(sig, 0.0), self.sig_end) / self.h
        i = min(int(x), len(self.Xd) - 2)
        a = x - i
        return (1 - a) * self.Xd[i] + a * self.Xd[i + 1]

    def dref(self, sig):
        if sig <= 0.0 or sig >= self.sig_end:
            return np.zeros(10)
        i = min(int(sig / self.h), len(self.Xd) - 2)
        return (self.Xd[i + 1] - self.Xd[i]) / self.h

    def handle_T(self, phi):
        return W.handle_pose(self.T_WH_bind, phi)

    def phi_of(self, q9):
        """由機器人構型反算門角（把手點繞鉸鏈的方位角）；回 (φ, 把手點)。"""
        pH = (K.fk(q9, 'link_tcp') @ self.T_EH)[:3, 3]
        d = math.atan2(pH[1], pH[0]) - self.ang_bind
        d = (d + math.pi) % (2 * math.pi) - math.pi
        return self.phi_bind - d, pH

    def closure_at(self, q9, phi):
        """構型 q9 對門角 φ 的閉鏈殘差：位置（m）、姿態（°）。"""
        Tc, Td = K.fk(q9, 'link_tcp'), self.handle_T(phi) @ self.T_HE
        return float(np.linalg.norm(Tc[:3, 3] - Td[:3, 3])), rot_deg(Td, Tc)

    def model_q(self, q):
        """設定點在模型下的穩態構型（s＋bias/α）：手臂「被命令去的位置」。與實測門角下的把手比＝內應力指標（r1 直接用 s，把靜態偏置誤當應力）。"""
        return np.r_[q[:3], self.s + np.asarray(self.cfg.arm_model.bias, float) / np.asarray(self.cfg.arm_model.alpha, float)]

    # ---- 停止路徑（含轉動的門）
    def _sweep(self, q, u, extras, tag, door='bound', phi0=None, omega0=0.0):
        """真管線停止預測。door＝'bound'：門角由機器人構型反算（綁定中、hold 停法）；'free'：門自由轉 φ(t)＝φ0＋ω0/d(1−e^{−dt})（release 停法／已解除）。
        回 None 或違反資訊；release 路徑（tag='release'）只記錄不回違反。"""
        worst = None
        fq_save = W.FINGER_Q
        if door == 'free':
            W.FINGER_Q = FINGER_OPEN
        try:
            for e in extras:
                Q = stop_traj_ramp(q, self.s, self.pipe, u, self.u_from_base if u is self.u_last else self._ramp_end(), 0 if u is not self.u_last else self.ramp_j, self.cfg, e)
                idx = list(range(0, len(Q), self.sweep_stride))
                if idx[-1] != len(Q) - 1:
                    idx.append(len(Q) - 1)
                for i in idx:
                    if not np.isfinite(Q[i]).all():
                        return dict(extra=e, i=i, tag=tag, nonfinite=True)
                    if door == 'bound':
                        phi_i, _ = self.phi_of(Q[i])
                        ep, er = self.closure_at(Q[i], phi_i)
                        self.mins['stop_closure_pos'], self.mins['stop_closure_rot'] = max(self.mins['stop_closure_pos'], ep), max(self.mins['stop_closure_rot'], er)
                        if ep > CL_TOL['pos_m'] or er > CL_TOL['rot_deg']:
                            self.n_emerg_closure += 1
                    else:
                        ts = i * C21.SAMPLE_EVERY * PHYS
                        phi_i = min(max(phi0 + omega0 / D_DOOR * (1.0 - math.exp(-D_DOOR * ts)), 0.0), math.radians(110.0))
                    c = (self._ccache if self._ccache is not None else W.clear_door)(np.r_[Q[i], phi_i], self.T_WH_bind)
                    if self.clear_inject is not None:
                        c = self.clear_inject(self, c)
                    self.mins['n_sweep'] += 1
                    tgtm = self.rel_min if tag == 'release' else self.mins['stop']
                    for k in KEYS:
                        tgtm[k] = min(tgtm[k], c[k])
                    bad = [k for k in KEYS if c[k] < W.THR[k]]
                    if (bad or not (-0.02 <= phi_i <= W.PHI_MAX + 0.02)) and worst is None:
                        worst = dict(extra=e, i=i, tag=tag, door=door, clear_bad=bad, phi_deg=round(math.degrees(phi_i), 2), at={k: round(float(c[k]), 4) for k in KEYS})
                        if tag != 'release':
                            return worst
            if self.sweep_inject is not None and tag != 'release':
                inj = self.sweep_inject(self, tag)
                if inj is not None:
                    return inj
            if tag == 'release' and worst is not None:
                self.n_release_viol += 1
                self.release_worst = worst
            return None
        finally:
            W.FINGER_Q = fq_save

    # ---- r9 平行掃掠
    def _sweep_prep(self, q, u, extras):
        """依原順序：停止軌跡、取樣點、構型反算門角、預測閉鏈偏差；遇非有限即截斷（原式在此返回）。"""
        w0 = time.perf_counter()
        items = []
        self._prep_settled = {}
        for e in extras:
            Q = stop_traj_ramp(q, self.s, self.pipe, u, self.u_from_base if u is self.u_last else self._ramp_end(), 0 if u is not self.u_last else self.ramp_j, self.cfg, e)
            self._prep_settled[e] = bool((len(Q) - 1) * C21.SAMPLE_EVERY < 2000 - C21.SAMPLE_EVERY)   # r11：步數上限內停穩
            idx = list(range(0, len(Q), self.sweep_stride))
            if idx[-1] != len(Q) - 1:
                idx.append(len(Q) - 1)
            for i in idx:
                if not np.isfinite(Q[i]).all():
                    items.append((e, i, None, None, None))
                    self.prep_ms.append((time.perf_counter() - w0) * 1e3)
                    return items
                Tc = W.FC.fk_jac(Q[i])[0] if self.g2['prep_fc'] else K.fk(Q[i], 'link_tcp')   # 同 phi_of／closure_at 的式子，只算一次 FK；r10 prep_fc＝FastChain
                pH = (Tc @ self.T_EH)[:3, 3]
                d_ = math.atan2(pH[1], pH[0]) - self.ang_bind
                d_ = (d_ + math.pi) % (2 * math.pi) - math.pi
                phi_i = self.phi_bind - d_
                Td = self.handle_T(phi_i) @ self.T_HE
                ep, er = float(np.linalg.norm(Tc[:3, 3] - Td[:3, 3])), rot_deg(Td, Tc)
                items.append((e, i, np.r_[Q[i], phi_i], ep, er))
        self.prep_ms.append((time.perf_counter() - w0) * 1e3)
        return items

    def _pool_submit(self, items):
        X = [it[2] for it in items if it[2] is not None]
        return self._pool.submit(self.T_WH_bind, W.FINGER_Q, W.PHI0, np.array(X)) if X else None

    def _pool_eval(self, items, h, tag):
        """回 (違反或 None, 失敗原因或 None)。判定順序、紀錄與 r8 的 _sweep 相同。"""
        res = []
        if h is not None:
            res, why, tms = self._pool.collect(h, POOL_TIMEOUT_S)
            self.pool_ms.append(tms)
            if res is None:
                return None, why
        k = 0
        fb = tag in ('stop_now', 'fallback')                           # r11：現在停同時評估期限備援
        fb_ok, fb_cl = fb and all(self._prep_settled.values()), [0.0, 0.0]
        for (e, i, x, ep, er) in items:
            if x is None:
                return dict(extra=e, i=i, tag=tag, nonfinite=True), None
            c = res[k]
            k += 1
            if fb:
                ep2, er2 = (ep, er) if self.fb_closure_inject is None else self.fb_closure_inject(self, ep, er)
                fb_cl = [max(fb_cl[0], ep2), max(fb_cl[1], er2)]
                if ep2 > CL_TOL['pos_m'] or er2 > CL_TOL['rot_deg'] or any(c[kk] < W.THR[kk] for kk in KEYS):
                    fb_ok = False
            if tag == 'fallback':
                continue
            self.mins['stop_closure_pos'], self.mins['stop_closure_rot'] = max(self.mins['stop_closure_pos'], ep), max(self.mins['stop_closure_rot'], er)
            if ep > CL_TOL['pos_m'] or er > CL_TOL['rot_deg']:
                self.n_emerg_closure += 1
            if self.clear_inject is not None:
                c = self.clear_inject(self, c)
            self.mins['n_sweep'] += 1
            for kk in KEYS:
                self.mins['stop'][kk] = min(self.mins['stop'][kk], c[kk])
            phi_i = float(x[9])
            bad = [kk for kk in KEYS if c[kk] < W.THR[kk]]
            if bad or not (-0.02 <= phi_i <= W.PHI_MAX + 0.02):
                return dict(extra=e, i=i, tag=tag, door='bound', clear_bad=bad, phi_deg=round(math.degrees(phi_i), 2), at={kk: round(float(c[kk]), 4) for kk in KEYS}), None
        if fb and not self._cs_on():
            self._fb_ok = bool(fb_ok)
            self._fb_info = dict(closure_pos_mm=round(fb_cl[0] * 1e3, 3), closure_rot_deg=round(fb_cl[1], 3), settled=all(self._prep_settled.values()), tag=tag)
        if tag == 'fallback':
            return None, None
        if self.sweep_inject is not None:
            inj = self.sweep_inject(self, tag)
            if inj is not None:
                return inj, None
        return None, None

    def _pool_sweep(self, q, u, extras, tag):
        items = self._sweep_prep(q, u, extras)
        return self._pool_eval(items, self._pool_submit(items), tag)

    def close_pool(self):
        if self._pool is not None and self._pool_owned:
            self._pool.close()

    def extend_reference(self, Xd_new, sigma_s, final=False):
        """r12：參考在 σ_s 起換成 Xd_new（同間隔 h；Xd_new[0] 應等於舊參考在 σ_s 的狀態）。σ 已過 σ_s ⇒ 拒絕（回 False）。"""
        i_s = int(round(sigma_s / self.h))
        if abs(i_s * self.h - sigma_s) > 1e-9 or sigma_s < self.sigma - 1e-9 or self.stop is not None or self.abort is not None:
            self.p2_log.append(dict(t=round(self.t, 3), ok=False, sigma=round(self.sigma, 3), sigma_s=round(sigma_s, 3), why='late_or_stopped'))
            return False
        i_s = min(i_s, len(self.Xd) - 1)
        jump = float(np.abs(np.asarray(Xd_new[0]) - self.Xd[i_s]).max())
        self.Xd = np.vstack([self.Xd[:i_s], np.asarray(Xd_new, float)])
        self._xd_ver += 1                                              # r17：參考版本（備援預測結果綁版本）
        self.sig_end = (len(self.Xd) - 1) * self.h
        if self.sigma < self.sig_end:
            self.t_ref_end, self._n_end = None, 0
        self.p2_open = not final
        self.p2_log.append(dict(t=round(self.t, 3), ok=True, sigma=round(self.sigma, 3), sigma_s=round(sigma_s, 3), final=bool(final), splice_jump=jump, sig_end=round(self.sig_end, 3)))
        return True

    def seam_check(self, Xd_new, sigma_s):
        """r13：接縫在執行層 ref／dref 語意下的核對。回 dict(ok, failed, values)。"""
        Xn = np.asarray(Xd_new, float)
        out = dict(ok=False, failed=[], values={})
        if not np.isfinite(Xn).all() or len(Xn) < 2:
            out['failed'].append('nonfinite_or_short')
            return out
        i_s = int(round(sigma_s / self.h))
        if abs(i_s * self.h - sigma_s) > 1e-9 or i_s < 1 or i_s > len(self.Xd) - 1:
            out['failed'].append('off_grid_or_range')
            return out
        a, b, c = self.Xd[i_s - 1], self.Xd[i_s], Xn[1]
        pj = float(np.abs(Xn[0] - b).max())
        v1, v2 = (b - a) / self.h, (c - Xn[0]) / self.h
        acc = (v2 - v1) / self.h
        th = float(b[2])
        Rt = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        vb2 = np.r_[Rt @ v2[:2], v2[2]]
        ab = np.abs(np.r_[Rt @ acc[:2], acc[2]])
        wheel = float(np.abs(W.WM @ vb2).max())
        sp = np.abs(v2[3:9] + W.TAU_SP * acc[3:9])
        vals = dict(pos_jump=pj, body_acc=[round(float(x), 4) for x in ab], wheel=round(wheel, 4), arm_rate=round(float(np.abs(v2[3:9]).max()), 4), arm_acc=round(float(np.abs(acc[3:9]).max()), 4),
                    setpoint_rate=round(float(sp.max()), 4), phi_rate=round(abs(float(v2[9])), 4), phi_acc=round(abs(float(acc[9])), 4))
        f = []
        if pj > 1e-6:
            f.append('pos_jump')
        if (ab > W.A_BODY + 1e-6).any():
            f.append('body_acc')
        if wheel > W.W_LIM + 1e-4:
            f.append('wheel')
        if float(np.abs(v2[3:9]).max()) > W.V_ARM + 1e-4:
            f.append('arm_rate')
        if float(np.abs(acc[3:9]).max()) > W.A_ARM + 1e-6:
            f.append('arm_acc')
        if float(sp.max()) > W.V_ARM + 1e-3:
            f.append('setpoint_rate')
        if abs(float(v2[9])) > W.PHI_RATE + 1e-4:
            f.append('phi_rate')
        if abs(float(acc[9])) > 5.0 + 1e-6:
            f.append('phi_acc')
        out.update(ok=not f, failed=f, values=vals)
        return out

    def stage_reference(self, Xd_new, sigma_s, final=False, ver=None):
        """r13：暫存新參考（不動本體參考）。回 (ok, 原因／接縫)。"""
        ver = self._last_ver + 1 if ver is None else int(ver)
        if ver <= self._last_ver:
            return False, dict(why='old_version', ver=ver, last=self._last_ver)
        if sigma_s < self.sigma - 1e-9:
            return False, dict(why='expired_seam', sigma=round(self.sigma, 3), sigma_s=round(sigma_s, 3))
        sc = self.seam_check(Xd_new, sigma_s)
        if not sc['ok']:
            return False, dict(why='seam', seam=sc)
        self.staged = dict(Xd=np.asarray(Xd_new, float).copy(), sigma_s=float(sigma_s), final=bool(final), ver=ver, seam=sc, t=round(self.t, 3))
        self._last_ver = ver
        return True, dict(seam=sc)

    def _light_copy(self):
        import copy
        big = ('step_log', 'pipe_log', 'cmd_hist', 'events', 'rec', 'tick_ms', 'solve_ms', 'sweep_ms', 'pool_ms', 'prep_ms', 'p2_log', 'fb_log', 'mon_ms', 'cl_ms', 'cand_ms', 'sn_ms', 'qp_log', 'pr_ts', 'pr_log', 'hold_log')
        saved = {k: getattr(self, k) for k in big if hasattr(self, k)}
        pool, stg, g2p = self._pool, self.staged, self.g2
        try:
            for k in saved:
                setattr(self, k, [])
            self._pool, self.staged, self.g2 = None, None, dict(self.g2, pool=0, cstop=None)
            csw, box, th, self._csw, self._cs_box, self._cs_th, self._cs_out = self._csw, self._cs_box, self._cs_th, None, None, None, None
            tmp = copy.deepcopy(self)
        finally:
            for k, v in saved.items():
                setattr(self, k, v)
            self._pool, self.staged, self.g2 = pool, stg, g2p
            self._csw, self._cs_box, self._cs_th = csw, box, th
            self._cs_out = None
        tmp._pool, tmp._pool_owned = pool, False
        tmp._csw, tmp._cs_box = csw, box                                # r17：試跑複本用同一個閉鏈一致停法（同工作程序、同停法物件；每次使用前各自設參考）
        tmp.g2 = dict(tmp.g2, pool=g2p['pool'], deadline_ms=None, cstop=csw)       # 試跑複本不是實時拍：期限只對本體的真實拍檢查
        return tmp

    def staged_step(self, t, q, phi_meas):
        """r13：每拍首（step 之前）呼叫。新參考最早影響命令的那一拍以複本試跑，全過才接上。回 None／'waiting'／決定紀錄。"""
        st = self.staged
        if st is None:
            return None
        if self.stop is not None or self.abort is not None:
            self.staged = None
            rec = dict(t=round(float(t), 3), ver=st['ver'], ok=False, why='exec_stopped')
            self.p2_log.append(rec)
            return rec
        if self.sigma > st['sigma_s'] + 1e-9:
            self.staged = None
            rec = dict(t=round(float(t), 3), ver=st['ver'], ok=False, why='expired_before_influence', sigma=round(self.sigma, 3), sigma_s=round(st['sigma_s'], 3))
            self.p2_log.append(rec)
            return rec
        reach = self.sigma + D_STEPS * PHYS + N * DT
        if reach < st['sigma_s'] - 1e-9 and self.sigma < self.sig_end - 1e-9:
            return 'waiting'
        w0 = time.perf_counter()
        tmp = self._light_copy()
        tmp.mins = dict(tmp.mins, stop_closure_pos=0.0, stop_closure_rot=0.0)
        okx = tmp.extend_reference(st['Xd'], st['sigma_s'], final=st['final'])
        fq_save, phi0_save = W.FINGER_Q, W.PHI0
        try:
            if okx:
                tmp.step(float(t), np.asarray(q, float).copy(), float(phi_meas))
        finally:
            W.FINGER_Q, W.PHI0 = fq_save, phi0_save
        dry = dict(extend_ok=okx, stop=None if tmp.stop is None else tmp.stop.get('why'), abort=None if tmp.abort is None else tmp.abort.get('why'), cl_fail=tmp._cl_fail,
                   stop_detail=tmp.stop, abort_detail=tmp.abort, stop_mode=tmp.stop_mode, fb_info=tmp._fb_info, xd_ver=tmp._xd_ver, pr_delta_abs_max=float(np.abs(tmp._pr_delta).max()), hold_active=tmp._hold is not None,   # N1：被拒細項（違反項、數值、取樣點、停法、參考版本）
                   stop_closure_pos_mm=round(tmp.mins['stop_closure_pos'] * 1e3, 3), stop_closure_rot_deg=round(tmp.mins['stop_closure_rot'], 3),
                   fb_ok=bool(tmp._fb_ok) if tmp._pool is not None else None, wall_ms=round((time.perf_counter() - w0) * 1e3, 1))
        valid = (okx and tmp.stop is None and tmp.abort is None and tmp._cl_fail is None and tmp.mins['stop_closure_pos'] <= CL_TOL['pos_m'] and tmp.mins['stop_closure_rot'] <= CL_TOL['rot_deg']
                 and (tmp._pool is None or tmp._fb_ok))
        self.staged = None
        if valid:
            self.extend_reference(st['Xd'], st['sigma_s'], final=st['final'])
        rec = dict(t=round(float(t), 3), ver=st['ver'], ok=bool(valid), why=None if valid else 'dry_run_rejected', dry=dry, sigma=round(self.sigma, 3), sigma_s=round(st['sigma_s'], 3))
        self.p2_log.append(rec)
        return rec

    def warmup(self, q, phi_meas):
        """r10：交接前預熱（接近階段即可做，不在控制拍內）。複製一份控制器（不含計算池）以起始狀態空跑一拍，
        讓求解器、閉鏈、淨空、停止預測的第一次呼叫成本發生在交接前；本控制器狀態完全不動（複本丟棄）。回複本那一拍的耗時 ms。"""
        import copy
        pool, self._pool = self._pool, None
        g2p, self.g2 = self.g2, dict(self.g2, pool=0, cstop=None)
        csw, box, self._csw, self._cs_box = self._csw, self._cs_box, None, [None]
        try:
            tmp = copy.deepcopy(self)
        finally:
            self._pool, self.g2, self._csw, self._cs_box = pool, g2p, csw, box
        tmp._pool, tmp._pool_owned = None, False
        tmp.g2 = dict(tmp.g2, pool=0, deadline_ms=None)
        fq_save, phi0_save = W.FINGER_Q, W.PHI0
        t0 = time.perf_counter()
        try:
            tmp.step(0.0, np.asarray(q, float).copy(), float(phi_meas))
        finally:
            W.FINGER_Q, W.PHI0 = fq_save, phi0_save
        return (time.perf_counter() - t0) * 1e3

    def _cs_params(self):
        return dict(ident=self._ident, bias=np.asarray(self.cfg.arm_model.bias, float), T_WH_bind=self.T_WH_bind, phi_bind=self.phi_bind, T_HE=self.T_HE, off_cl0=self.off_cl0, G0=self.G0,
                    joint_kp=self.joint_kp, h=self.h, Xd=self.Xd, bias0=self._bias0.copy(), gravity3=bool(self.g2.get('gravity3')), fric_ff=self.g2['fric_ff'], lim_lo=self.lim_lo, lim_hi=self.lim_hi, q_lo=np.asarray(LITE6_SAFE.lower, float), q_hi=np.asarray(LITE6_SAFE.upper, float))

    def _cs_on(self):
        return self._csw is not None and self.bound and self.off_cl0 is not None and self._pool is not None

    def _cs_obj(self):
        """主程序的停法物件（與工作程序同參數）；每次取用都設成本控制器當下的參考（P2 換段後不沿用舊段）。"""
        if self._cs_box[0] is None:
            import door_cstop as DCS
            self._cs_box[0] = DCS.CStop(self._cs_params())
            self._csw.init(self._cs_params())
        self._cs_box[0].set_ref(self.Xd)
        self._cs_box[0].set_delta(self._pr_delta)                    # r22：主程序停法物件用當下的 δ（工作程序由狀態快照帶入）
        return self._cs_box[0]

    def _pr_commit(self):
        """候選命令提交時一併提交 δ：δ 增量化成命令分量，走與手臂命令相同的在途管線。"""
        self._pr_commit_tick = True
        if self._pr_cand is None:
            self._pr_u = np.zeros(6)
            return
        self._pr_u = (self._pr_cand - self._pr_delta) / self.cfg.dt
        self._pr_delta = self._pr_cand.copy()
        inf_ = self._pr_inf
        self.pr_log.append((round(self.t, 3), float(np.abs(self._pr_delta).max()), inf_.get('Fc_norm', float('nan')), inf_.get('Mc_norm', float('nan')), inf_.get('lam_n', float('nan')), bool(inf_.get('sat', False)), bool(inf_.get('updated', False))))

    def _pr_use(self):
        """候選命令用的 δ：有候選用候選，否則用已提交值。"""
        return self._pr_cand if self._pr_cand is not None else self._pr_delta

    def _pr_model(self, d):
        """內部受控對象模型的穩態偏置＝bias0−α·δ。這是「剛性閉鏈近似」（綁定下手臂不隨設定點偏置移動），不是已辨識的受控對象偏置；bias0 與 δ 各自獨立保存。"""
        self.cfg.arm_model.bias = self._bias0 - np.asarray(self.cfg.arm_model.alpha, float) * np.asarray(d, float)

    def _cs_state(self, q, pr_delta=None, **kw):
        seed = self.q_cl_prev if self.q_cl_prev is not None else np.asarray(q[3:], float)
        return dict(q=np.asarray(q, float), s=self.s, pipe=self.pipe, u_prev=self.u_last, sigma=self.sigma, g=self.g, seed=seed, Xd=self.Xd, se_end=self._se_end, base_hold=self._hold, pr_delta=(self._pr_delta if pr_delta is None else np.asarray(pr_delta, float)).copy(), **kw)

    def _cs_bg(self, ver, out):
        """背景執行緒（主程序求解期間）：等向前模擬 → 取樣構型送計算池核淨空 → 收結果。只寫自己這次請求的 out（不寫控制器狀態）。
        status：ok／infeasible（模型判此停法不可行）／missing（結果缺失、逾時、程序或計算池失效）。"""
        try:
            r = self._csw.collect(ver, 0.2)
            if r is None:
                out.update(status='missing', why='sim_timeout_or_dead')
            elif not r['ok'] and str(r['why']).startswith('exception'):
                out.update(status='missing', why=r['why'])
            elif not r['ok']:
                out.update(status='infeasible', why=r['why'])
            else:
                with _pool_lock(self._pool):
                    h_ = self._pool.submit(self.T_WH_bind, W.FINGER_Q, W.PHI0, r['X'])
                    res, why, _ = self._pool.collect(h_, POOL_TIMEOUT_S)
                if res is None:
                    out.update(status='missing', why='pool_' + str(why))
                else:
                    out.update(status='ok', res=res, r=r)
        except Exception as e:                                         # noqa: BLE001
            out.update(status='missing', why='bg_exception:' + repr(e)[:120])
        out['done'] = True

    def _cs_submit(self, q):
        """現在停（實際會用的閉鏈一致停法）的向前模擬：送獨立程序＋背景執行緒，與求解並行。"""
        import threading
        if self._cs_th is not None and self._cs_th.is_alive():         # 作廢的上一筆仍在用計算池：先等它結束（同一時間只一個執行緒用池）
            self._cs_th.join(0.3)
        self._cs_obj()
        out = dict(done=False, xd_ver=self._xd_ver, t=self.t)           # 每次請求自己的結果盒：遲到的執行緒只會寫到舊盒，不覆寫下一拍
        ver_ = self._csw.submit(self._cs_state(q))
        th = threading.Thread(target=self._cs_bg, args=(ver_, out), daemon=True)
        th.start()
        self._cs_th, self._cs_out = th, out

    def _cs_judge(self, r, res, tag):
        """回 (淨空違反 dict 或 None, 備援可用, 資訊)。"""
        mn = {k: min(float(c[k]) for c in res) for k in KEYS}
        for k in KEYS:
            self.mins['stop'][k] = min(self.mins['stop'][k], mn[k])
        self.mins['n_sweep'] += len(res)
        cp, cr = r['closure']
        if self.fb_closure_inject is not None:
            cp, cr = self.fb_closure_inject(self, cp, cr)
        self.mins['stop_closure_pos'], self.mins['stop_closure_rot'] = max(self.mins['stop_closure_pos'], cp), max(self.mins['stop_closure_rot'], cr)
        bad = [k for k in KEYS if mn[k] < W.THR[k]]
        ok = bool((not bad or _lenient_shell_ok(dict(clear_bad=bad, at=mn))) and cp <= CL_TOL['pos_m'] and cr <= CL_TOL['rot_deg'])
        self._cs_who = {k: str((min(res, key=lambda c_: float(c_[k])).get('who') or {}).get(k)) for k in bad} if bad else None   # 最近障礙名稱（Codex 184816：base 是門扇與環境的最小值）   # 寬鬆展示：只外殼—橫桿且 ≥3 mm 時不算失敗（記錄另見 lenient_log）
        info = dict(law='cstop', tag=tag, n_ticks=r['n_ticks'], sim_ms=round(r['sim_ms'], 1), min_clear={k: round(v, 4) for k, v in mn.items()}, closure_pos_mm=round(cp * 1e3, 3), closure_rot_deg=round(cr, 3), n_samples=len(res), xd_ver=self._xd_ver)
        return (dict(tag=tag, clear_bad=bad, at={k: round(v, 4) for k, v in mn.items()}, who=self._cs_who) if bad else None), ok, info

    def _cs_collect(self):
        """求解後取背景結果。回 (status, 淨空違反或 None)；status＝ok／infeasible／missing。並設定本拍備援是否可用。
        missing（逾時、例外、程序或計算池失效、參考版本不符）不是「核過」：呼叫端須中止。"""
        th, out = self._cs_th, self._cs_out
        self._cs_th, self._cs_out = None, None
        if th is None or out is None:
            self._fb_ok, self._fb_info = False, dict(law='cstop', status='missing', why='not_submitted')
            return 'missing', None
        th.join(0.3)
        if self.cs_inject is not None:
            inj = self.cs_inject(self, 'stop_now')
            if inj is not None:
                out = dict(done=True, xd_ver=out['xd_ver'], status=inj, why='injected')
        if th.is_alive() or not out.get('done'):
            self._fb_ok, self._fb_info = False, dict(law='cstop', status='missing', why='bg_timeout')
            return 'missing', None
        if out['xd_ver'] != self._xd_ver:
            self._fb_ok, self._fb_info = False, dict(law='cstop', status='missing', why='ref_version_mismatch')
            return 'missing', None
        if out['status'] != 'ok':
            self._fb_ok, self._fb_info = False, dict(law='cstop', status=out['status'], why=out.get('why'))
            return out['status'], None
        bad, ok, info = self._cs_judge(out['r'], out['res'], 'stop_now_cstop')
        self._sn_tr = out['r'].get('tr')
        self._fb_ok, self._fb_info = ok, info
        return 'ok', bad

    def _cs_cand_submit(self, q, u):
        """候選提交前核對（取代各軸減速的 (1,2) 掃掠）的模擬請求：提交 u 一拍後以閉鏈一致停法停下。用第二個工作程序（不等現在停那一個）。
        只核 hold＝1：同步執行下，下一拍若晚到，備援命令就在該拍取代候選，候選命令恰好作用一拍。
        原各軸減速的 (1,2) 中「同一命令再多留一拍」在閉鏈手臂下不是會發生的情形（手臂命令每拍依閉鏈重算；多留一拍即破壞閉鏈，假世界起步即 arm_rate 不可行）。"""
        self._cs_obj()
        w = 1 if self._csw.n > 1 else 0
        if not self.stream_cand:
            return w, self._csw.submit(self._cs_state(q, pr_delta=self._pr_use(), u_first=np.asarray(u, float), hold=1, se_first=self._se_cand, rec=bool(self.rec_cand)), w=w), self._xd_ver
        import threading                                               # A1（Codex 20261117_003351）：模擬分批回傳取樣點，轉送執行緒一到就送淨空池；放行仍等「整段模擬完成＋全部批次齊全」
        ver = self._csw.submit(self._cs_state(q, pr_delta=self._pr_use(), u_first=np.asarray(u, float), hold=1, se_first=self._se_cand, rec=bool(self.rec_cand), stream=True, stream_chunk=STREAM_CHUNK), w=w)
        box = dict(done=False, ver=ver, xv=self._xd_ver, t_tick=self.t)
        th = threading.Thread(target=self._cand_stream_fw, args=(w, ver, box, self.T_WH_bind, W.FINGER_Q, W.PHI0), daemon=True)
        th.start()
        return w, ver, self._xd_ver, box, th

    def _cand_stream_fw(self, w, ver, box, T_WH, fq, phi0):
        """轉送執行緒：只寫自己的 box。批次以 (拍、參考版本、模擬版本、起始索引) 綁定；池的版本號另外保證結果不混用。工作程序分兩組輪流（不相交、可同時在途）。"""
        try:
            pool = self._pool
            n = pool.n
            groups = [list(range(0, n // 2)), list(range(n // 2, n))] if n >= 2 else [list(range(n))]
            inflight = [None] * len(groups)                            # (i0, 個數, 池 handle)
            got, gi, nsent = {}, [0], [0]
            t0 = time.perf_counter()
            fail = []

            def _drain(g):
                it = inflight[g]
                if it is None:
                    return
                inflight[g] = None
                res, why, _ = pool.collect(it[2], POOL_TIMEOUT_S)
                if res is None or len(res) != it[1]:
                    fail.append('pool_' + str(why))
                else:
                    got[it[0]] = res

            lock, held = _pool_lock(pool), [False]
            box['stage'] = [('start', round((time.perf_counter() - t0) * 1e3, 1))]   # 診斷：各階段時間（卡住時也看得到停在哪）

            def _send(i0, Xc):
                if i0 != nsent[0]:                                     # 批次須連續（起始索引＝已送數）；否則視為結果缺失
                    fail.append('batch_gap_%d_%d' % (i0, nsent[0]))
                    return
                if not held[0]:                                        # 第一次要用池時才取鎖（現在停背景執行緒若仍在用池就等它；之後持有到全部收齊）
                    box['stage'].append(('lock_wait', round((time.perf_counter() - t0) * 1e3, 1)))
                    lock.acquire()
                    held[0] = True
                    box['stage'].append(('lock_got', round((time.perf_counter() - t0) * 1e3, 1)))
                g = gi[0] % len(groups)
                gi[0] += 1
                _drain(g)
                inflight[g] = (int(i0), len(Xc), pool.submit(T_WH, fq, phi0, Xc, workers=groups[g]))
                nsent[0] += len(Xc)

            try:
                r = self._csw.collect_stream(ver, 0.2, w=w, on_part=_send)
                box['stage'].append(('cs_done' if r is not None else 'cs_none', round((time.perf_counter() - t0) * 1e3, 1)))
                if r is not None and r.get('X') is not None and r['ok']:
                    Xf = np.asarray(r['X'], float)
                    if nsent[0] < len(Xf):
                        _send(nsent[0], Xf[nsent[0]:])
                for g in range(len(groups)):
                    _drain(g)
                box['stage'].append(('drained', round((time.perf_counter() - t0) * 1e3, 1), list(fail)))
            finally:
                if held[0]:
                    lock.release()
            res = None
            if r is not None and r['ok'] and not fail:
                keys = sorted(got)
                res = [c for k in keys for c in got[k]]
                if len(res) != len(r['X']) or keys[0] != 0:
                    fail.append('coverage')
                    res = None
            box.update(r=r, res=res, fail=fail, n_batches=gi[0], fw_ms=round((time.perf_counter() - t0) * 1e3, 2))
        except Exception as e:                                         # noqa: BLE001
            box.update(r=None, res=None, fail=['fw_exception:' + repr(e)[:150]])
        box['done'] = True

    def _solve_cand(self, q, phi_meas):
        """求解；閉鏈一致停法啟用時，本拍命令一確定（閉鏈手臂前三個節點算完）就把候選核對的模擬送出，與其餘求解工作重疊。
        回 (u, 候選請求或 None)。提早送出的命令與最後命令不同（>1e-9）或沒送 ⇒ 以最後命令重送。"""
        on = self._cs_on()
        self._hc_e = None
        self._pr_model(self._pr_use())                               # 候選求解與其預測用候選 δ 的模型；未提交則於拍末回復
        if on:
            def _cb(ue, q=q):
                self._hc_e = (self._cs_cand_submit(q, ue), np.asarray(ue, float).copy())
            self._early_cb = _cb
        try:
            u = self._solve(q, phi_meas)
        finally:
            self._early_cb = None
        he, self._hc_e = self._hc_e, None
        if not on or u is None:
            return u, None
        if he is not None and float(np.abs(he[1] - u).max()) <= 1e-9:
            return u, he[0]
        return u, self._cs_cand_submit(q, u)

    def _cs_cand_finish(self, hc, tag):
        """回 (status, 淨空違反或 None, 資訊)。infeasible＝提交後沒有可行的閉鏈一致停法 ⇒ 候選不得提交；missing＝結果缺失（呼叫端依「掃掠不可用」處理）。
        須在現在停的背景執行緒結束後呼叫（計算池同一時間只由一個執行緒使用）。"""
        if len(hc) == 5:                                               # A1 串流：等轉送執行緒（模擬完成＋全部批次結果齊全）
            w, ver, xv, box, th = hc
            _wc = time.perf_counter()
            th.join(0.5)
            r = box.get('r') if box.get('done') else None
            self._cand_diag = dict(stream=True, stage=list(box.get('stage') or []), join_wait_ms=round((time.perf_counter() - _wc) * 1e3, 2), fw_ms=box.get('fw_ms'), n_batches=box.get('n_batches'), fail=box.get('fail'), sim_ms=None if r is None else round(float(r.get('sim_ms') or 0.0), 2), n_ticks=None if r is None else r.get('n_ticks'))
            if self.cs_inject is not None:
                inj = self.cs_inject(self, tag)
                if inj is not None:
                    return inj, None, dict(why='injected')
            if not box.get('done') or r is None or xv != self._xd_ver or ((not r['ok']) and str(r['why']).startswith('exception')):
                return 'missing', None, dict(why='stream_not_done' if not box.get('done') else (None if r is None else ('ref_version_mismatch' if xv != self._xd_ver else r['why'])))
            if not r['ok']:
                return 'infeasible', None, dict(why=r['why'])
            if box.get('res') is None:
                return 'missing', None, dict(why='stream_' + ','.join(box.get('fail') or ['no_result']))
            res = box['res']
            self.pool_ms.append(dict(stream_fw=box.get('fw_ms')))
            bad, ok, info = self._cs_judge(r, res, tag)
            self._cand_tr = r.get('tr')
            self._cand_out = r.get('out') if (ok and (bad is None or _lenient_shell_ok(bad))) else None
            self._cand_X, self._cand_props = (r.get('X'), r.get('props')) if (ok and (bad is None or _lenient_shell_ok(bad))) else (None, None)
            if not ok and bad is None:
                self.n_emerg_closure += 1
                return 'infeasible', None, dict(why='closure_over_tol', closure_pos_mm=info['closure_pos_mm'], closure_rot_deg=info['closure_rot_deg'])
            return 'ok', bad, info
        w, ver, xv = hc
        _wc = time.perf_counter()
        r = self._csw.collect(ver, 0.2, w=w)
        self._cand_diag = dict(collect_wait_ms=round((time.perf_counter() - _wc) * 1e3, 2), sim_ms=None if r is None else round(float(r.get('sim_ms') or 0.0), 2), n_ticks=None if r is None else r.get('n_ticks'))   # 只記錄
        if self.cs_inject is not None:
            inj = self.cs_inject(self, tag)
            if inj is not None:
                return inj, None, dict(why='injected')
        if r is None or xv != self._xd_ver or ((not r['ok']) and str(r['why']).startswith('exception')):
            return 'missing', None, dict(why=None if r is None else ('ref_version_mismatch' if xv != self._xd_ver else r['why']))
        if not r['ok']:
            return 'infeasible', None, dict(why=r['why'])
        _wp = time.perf_counter()
        with _pool_lock(self._pool):
            res, why, tms = self._pool.collect(self._pool.submit(self.T_WH_bind, W.FINGER_Q, W.PHI0, r['X']), POOL_TIMEOUT_S)
        self.pool_ms.append(tms)
        self._cand_diag.update(pool_wall_ms=round((time.perf_counter() - _wp) * 1e3, 2), n_samples=len(r['X']))
        if res is None:
            return 'missing', None, dict(why='pool_' + str(why))
        bad, ok, info = self._cs_judge(r, res, tag)
        self._cand_tr = r.get('tr')                                    # 診斷用：候選核對的逐拍預測
        self._cand_out = r.get('out') if (ok and (bad is None or _lenient_shell_ok(bad))) else None   # 只有核過的停止模擬才可當尾段
        self._cand_X, self._cand_props = (r.get('X'), r.get('props')) if (ok and (bad is None or _lenient_shell_ok(bad))) else (None, None)
        if not ok and bad is None:                                     # 提交後的停止預測閉鏈超限 ⇒ 提交後沒有合格備援 ⇒ 候選不得提交
            self.n_emerg_closure += 1
            return 'infeasible', None, dict(why='closure_over_tol', closure_pos_mm=info['closure_pos_mm'], closure_rot_deg=info['closure_rot_deg'])
        return 'ok', bad, info

    def _deadline_stop(self, elapsed_ms, tag):
        """r10：控制期限已過 ⇒ 不提交本拍候選。正常段：本拍現在停已驗過 ⇒ 有界減速停法（settled）；沒驗到 ⇒ 模擬中止。
        路徑停止中：上一拍提交的候選已含「下一拍起減速」的預測核對 ⇒ 直接轉有界減速。"""
        self.n_deadline_miss += 1
        self.fb_log.append(dict(t=round(float(self.t), 3), tag=tag, fb_ok=self._fb_ok, fb=self._fb_info, elapsed_ms=round(elapsed_ms, 1), se_end_committed=round(self._se_end, 4), sigma=round(self.sigma, 4), g=round(self.g, 3),
                                sigma_plus_delay_g=round(self.sigma + D_STEPS * PHYS * self.g, 4)))
        mode_fb = 'cstop' if self._cs_on() else 'settled'
        if self.stop is None:
            if self._sn_verified and self._fb_ok:
                self.stop = dict(t=round(float(self.t), 3), why='tick_deadline_miss', sigma=round(self.sigma, 3), mode=mode_fb, elapsed_ms=round(elapsed_ms, 1), tag=tag, fb=self._fb_info)
                self.t_path_stop = self.t
                self.stop_mode = mode_fb
                self._ev('stop', why='tick_deadline_miss', mode=mode_fb, elapsed_ms=round(elapsed_ms, 1), tag=tag, fb=self._fb_info)
            else:
                self._stop('tick_deadline_miss_unverified', elapsed_ms=round(elapsed_ms, 1), tag=tag)
        elif self.stop_mode == 'path':
            if self._fb_ok:
                self.stop_mode = mode_fb
                self._ev('path_deadline_to_settled', elapsed_ms=round(elapsed_ms, 1), tag=tag, fb=self._fb_info)
            else:
                self._stop('path_deadline_miss_unverified', elapsed_ms=round(elapsed_ms, 1), tag=tag, fb=self._fb_info)

    # ---- 求解（同一個全身問題）
    def _nav_obs(self, q):
        """導航前端障礙物（相對擷取時機器人位姿）→ 世界；過期 ⇒ 0.5 s 內沿用上一份，否則 None。"""
        ni = getattr(self, 'nav_in', None)
        if ni is not None and not ni.get('stale'):                    # 含導航到點後的 halted 訊息（障礙物照原規則持續提供）
            c0, s0 = math.cos(q[2]), math.sin(q[2]); out = []
            for o_ in ni.get('obs', []):
                d_ = dict(x=float(q[0] + c0 * o_[0] - s0 * o_[1]), y=float(q[1] + s0 * o_[0] + c0 * o_[1]), radius=float(o_[2]), vx=float(c0 * o_[3] - s0 * o_[4]), vy=float(s0 * o_[3] + c0 * o_[4]))
                if math.isfinite(o_[5]):
                    d_['margin'] = float(o_[5])
                if o_[6] > 0.5:
                    d_['static'] = True
                out.append(d_)
            ap_ = getattr(self, 'app_prof', None)
            if ap_ is not None and ap_.get('door') is not None:          # 開門中：門扇（依實測門角）附近障礙點不進 CBF（同抓持段）
                import door_motm_grasp_ctrl as _G
                out = _G.door_filter(out, ap_, getattr(self, '_phi_meas_last', None))
            self._obs_last = (self.t, out)
            return out
        ol = getattr(self, '_obs_last', None)
        return ol[1] if (ol is not None and self.t - ol[0] <= 0.5) else None

    def _gov_factor(self):
        """各淨空項以自身門檻與預警帶正規化到 [0,1]，取最小（不同單位不直接比較）。資料＝最近一次通過的候選停止路徑，與本拍現在停核對（若有）。"""
        ms = [m for m in (self._gov_m, (self._fb_info or {}).get('min_clear') if isinstance(self._fb_info, dict) else None) if m]
        if not ms:
            return 1.0
        f = 1.0
        for m in ms:
            for k in KEYS:
                if k in m:
                    f = min(f, float(np.clip((float(m[k]) - W.THR[k]) / G_GOV_BAND[k], 0.0, 1.0)))
        return f

    def _first_fail_snap(self, q, u, stc, b_, infc):
        """第一個失敗快照（Codex 191426）：含「QP 解出、但停止尾段被拒」。分類：g 已在下限仍被拒 ⇒ 幾何／模型不相容嫌疑；否則 ⇒ 太快嫌疑。"""
        self.first_fail = dict(t=round(float(self.t), 3), kind='qp_solved_tail_rejected', status=stc, bad=b_, info=infc, sigma=round(self.sigma, 5), g=round(self.g, 4),
                               phi_ref_deg=round(math.degrees(float(self.ref(self.sigma)[9])), 3), phi_meas_deg=(round(math.degrees(self._phi_meas_last), 3) if getattr(self, "_phi_meas_last", None) is not None else None),
                               gov_margins=self._gov_m, q=[round(float(x), 6) for x in q], u=[round(float(x), 6) for x in u],
                               cls=('geometry_or_model_suspect' if self.g <= G_GOV_MIN + 1e-6 else 'too_fast_suspect'))
        self._ev('first_fail_snapshot', **{k: v for k, v in self.first_fail.items() if k in ('kind', 'status', 'g', 'cls')}, t_snap=self.first_fail['t'])
        d_ = os.environ.get('DOOR_QP_DUMP_DIR')
        if d_:
            try:
                import json as _j
                os.makedirs(d_, exist_ok=True)
                with open(os.path.join(d_, 'first_fail_t%07.3f.json' % self.t), 'w') as f_:
                    _j.dump(self.first_fail, f_, default=str, indent=1)
            except Exception:
                pass

    def _dump_qp_fail(self, ck, qp, sp, sig_eff, g, X, Ts):
        """失敗拍可重現資料（Codex 20261117_190256）：P,q,A,l,u、約束類別、GI 原因、OSQP 狀態／迭代／殘差、管線前推後狀態、σ、g、門角參考序列。寫到 DOOR_QP_DUMP_DIR。"""
        d_ = os.environ.get('DOOR_QP_DUMP_DIR')
        if not d_:
            return
        try:
            os.makedirs(d_, exist_ok=True)
            qi = dict(ck.get('qp') or {})
            dump = qi.pop('dump', None) or {}
            marks = dump.pop('marks', [])
            phis = [float(self.ref(sig_eff + k_ * DT * g)[9]) for k_ in range(N + 1)]
            meta = dict(t=float(self.t), fallback=str(ck.get('fallback')), coupled_status=str(ck.get('coupled_status')), qp_info={k_: (v_ if isinstance(v_, (int, float, str, type(None))) else str(v_)) for k_, v_ in qi.items()},
                        marks=[[str(a_), int(b_), int(c_)] for a_, b_, c_ in marks], sigma_eff=float(sig_eff), g=float(g), phi_ref=phis)
            np.savez_compressed(os.path.join(d_, 'qpfail_t%07.3f.npz' % self.t), qp=np.asarray(qp, float), sp=np.asarray(sp, float), u_prev=np.asarray(self.u_prev, float),
                                X_ref=np.array([np.asarray(Xk, float) for Xk in X]), T_seq=np.array([np.asarray(T_, float) for T_ in Ts]), meta=np.array(json.dumps(meta, default=str)),
                                **{k_: np.asarray(v_) for k_, v_ in dump.items() if k_ != 'gi_why'})
            self.qp_dumps = getattr(self, 'qp_dumps', 0) + 1
        except Exception as e_:                                       # noqa: BLE001
            self.qp_dump_err = repr(e_)[:200]

    def _solve(self, q, phi_meas):
        w0 = time.perf_counter()
        self._phi_meas_last = float(phi_meas)
        cfg = self.cfg
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        g = self.g
        sig_eff = self.sigma + D_STEPS * PHYS * g
        self._se_cand = min(sig_eff + DT * g, self.sig_end)           # r17：本拍候選命令的參考終點（提交時才成為 _se_end）
        X, xi, Ts, qr = [], [], [], []
        for k in range(N + 1):
            sk = sig_eff + k * DT * g
            x, dx = self.ref(sk), self.dref(sk)
            c, s_ = math.cos(x[2]), math.sin(x[2])
            X.append(from_xytheta(float(x[0]), float(x[1]), float(x[2])))
            xi.append([g * (c * dx[0] + s_ * dx[1]), g * (-s_ * dx[0] + c * dx[1]), g * dx[2]])
            Ts.append(self.handle_T(float(x[9])) @ self.T_HE)        # 預測門角 φ̂_k＝φ_ref(σ_k)
            qr.append((x[3:9], g * dx[3:9]))
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        al = np.asarray(cfg.arm_model.alpha, float)
        tau, off = PHYS / al, np.asarray(cfg.arm_model.bias, float) / al
        va, sk_, u_nom = float(B.W_ARM['v_arm']), sp.copy(), []
        for k in range(N):
            s_ref = qr[k + 1][0] + tau * qr[k + 1][1] - off
            u = np.clip((s_ref - sk_) / DT, -va, va)
            u_nom.append(u)
            sk_ = sk_ + DT * u
        arm = dict(u_arm_nom=np.array(u_nom), mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_NEAR,
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=False, qp_solver='gi', **B.W_ARM)
        if self.arm_mode == 'qp' and CL_ROWS is not None:              # 主線修正 A：閉鏈位置＋姿態硬列進同一全身 QP（目標 T_seq＝T_WH(φ_ref)·T_HE；底盤與手臂耦合同列）
            arm['contact_tube'] = dict(CL_ROWS)
        if BASE_KEEP is not None:                                      # 底盤—門扇淨空下界（Codex 184816）：門扇箱體依參考門角序列 φ_ref(σ_k)（同 clear_door_fast 的門扇幾何）
            sh_, leaf_ = W._static(); bx_ = []
            for k_ in range(N):
                ph_ = float(self.ref(sig_eff + (k_ + 1) * DT * g)[9]); Rp_ = W.Rz4(-ph_)[:3, :3]
                bx_.append((Rp_ @ np.asarray(leaf_[1], float), Rp_, np.asarray(leaf_[3], float)))
            arm['base_keepout'] = dict(dmin=BASE_KEEP, n=N, boxes=bx_, RB=float(W.DOC.RB), act=0.30)
        w_pre = time.perf_counter()
        obs_ = self._nav_obs(q) if getattr(self, 'unified', False) else None   # 單一控制器：導航 CBF 障礙物進開門段同一 QP
        if getattr(self, 'unified', False) and obs_ is None:
            return None                                                # 障礙物資料過期 ⇒ 求解失敗處置（停止），不在無障礙資料下續算
        if self.g2.get('deadline_ms') is not None and getattr(self, '_w_tick', None) is not None:   # 整個 QP 流程共用時間預算（整拍期限扣候選核對與通訊保留；Codex 第九趟）
            arm['t_deadline'] = self._w_tick + (float(self.g2['deadline_ms']) - QP_RESERVE_MS) / 1e3
        try:
            r = self.mpc.solve(X_now=from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs_)
        except GR.QPTimeBudget as e_:
            self._qp_tb = dict(t=round(self.t, 3), why=str(e_)[:120], elapsed_ms=round((time.perf_counter() - self._w_tick) * 1e3, 1))
            self.qp_tb_log = getattr(self, 'qp_tb_log', []) + [self._qp_tb]
            if getattr(e_, 'dump', None) is not None and os.environ.get('DOOR_QP_DUMP_DIR'):   # 逾時拍也存當拍矩陣與分項（背景寫檔）
                import threading
                def _wr(sd=e_.dump, mt=dict(self._qp_tb, sigma=round(float(self.sigma), 5)), d_=os.environ['DOOR_QP_DUMP_DIR']):
                    try:
                        os.makedirs(d_, exist_ok=True)
                        np.savez(os.path.join(d_, 'timebudget_t%07.3f.npz' % mt['t']), P=sd['P'].toarray(), q=sd['q'], A=sd['A'].toarray(), l=sd['l'], u=sd['u'], meta=json.dumps(dict(mt, info=sd.get('info'), marks=sd.get('marks')), default=str))
                    except Exception:                                 # noqa: BLE001
                        pass
                threading.Thread(target=_wr, daemon=True).start()
            return None
        except ValueError as e_:                                       # 底盤名目點在門扇箱內 ⇒ 明確不可行 ⇒ 求解失敗處置（原停止）
            if 'base_keepout' not in str(e_):
                raise
            self.keepout_fail = getattr(self, 'keepout_fail', []) + [dict(t=round(self.t, 3), err=str(e_)[:120])]
            return None
        self.solve_ms.append((time.perf_counter() - w0) * 1e3)
        qi_ = (r.arm_check or {}).get('qp') or {}
        if qi_.get('slow_dump') is not None and os.environ.get('DOOR_QP_DUMP_DIR'):   # 慢成功拍：背景執行緒寫檔（不佔本拍計時路徑）
            import threading
            sd_ = qi_.pop('slow_dump')
            meta_ = dict(t=round(float(self.t), 3), sigma=round(float(self.sigma), 5), gi_ms=qi_.get('gi_ms'), gi_steps=qi_.get('gi_steps'), gi_why=qi_.get('gi_why'), osqp_ms=qi_.get('osqp_ms'), asm_ms=qi_.get('asm_ms'),
                         osqp_status=qi_.get('osqp_status'), osqp_iter=qi_.get('osqp_iter'), solve_ms=round((time.perf_counter() - w0) * 1e3, 2), marks=sd_.get('marks'))
            def _wr(sd=sd_, mt=meta_, d_=os.environ['DOOR_QP_DUMP_DIR']):
                try:
                    os.makedirs(d_, exist_ok=True)
                    np.savez(os.path.join(d_, 'slow_t%07.3f.npz' % mt['t']), P=sd['P'].toarray(), q=sd['q'], A=sd['A'].toarray(), l=sd['l'], u=sd['u'], meta=json.dumps(mt, default=str))
                except Exception:                                     # noqa: BLE001
                    pass
            threading.Thread(target=_wr, daemon=True).start()
        elif 'slow_dump' in qi_:
            qi_.pop('slow_dump')
        f_ = lambda v_: float('nan') if v_ is None else float(v_)       # noqa: E731
        self.qp_log.append((float(self.t), float(self.stop is not None), (w_pre - w0) * 1e3, (time.perf_counter() - w_pre) * 1e3, f_(qi_.get('asm_ms')), f_(qi_.get('gi_ms')), f_(qi_.get('osqp_ms')), 0.0 if qi_.get('solver') == 'gi' else 1.0))
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None:
            fb_ok_ = (OPEN_FB_TICKS > 0 and ck.get('fallback') == 'base_postprocess_changed' and not ck.get('failed') and ck.get('arm_decel_valid')
                      and r.u_arm_seq is not None and np.isfinite(np.asarray(r.u_opt, float)).all() and np.isfinite(np.asarray(r.u_arm_seq[0], float)).all())
            if fb_ok_:                                                 # 11/25 第 132、143 趟開門中「底盤後處理改動」一拍即停整趟；同 W-GMPC 其他控制器（wg_result，Isaac 93）：連續 ≤OPEN_FB_TICKS 拍照求解器備援命令（hold 底盤＋已驗證手臂減速），下一拍重解；候選仍經核對
                self._open_fb_run = getattr(self, '_open_fb_run', 0) + 1
                if self._open_fb_run <= OPEN_FB_TICKS:
                    self.n_open_fb = getattr(self, 'n_open_fb', 0) + 1
                    if hasattr(self, '_ev'):
                        self._ev('open_fallback_tick', why='base_postprocess_changed', run=self._open_fb_run, n_total=self.n_open_fb, sigma=round(float(sig_eff), 4))
                    fb_ok_ = 'go'
            if fb_ok_ != 'go':
                self._dump_qp_fail(ck, qp, sp, sig_eff, g, X, Ts)
                return None
        else:
            self._open_fb_run = 0
        u = np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]          # 備援拍：底盤＝求解器備援（hold 底盤解），閉鏈模式下手臂仍由下方閉鏈一致式依此底盤重算
        if self.arm_mode == 'closure':
            w_cl = time.perf_counter()
            ua = self._closure_arm(qp, sp, r, np.array(xi), sig_eff, g, tau, va)
            self.cl_ms.append((time.perf_counter() - w_cl) * 1e3)
            if ua is None:
                return None
            u = np.r_[np.asarray(r.u_opt, float), ua]
        return u if np.isfinite(u).all() else None

    # ---- A：閉鏈一致設定點（Codex 20261112_234431／20261113_000935）
    def _ik_closure(self, b, phi, seed):
        """固定 T_HE：給底盤位姿 b 與門角 φ，解手臂 q 使 TCP＝把手(φ)·T_HE；以前一解為種子（連續分支）。回 (q, 殘差, σ_min)。"""
        Tg = self.handle_T(float(phi)) @ self.T_HE
        if self.g2['ik_fc']:
            return DFK.ik_closure_fc(Tg, np.asarray(b, float), seed)
        q = np.asarray(seed, float).copy()
        res = np.inf
        for _ in range(30):
            q9 = np.r_[b, q]
            Tc = K.fk(q9, 'link_tcp')
            dR = Tg[:3, :3] @ Tc[:3, :3].T
            e = np.r_[Tg[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
            res = float(np.linalg.norm(e))
            if res < 1e-10:
                break
            J = K.jacobian(q9, 'link_tcp')[:, 3:9]
            q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
        J = K.jacobian(np.r_[b, q], 'link_tcp')[:, 3:9]
        return q, res, float(np.linalg.svd(J, compute_uv=False).min())

    def _off_cl(self, q9):
        """A 的靜態偏置 q−s：以開門起點**實測** (q₀−s₀)（含綁定預載）為錨，加 URDF 重力隨構型的變化 (G(q)−G(q₀))/kp。"""
        return self.off_cl0 + ((self._grav(q9) if self._grav is not None else BM.arm_gravity(K, q9, self.arm_bodies)) - self.G0) / self.joint_kp

    def _closure_arm(self, qp, sp, r, xi_ref, sig_eff, g, tau, va):
        """底盤序列＝本拍 GMPC 解（ξ_ref＋δξ，含逐物理步斜坡）；由管線清空後的預測狀態 qp 起推，得命令生效時刻起每拍末的預測底盤；
        門角＝同一份參考在預測時刻的 φ_ref（與 GMPC 的 T_seq 同一定義）；每拍末解閉鏈 IK → q_cl，設定點 s＝q_cl＋τ·q̇_cl−off；手臂速度＝相鄰設定點差／DT。
        回本拍手臂速度命令（第一拍）；IK 失敗回 None（＝求解失敗 ⇒ 模擬中止）。"""
        xi_seq = xi_ref[:N] + np.asarray(r.delta_xi_all, float)[:N]
        b = np.asarray(qp[:3], float).copy()
        u_prev = np.asarray(self.pipe[-1][:3], float).copy()
        B_ = [b.copy()]
        for k in range(N):
            for j in range(NPH):
                a_ = min(j + 1, NPH) / NPH if BASE_RAMP else 1.0
                v = u_prev + a_ * (xi_seq[k] - u_prev)
                c, s_ = math.cos(b[2]), math.sin(b[2])
                b = b + PHYS * np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], v[2]])
            u_prev = xi_seq[k].copy()
            B_.append(b.copy())
        PH_ = [float(self.ref(sig_eff + k * DT * g)[9]) for k in range(N + 1)]
        seed = self.q_cl_prev if self.q_cl_prev is not None else np.asarray(qp[3:], float)
        Q, SG = [], []
        warm = self.Q_cl_prev_full if self.g2['ik_warm'] else None
        for k in range(N + 1):
            q, res, sm = self._ik_closure(B_[k], PH_[k], warm[k + 1] if (warm is not None and k + 1 <= N) else seed)
            SG.append(sm)
            self.cl_stats['ik_resid_max'] = max(self.cl_stats['ik_resid_max'], res)
            self.cl_stats['sigma_min'] = min(self.cl_stats['sigma_min'], sm)
            if not (res < 1e-6 and np.isfinite(q).all()) or float(np.abs(q - seed).max()) > 0.5:
                self.cl_stats['ik_fail'] += 1
                self._cl_fail = [dict(what='ik_fail', k=k, resid=res)]
                self.cl_stats['fails'].append(dict(t=round(self.t, 3), bad=self._cl_fail))
                return None
            Q.append(q)
            seed = q
            if k == 2 and self._early_cb is not None:                  # r17：本拍手臂命令只依賴前三個節點（中央差分）——先算出並交給候選核對提早派工；
                S1e = Q[1] + tau * (Q[2] - Q[0]) / (2 * DT) - self._off_cl(np.r_[B_[1], Q[1]]) + FF.ff((Q[2] - Q[0]) / (2 * DT), self.g2['fric_ff'], self.joint_kp) + self._pr_use()   # 其餘節點（限制驗收用）照常算完，不過則候選作廢
                self._early_cb(np.r_[np.asarray(r.u_opt, float), (S1e - np.asarray(sp, float)) / DT])
        Q = np.array(Q)
        self.q_cl_prev = Q[1].copy()
        self.Q_cl_prev_full = Q.copy()
        Qd = np.gradient(Q, DT, axis=0)
        S_ = np.array([Q[k] + tau * Qd[k] - self._off_cl(np.r_[B_[k], Q[k]]) + FF.ff(Qd[k], self.g2['fric_ff'], self.joint_kp) + self._pr_use() for k in range(N + 1)])   # r19：庫倫摩擦前饋（door_fric_ff）
        U = np.diff(np.vstack([np.asarray(sp, float)[None], S_[1:]]), axis=0) / DT
        self.cl_stats['n'] += 1
        self.cl_stats['s0_jump_max'] = max(self.cl_stats['s0_jump_max'], float(np.abs(S_[0] - sp).max()))
        # 替換後命令自己的限制驗收（不沿用 QP 的 arm_check）：任一不過 ⇒ 候選失敗（不夾住、不提交）
        acc = np.abs(np.diff(np.vstack([np.asarray(self.u_prev[3:], float)[None], U]), axis=0)) / DT     # 含承接第一拍（對上一個提交的手臂命令）
        rate = float(np.abs(U).max())
        self.cl_stats['rate_max'] = max(self.cl_stats['rate_max'], rate)
        self.cl_stats['acc_max'] = max(self.cl_stats['acc_max'], float(acc.max()))
        lo_q, hi_q = np.asarray(LITE6_SAFE.lower, float), np.asarray(LITE6_SAFE.upper, float)
        sig_q = float(min(SG))                                         # 本拍整個視窗的 IK 構型可操作度
        bad = []
        if rate > va + 1e-12:
            self.cl_stats['rate_clip'] += 1
            bad.append(dict(what='arm_rate', max=round(rate, 4), lim=va))
        if float(acc.max()) > C0.A_ARM + 1e-9:
            bad.append(dict(what='arm_acc', max=round(float(acc.max()), 4), lim=C0.A_ARM))
        if np.any(S_[1:] < self.lim_lo) or np.any(S_[1:] > self.lim_hi):
            bad.append(dict(what='setpoint_limit'))
        if np.any(Q < lo_q) or np.any(Q > hi_q):
            bad.append(dict(what='ik_joint_limit'))
        if sig_q < W.SIG_FLOOR:
            bad.append(dict(what='manipulability', sigma_min=round(sig_q, 5), floor=W.SIG_FLOOR))
        if self.closure_inject is not None:                          # 測試用：真鏈反例（例如收緊限位）
            bad += self.closure_inject(self, U, S_, Q) or []
        if bad:
            self._cl_fail = bad
            self.cl_stats['fails'].append(dict(t=round(self.t, 3), bad=bad))
            return None
        return U[0].copy()

    def _ramp_end(self):
        """若在本拍首提交新命令，其過渡起點＝上一個物理步實際送出的底盤命令。"""
        return np.asarray(self.pipe[-1][:3], float).copy()

    def _decel(self):
        da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
        return self.u_last + np.clip(-self.u_last, -da, da)

    def _monitor(self, q, phi_meas):
        """每控制拍（含停止期間）的實測監看；回 (原因, 細節) 或 None，並記錄一列。"""
        x_ref = self.ref(self.sigma)
        if self.bound:
            ep, er = self.closure_at(q, phi_meas)
            sp_, sr_ = self.closure_at(self.model_q(q), phi_meas)
        else:
            ep = er = sp_ = sr_ = float('nan')
        c = W.clear_door(np.r_[q, phi_meas], self.T_WH_bind)
        if self.clear_inject is not None:
            c = self.clear_inject(self, c)
        for k in KEYS:
            self.mins['act'][k] = min(self.mins['act'][k], c[k])
        Tm, Tr = K.fk(q, 'link_tcp'), K.fk(x_ref[:9], 'link_tcp')
        self.rec.append([self.t, self.sigma, self.g, phi_meas, x_ref[9], ep, er, sp_, sr_, float(np.linalg.norm(q[:2] - x_ref[:2])), abs(q[2] - x_ref[2]), float(np.linalg.norm(Tm[:3, 3] - Tr[:3, 3])), *[c[k] for k in KEYS],
                         float(np.abs(self.u_last[3:]).max()), float(np.hypot(self.u_last[0], self.u_last[1])), (self.force_n if self.force_n is not None else float('nan')), float(self.bound),
                         {'None': 0, 'path': 1, 'settled': 1, 'cstop': 1, 'abort': 4}[str(self.stop_mode)]])
        bad = [k for k in KEYS if c[k] < W.THR[k]]
        out = None
        if bad:
            out = ('clearance', dict(bad=bad, at={k: round(float(c[k]), 4) for k in KEYS}))
        elif self.bound and (ep > CL_TOL['pos_m'] or er > CL_TOL['rot_deg']):
            out = ('closure_residual', dict(pos_mm=round(ep * 1e3, 3), rot_deg=round(er, 3)))
        elif self.bound and self.stop is None and (sp_ > STRESS_TOL['pos_m'] or sr_ > STRESS_TOL['rot_deg']):
            out = ('setpoint_model_deviation', dict(pos_mm=round(sp_ * 1e3, 3), rot_deg=round(sr_, 3)))
        # 停止中「惡化」：最小淨空比上一拍再降 > 2 mm，或實測閉鏈再增 > 2 mm
        if out is None and self.stop is not None and self.stop_mode == 'path' and self.mon_prev is not None:
            dmin = min(c[k] - W.THR[k] for k in KEYS)
            if dmin < self.mon_prev[0] - 0.002 or (self.bound and ep > self.mon_prev[1] + 0.002):
                out = ('path_monitor_worsening', dict(clear_margin_mm=round(dmin * 1e3, 3), closure_mm=round(ep * 1e3, 3)))
        self.mon_prev = (min(c[k] - W.THR[k] for k in KEYS), ep if self.bound else 0.0)
        return out

    def step(self, t, q, phi_meas, force_n=None, w_ext=None):
        """每物理步。q＝實測 (底盤 3＋手臂 6)、phi_meas＝實測門角（綁定中＝把手方位反算或鉸鏈讀值；解除後必須是門的獨立量測）、
        force_n＝手腕傳遞力範數（模擬保護；None＝無讀值）。回 dict(arm_target, base_twist, stopped, done, request_unbind, request_finger_open)。"""
        self.t = float(t)
        q = np.asarray(q, float)
        ab = dict(base_twist=np.zeros(3), stopped=True, done=False, abort=True, request_unbind=False, request_finger_open=False)
        if self.abort is not None:                                   # 已中止：不再產生任何命令
            return dict(ab, arm_target=self.s.copy())
        tick = self.n_step % NPH == 0
        w_t = time.perf_counter()
        self._w_tick, self._qp_tb = w_t, None                         # QP 時間預算的拍起點；本拍是否逾時
        self.force_n = None if force_n is None else float(force_n)
        self._w_ext = None if w_ext is None else (np.asarray(w_ext[0], float), np.asarray(w_ext[1], float))
        if self.force_n is not None:
            self.force_max = max(self.force_max, self.force_n)
            if self.force_n > FORCE_LIMIT_N and self.bound:          # 每物理步、任何模式都檢查
                self._stop('overforce', force_n=round(self.force_n, 3), limit_n=FORCE_LIMIT_N)
        if not (np.isfinite(q).all() and math.isfinite(phi_meas)):
            self._stop('state_nonfinite')
        if self.abort is not None:
            return dict(ab, arm_target=self.s.copy())
        if tick and self.off_cl0 is None:                            # A：開門起點靜態偏置（第一拍、只取一次）；r7：移動承接用上游靜態估計，不取含動態落後的 q−s
            self.off_cl0 = self.off_static.copy() if self.off_static is not None else q[3:] - self.s
            self.G0 = self._grav(q) if self._grav is not None else BM.arm_gravity(K, q, self.arm_bodies)
            self.anchor = dict(t=self.t, q0=q.tolist(), s0=self.s.tolist(), G0=self.G0.tolist(), off0=self.off_cl0.tolist(), joint_kp=self.joint_kp, off_source='handover_off_static' if self.off_static is not None else 'measured_q_minus_s', q_minus_s=(q[3:] - self.s).tolist(),
                               note='錨定含綁定預載，不是自由空間辨識偏置；q−s＝G/kp')
        if tick:
            if self._ccache is not None:
                self._ccache.clear()                                   # 快取只在同一拍內有效
            self._sn = None                                            # r9：上一拍未收的派工作廢（遲到結果會因版本不符被丟棄）
            self._sn_verified = False                                  # r10：本拍現在停是否已驗過
            self._fb_ok, self._fb_info = False, None                   # r11：本拍備援重驗
            self._pr_cand, self._pr_commit_tick = None, False
            if self.g2['preload_release'] and self.bound and self._w_ext is not None and (self.stop is None or self.stop_mode == 'path'):
                # r23（B2）：新 δ 只是候選——本拍「現在停」用已提交的 δ；候選 δ 連同候選命令及其停止預測通過才提交；備援停止（cstop）期間 δ 凍結
                Tt_, Jt_ = DFK.FC.fk_jac(q)
                self._pr_cand, inf_ = PR.update(self._pr_delta, self._w_ext[0], self._w_ext[1], Tt_[:3, 3], Jt_[:, 3:9], self.joint_kp, self.cfg.dt)
                self._pr_inf = inf_
            if self.q_tick is not None:
                self.rate = dict(base=float(np.linalg.norm(q[:2] - self.q_tick[:2])) / self.cfg.dt, yaw=abs(q[2] - self.q_tick[2]) / self.cfg.dt, arm=float(np.abs(q[3:] - self.q_tick[3:]).max()) / self.cfg.dt, door=abs(phi_meas - self.phi_tick) / self.cfg.dt)
            self.q_tick, self.phi_tick = q.copy(), float(phi_meas)
            early_sn = False
            if self.stop is None:                                      # r20（A2）：進度率先算（只用 σ 與實測門角），現在停提前到監看之前派工——用本拍狀態、更新後的 g、同一參考版本
                x_ref = self.ref(self.sigma)
                lead = float(x_ref[9] - phi_meas)
                self.g = float(np.clip((LEAD_MAX - lead) / LEAD_BAND, 0.0, 1.0)) if self.sigma < self.sig_end else 0.0
                if self.g_inject is not None:
                    self.g = float(self.g_inject(self, self.g))
                if G_GOV and self.g > 0.0:
                    g_door = self.g
                    fac = self._gov_factor()
                    g_t = g_door * max(fac, G_GOV_MIN) if fac < 1.0 else g_door
                    g_prev = self._gov_g_prev if self._gov_g_prev is not None else g_t
                    dg = self.cfg.dt / T_STOP                          # g 的變化率沿用路徑停止的減速率（不新增更快的加減速）
                    self.g = float(np.clip(g_t, g_prev - dg, g_prev + dg))
                    self.g = min(self.g, g_door)
                    self._gov_g_prev = self.g
                    self.gov_log.append((round(self.t, 3), round(g_door, 3), round(fac, 3), round(self.g, 3), self._gov_m))
                if self._pool is not None and self._cs_on() and not self.tail_cert_mode:
                    w0 = time.perf_counter()
                    self._cs_submit(q)
                    self._sn, early_sn = 'cstop', True
                    self.sweep_ms.append((time.perf_counter() - w0) * 1e3)
            w_m = time.perf_counter()
            m = self._monitor(q, phi_meas)
            self.mon_ms.append((time.perf_counter() - w_m) * 1e3)
            if self.stop is None:
                if m is not None:
                    self._stop(m[0], **m[1])
                else:
                    self._stall = self._stall + self.cfg.dt if (self.g <= 1e-9 and self.sigma < self.sig_end) else 0.0
                    if self._stall > STALL_S:
                        self._stop('door_not_following', lead_deg=round(math.degrees(lead), 3))
                if self.stop is None and self.sigma >= self.sig_end:
                    if self.t_ref_end is None:
                        self.t_ref_end = self.t
                        self._ev('ref_end', phi_meas_deg=round(math.degrees(phi_meas), 3))
                    good = phi_meas >= self.phi_goal - END['phi_tol'] and self.rate['arm'] <= END['arm_rps'] and self.rate['base'] <= END['base_mps'] and self.rate['yaw'] <= END['yaw_rps'] and self.rate['door'] <= END['door_rps']
                    self._n_end = self._n_end + 1 if good else 0
                    if self._n_end >= END['n_ticks']:
                        self.done = True
                        self._stop('open_done', phi_meas_deg=round(math.degrees(phi_meas), 3))
                    elif self.p2_open:                                 # r12：等延伸（不判 end_not_reached）
                        if self.t - self.t_ref_end > P2_REST_WAIT_S:
                            self._stop('p2_no_extension', phi_meas_deg=round(math.degrees(phi_meas), 3))
                    elif self.t - self.t_ref_end > END['wait_s']:
                        self._stop('end_not_reached', phi_meas_deg=round(math.degrees(phi_meas), 3), rates={k: round(v, 5) for k, v in self.rate.items()})
                if self.stop is None and not self.tail_cert_mode:
                    w0 = time.perf_counter()
                    if early_sn:
                        pass                                           # 已在監看前派工
                    elif self._pool is not None:                         # r9：現在停先派工，與求解同時算，求解後才收
                        if self._cs_on():
                            self._cs_submit(q)                         # r16：綁定中「現在停」核實際會用的閉鏈一致停法（獨立程序模擬＋背景執行緒送淨空；與求解並行）
                            self._sn = 'cstop'
                        else:
                            items0 = self._sweep_prep(q, self.u_last, (0,))
                            self._sn = (items0, self._pool_submit(items0))
                        self.sweep_ms.append((time.perf_counter() - w0) * 1e3)
                    else:
                        b_ = self._sweep(q, self.u_last, (0,), 'stop_now')
                        self.sweep_ms.append((time.perf_counter() - w0) * 1e3)
                        if b_ is not None:
                            self._stop('stop_sweep_violation', **b_)
                        else:
                            self._sn_verified = True
                if self.stop is not None and self._sn == 'cstop':      # 監看或其他判定已停止 ⇒ 提前派工的結果作廢
                    self._sn = None
            elif m is not None and m[0] not in ('setpoint_model_deviation',) and self.stop_mode == 'path':
                self._stop(m[0] if m[0].startswith('path_') else 'path_' + m[0], **m[1])
            elif m is not None and m[0] not in ('setpoint_model_deviation',) and self.stop_mode == 'cstop':   # r16：備援停止中監看違反 ⇒ 模擬中止
                self._stop('cstop_' + m[0], **m[1])
        # ---- 命令
        if tick:
            if self.stop is None:
                self._cl_fail = None
                u, hc_ = self._solve_cand(q, phi_meas)                 # r17：候選模擬提早送出（第二個工作程序），與求解後段及現在停的淨空核對重疊
                if self._sn == 'cstop':                                 # r16：淨空違反 ⇒ 同原語意停止；閉鏈超限只使備援不可用
                    self._sn = None
                    w1 = time.perf_counter()
                    st0, b0c = self._cs_collect()
                    self.sn_ms.append((time.perf_counter() - w1) * 1e3)
                    self.sweep_ms[-1] += (time.perf_counter() - w1) * 1e3
                    if st0 == 'missing':                               # r17：結果缺失／程序失效 ≠ 核過 ⇒ 模擬中止
                        self._stop('cstop_check_unavailable', fb=self._fb_info)
                    elif b0c is not None:
                        self._stop('stop_sweep_violation', **b0c)
                    else:
                        self._sn_verified = True                       # 已得到判定（可行與否見 _fb_ok）
                elif self._sn is not None:                             # r9：收現在停；不過 ⇒ 丟棄求解結果
                    items0, h0 = self._sn
                    self._sn = None
                    w1 = time.perf_counter()
                    b0, why0 = self._pool_eval(items0, h0, 'stop_now')
                    self.sweep_ms[-1] += (time.perf_counter() - w1) * 1e3
                    if why0 is not None:
                        self._stop('stop_sweep_unavailable', pool_why=why0)
                    elif b0 is not None:
                        self._stop('stop_sweep_violation', **b0)
                    else:
                        self._sn_verified = True
                if self.stop is not None:
                    pass
                elif u is None and getattr(self, '_qp_tb', None) is not None:
                    self._deadline_stop(self._qp_tb['elapsed_ms'], 'qp_time_budget')   # QP 流程逾時 ⇒ 同晚到處置（不提交）
                elif u is None:
                    self._stop('closure_cmd_invalid', bad=self._cl_fail) if self._cl_fail else self._stop('solver_fail')
                else:
                    w0 = time.perf_counter()
                    if hc_ is not None:                                 # r17：候選核對用實際停法
                        stc, b_, infc = self._cs_cand_finish(hc_, 'candidate')
                        self.cand_ms.append((time.perf_counter() - w0) * 1e3)
                        if stc == 'ok' and b_ is None and isinstance(infc, dict) and infc.get('min_clear'):
                            self._gov_m = dict(infc['min_clear'])
                        elif stc in ('ok', 'infeasible') and self.first_fail is None:
                            self._first_fail_snap(q, u, stc, b_, infc)
                        why_c = None
                        if stc == 'missing':
                            why_c = infc
                        elif stc == 'infeasible':
                            b_ = dict(tag='candidate_cstop', infeasible=infc)
                    elif self._pool is not None:
                        b_, why_c = self._pool_sweep(q, u, (1, 2), 'candidate')
                    else:
                        b_, why_c = self._sweep(q, u, (1, 2), 'candidate'), None
                    if self.sweep_ms:
                        self.sweep_ms[-1] += (time.perf_counter() - w0) * 1e3
                    if why_c is not None:
                        self._stop('candidate_sweep_unavailable', pool_why=why_c)
                    elif b_ is not None:
                        if _lenient_shell_ok(b_):
                            self.__dict__.setdefault('lenient_log', []).append(dict(t=round(self.t, 3), why='candidate_stop_shell_bar', at=b_.get('at')))   # 寬鬆展示：預測停止路徑外殼—橫桿 ≥3 mm 只記錄（實測淨空與其他項照舊停止）
                        else:
                            self._stop('candidate_stop_violation', **b_)
                if self.stop is None and self.g2['deadline_ms'] is not None and ((time.perf_counter() - w_t) * 1e3 > self.g2['deadline_ms'] or (self.deadline_inject is not None and self.deadline_inject(self))):
                    self._deadline_stop((time.perf_counter() - w_t) * 1e3, 'candidate')   # r10：晚到候選不提交
                if self.stop is None:
                    self.u_last = u
                    self._se_end = self._se_cand
                    self._pr_commit()
                    self.sigma = min(self.sigma + self.cfg.dt * self.g, self.sig_end)
            if self.stop is not None and self.stop_mode == 'path':
                self.g = max(self.g - self.cfg.dt / T_STOP, 0.0)
                slow = self.rate['arm'] <= END['arm_rps'] and self.rate['base'] <= END['base_mps'] and self.rate['yaw'] <= END['yaw_rps']
                if self.g <= 0.0 and slow:
                    self.stop_mode = 'settled'
                    self._ev('path_stop_settled')
                elif self.t - self.t_path_stop > PATH_STOP_MAX_S:
                    self._stop('path_stop_timeout')
                else:
                    self._cl_fail = None
                    hfb = None
                    cs_on = self._cs_on()
                    sn_on = cs_on and not self.tail_cert_mode           # A2：尾段證書模式下路徑停止也不重算現在停（備援＝上一段已核尾段）
                    if sn_on:
                        self._cs_submit(q)                             # r16：路徑停止中每拍重驗閉鏈一致備援（取代各軸減速預測）
                    if not cs_on and self.g2['deadline_ms'] is not None and self._pool is not None:   # r11：路徑停止中每拍重驗備援（現在停預測），與求解並行
                        itf = self._sweep_prep(q, self.u_last, (0,))
                        hfb = (itf, self._pool_submit(itf))
                    u, hc_ = self._solve_cand(q, phi_meas)
                    if hfb is not None:
                        _, whyf = self._pool_eval(hfb[0], hfb[1], 'fallback')
                        if whyf is not None:
                            self._fb_ok = False
                    stp = self._cs_collect()[0] if sn_on else None
                    if stp == 'missing':
                        self._stop('path_cstop_check_unavailable', fb=self._fb_info)
                    elif u is None and getattr(self, '_qp_tb', None) is not None:
                        self._deadline_stop(self._qp_tb['elapsed_ms'], 'path_qp_time_budget')
                    elif u is None:
                        self._stop('closure_cmd_invalid', bad=self._cl_fail) if self._cl_fail else self._stop('solver_fail')
                    else:
                        if cs_on:
                            stc, b_, infc = self._cs_cand_finish(hc_, 'path_candidate')
                            if (stc == 'infeasible' or b_ is not None) and self.first_fail is None:
                                self._first_fail_snap(q, u, stc, b_, infc)
                            why_p = infc if stc == 'missing' else None
                            if stc == 'infeasible':
                                b_ = dict(tag='path_candidate_cstop', infeasible=infc)
                        elif self._pool is not None:
                            b_, why_p = self._pool_sweep(q, u, (1, 2), 'path_candidate')
                        else:
                            b_, why_p = self._sweep(q, u, (1, 2), 'path_candidate'), None     # 路徑停止的新候選同樣經提交前核對
                        if why_p is not None:
                            self._stop('path_candidate_sweep_unavailable', pool_why=why_p)
                        elif b_ is not None:
                            self._stop('path_candidate_rejected', **b_)
                        elif self.g2['deadline_ms'] is not None and ((time.perf_counter() - w_t) * 1e3 > self.g2['deadline_ms'] or (self.deadline_inject is not None and self.deadline_inject(self))):
                            self._deadline_stop((time.perf_counter() - w_t) * 1e3, 'path_candidate')   # 路徑停止中晚到：不提交，改已驗證的有界減速
                        else:
                            self.u_last = u
                            self._se_end = self._se_cand
                            self._pr_commit()
                            self.sigma = min(self.sigma + self.cfg.dt * self.g, self.sig_end)
            if self.stop is not None and self.stop_mode == 'cstop' and self.abort is None:   # r16：閉鏈一致備援停法
                if self.stopped_t is None:
                    self.stopped_t = self.t
                seed = self.q_cl_prev if self.q_cl_prev is not None else np.asarray(q[3:], float)
                rc = self._cs_obj().step_cmd(q, self.s, self.pipe, self.u_last, self.sigma, self.g, seed, self._se_end, self._hold)   # r17：_cs_obj 每次設成當下參考；起點＝已提交終點
                if not rc['ok']:
                    self._stop('cstop_cmd_invalid', cs_why=rc.get('why'))
                else:
                    self.u_last, self.sigma, self.g, self.q_cl_prev, self._se_end = rc['u'], rc['sigma'], rc['g'], rc['q1'], rc['se_end']
                    if rc['hold'] is not None and self._hold is None:
                        self._hold_t0 = self.t
                        self._ev('base_hold_anchor', anchor=[round(float(x), 5) for x in rc['hold']])
                    self._hold = rc['hold']
                    if self._hold is not None:
                        self.hold_log.append((round(self.t, 3), float(np.hypot(rc['u'][0], rc['u'][1])), float(abs(rc['u'][2])), float(np.hypot(*(q[:2] - self._hold[:2]))), float(abs(q[2] - self._hold[2]))))
                    self.cs_log.append((round(self.t, 3), round(self.g, 3), float(np.abs(rc['u'][3:]).max()), round(self._se_end, 4)))
            if self.stop is not None and self.stop_mode == 'settled':
                if self.stopped_t is None:
                    self.stopped_t = self.t
                self.u_last = self._decel()
            if self.g2['preload_release']:                             # 只記錄：提交判定前先存候選
                self._pr_cand_last = None if self._pr_cand is None else self._pr_cand.copy()
            if not self._pr_commit_tick:                               # 本拍沒有提交候選（被拒、晚到、停止中）：δ 與模型一起回到已提交值
                if self._pr_cand is not None:
                    self.pr_rejected = getattr(self, 'pr_rejected', 0) + 1
                self._pr_u = np.zeros(6)
                self._pr_cand = None
            self._pr_model(self._pr_delta)
            if self.g2['preload_release']:
                cl_ = self._pr_cand_last
                ev_ = 0 if cl_ is None else (1 if self._pr_commit_tick else -1)
                w_ = np.full(6, np.nan) if self._w_ext is None else np.r_[self._w_ext[0], self._w_ext[1]]
                self.pr_ts.append(np.r_[self.t, ev_, np.full(6, np.nan) if cl_ is None else cl_, self._pr_delta, self._pr_s, w_])
            self.u_prev = self.u_last.copy()
            self.tick_ms.append((time.perf_counter() - w_t) * 1e3)
        if tick:                                                   # 新的一拍：過渡起點＝上一物理步實際送出的底盤命令
            self.u_from_base, self.ramp_j = self._ramp_end(), 0
        cmd = ramp_cmd(self.u_from_base, self.u_last, self.ramp_j)
        self.ramp_j += 1
        self.pipe.append(cmd)
        self._pr_pipe.append(self._pr_u.copy())                      # δ 的命令分量與手臂命令同步進出管線
        self._pr_s = self._pr_s + PHYS * self._pr_pipe.pop(0)
        self.cmd_hist.append(cmd[:3].copy())
        ua = self.pipe.pop(0)
        if tick:
            self.pipe_log.append(np.r_[self.t, np.concatenate(self.pipe)])
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            self._stop('setpoint_limit')
        self.s = nxt
        self.step_log.append(np.r_[self.t, float(tick), self.sigma, self.g, self.sigma + D_STEPS * PHYS * self.g, self.u_last, cmd, ua, self.s])
        if self.stopped_t is not None and self.q_last is not None and self.stop_ok is None and self.abort is None:
            dq = float(np.abs(q[3:] - self.q_last[3:]).max()) / PHYS
            vb = float(np.hypot(*(q[:2] - self.q_last[:2]))) / PHYS
            wy = abs(q[2] - self.q_last[2]) / PHYS
            dd = abs(phi_meas - self.phi_last) / PHYS
            empty = not np.any(self.u_last) and not any(np.any(u) for u in self.pipe)
            hold_on = self.stop_mode == 'cstop' and self._hold is not None
            if hold_on:                                                # r21：保持模式下「管線」一項＝錨定前發出的運動命令已全部排出管線（錨定後已過 D 個物理步），之後管線內只有保持模式的命令；保持輸出另列，不是零命令；實測速率門檻與持續時間不變
                empty = bool(DCSM.motion_cleared(int(round((self.t - self._hold_t0) / PHYS))))
            st = dict(pipe=bool(empty), arm=bool(dq < STOP_STILL['arm_rps']), base=bool(vb < STOP_STILL['base_mps'] and wy < STOP_STILL['yaw_rps']), door=bool(dd < STOP_STILL['door_rps']))
            self._still = self._still + 1 if all(st.values()) else 0
            lim = STOP_STILL['timeout_s'] + (3.0 if not self.bound else 0.0)   # 解除後門自由減速需要較久
            if self._still * PHYS >= STOP_STILL['hold_s'] or self.t - self.stopped_t >= lim:
                self.stop_ok = dict(t=round(self.t, 3), since_stop_s=round(self.t - self.stopped_t, 3), parts=st, base_hold=None if not hold_on else dict(active=True, anchor=[round(float(x), 5) for x in self._hold], cmd_now=[round(float(x), 5) for x in self.u_last[:3]], pipe_zero=bool(not np.any(self.u_last) and not any(np.any(u) for u in self.pipe)), note='pipe 項＝運動管線清空＋保持啟用；保持輸出非零時不是零命令'), arm_rate=dq, base_speed=vb, yaw_rate=wy, door_rate=dd, bound=self.bound, ok=bool(self._still * PHYS >= STOP_STILL['hold_s']))
                if self.stop_mode == 'cstop' and not self.stop_ok['ok']:        # r16：備援 4 s 內未停穩 ⇒ 模擬中止
                    self._stop('cstop_not_settled', parts=st)
        self.q_last, self.phi_last = q.copy(), float(phi_meas)
        self.n_step += 1
        if self.abort is not None:                                   # 本步中途觸發（例如求解失敗、設定點限位）：本步命令也不送
            return dict(ab, arm_target=self.s.copy())
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), stopped=self.stop is not None, done=self.done, abort=False, request_unbind=False, request_finger_open=False)

    STEP_LOG_COLS = ['t', 'tick', 'sigma', 'g', 'sigma_eff_pred'] + [f'u_commit_{i}' for i in range(9)] + [f'cmd_in_{i}' for i in range(9)] + [f'u_effective_{i}' for i in range(9)] + [f'setpoint_{i}' for i in range(6)]

    def config_readback(self):
        """實際使用的模組、斜坡權重、時鐘選項與延遲值（讀回物件本身，不抄常數）。"""
        import hashlib
        sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()   # noqa: E731
        return dict(exec_module=__file__, exec_sha256=sha(__file__), solver_module=type(self.mpc).__module__, solver_file=sys.modules[type(self.mpc).__module__].__file__, solver_sha256=sha(sys.modules[type(self.mpc).__module__].__file__),
                    arm_mode=self.arm_mode, joint_kp=self.joint_kp, body_model=BM.__file__, body_model_sha256=sha(BM.__file__),
                    cmd_ramp_w=getattr(self.mpc.cfg, 'cmd_ramp_w', None), BASE_RAMP=BASE_RAMP, START_LAT_ALIGN=START_LAT_ALIGN, sigma0_s=self.sigma0, latency_steps=D_STEPS, phys_dt=PHYS, latency_s=D_STEPS * PHYS, nph=NPH, ctrl_dt=self.cfg.dt)

    def report(self):
        Rr = np.array(self.rec, dtype=float) if self.rec else np.zeros((1, 21))
        Rb = Rr[Rr[:, 19] > 0.5] if (Rr[:, 19] > 0.5).any() else Rr[:1]

        def st(col, sc=1.0):
            return dict(max=round(float(Rr[:, col].max()) * sc, 4), p95=round(float(np.percentile(Rr[:, col], 95)) * sc, 4), end=round(float(Rr[-1, col]) * sc, 4))

        def stb(col, sc=1.0):                                      # 只取綁定中的列
            return dict(max=round(float(np.nanmax(Rb[:, col])) * sc, 4), p95=round(float(np.nanpercentile(Rb[:, col], 95)) * sc, 4), end=round(float(Rb[-1, col]) * sc, 4))

        def pct(a):
            a = np.array(a or [0.0])
            return dict(p50=round(float(np.percentile(a, 50)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1))
        return dict(arm_mode=self.arm_mode, closure_anchor=self.anchor, closure_stats={k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in self.cl_stats.items()}, stop=self.stop, done=self.done, stop_effect=self.stop_ok, events=self.events, sigma_end=round(self.sig_end, 2), sigma=round(self.sigma, 3),
                    closure_measured=dict(pos_mm=stb(5, 1e3), rot_deg=stb(6)), setpoint_model_deviation=dict(pos_mm=stb(7, 1e3), rot_deg=stb(8), note='設定點模型的幾何偏差監看（主要是 τ·q̇ 動態超前），不是內應力'), base_track_m=st(9), yaw_track_rad=st(10), tcp_vs_ref_mm=st(11, 1e3),
                    lead_deg_max=round(math.degrees(float((Rr[:, 4] - Rr[:, 3]).max())), 3), g_min=round(float(Rr[:, 2].min()), 3), g_lt1_frac=round(float((Rr[:, 2] < 1 - 1e-9).mean()), 3),
                    min_clear_actual={k: round(float(v), 4) for k, v in self.mins['act'].items()}, min_clear_stop_sweep={k: round(float(v), 4) for k, v in self.mins['stop'].items()},
                    abort=self.abort, bound=self.bound, unbind_t=self.unbind_t, force_max_n=round(self.force_max, 3), force_limit_n=FORCE_LIMIT_N,
                    hold_stop_path=dict(closure_pos_mm_max=round(self.mins['stop_closure_pos'] * 1e3, 3), closure_rot_deg_max=round(self.mins['stop_closure_rot'], 3), n_cfg_over_tol=self.n_emerg_closure, note='命令減速停法的預測閉鏈偏差（只記錄；本版命令減速只用在 path 停到 settled 之後）'), stop_mode=self.stop_mode, n_sweep_cfg=self.mins['n_sweep'],
                    arm_cmd_max=round(float(Rr[:, 16].max()), 4), base_cmd_max=round(float(Rr[:, 17].max()), 4), arm_sat_s=round(float((Rr[:, 16] >= float(B.W_ARM['v_arm']) - 1e-3).sum()) * self.cfg.dt, 2),
                    phi_end_last_tick_deg=round(math.degrees(float(Rr[-1, 3])), 3), phi_end_deg=round(math.degrees(float(Rr[-1, 3])), 3), phi_end_note='最後控制拍的門角；最終物理紀錄的門角由執行器另報', t_end=round(float(self.t), 2),
                    base_cmd_phys=dict(acc_max=[round(float(x), 3) for x in (np.abs(np.diff(np.array(self.cmd_hist), axis=0)).max(axis=0) / PHYS if len(self.cmd_hist) > 1 else np.zeros(3))], limit=LIM.A_BODY.tolist(), ramp=BASE_RAMP, note='逐物理步命令差分（世界前的本體命令）'), limits=LIM.summary(),
                    timing=dict(solve_ms=pct(self.solve_ms), sweep_ms=pct(self.sweep_ms), whole_tick_ms=pct(self.tick_ms), monitor_ms=pct(self.mon_ms), closure_arm_ms=pct(self.cl_ms), candidate_check_ms=pct(self.cand_ms), stop_now_collect_ms=pct(self.sn_ms), prep_ms=pct(self.prep_ms),
                                note='solve_ms＝QP（不含 closure_arm_ms）；sweep_ms 含 stop_now_collect＋candidate_check；整拍另含監看與其他'),
                    g_gov=dict(on=G_GOV, band=G_GOV_BAND, g_min=G_GOV_MIN, n_log=len(self.gov_log), g_min_seen=(min(r_[3] for r_ in self.gov_log) if self.gov_log else None), n_scaled=sum(1 for r_ in self.gov_log if r_[2] < 1.0), tail=self.gov_log[-40:], first_fail=self.first_fail),
                    params=dict(T_STOP=T_STOP, PATH_STOP_MAX_S=PATH_STOP_MAX_S, PATH_OK=list(PATH_OK), FORCE_LIMIT_N=FORCE_LIMIT_N, LEAD_MAX_deg=3.0, LEAD_BAND_deg=1.0, STALL_S=STALL_S, CL_TOL=CL_TOL, STRESS_TOL=STRESS_TOL, END={k: (round(math.degrees(v), 3) if k == 'phi_tol' else v) for k, v in END.items()}, STOP_STILL=STOP_STILL, thresholds=W.THR, finger_q=W.FINGER_Q))
