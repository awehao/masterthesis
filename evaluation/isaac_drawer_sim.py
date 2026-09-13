"""Isaac：固定底盤的手臂經抓取關係帶動被動抽屜。

與第一階段（isaac_manip_sim.py）的關係：那支停在預抓取、**沒有接觸**，而且維持
凍結不動。本支是新檔，多了抽屜資產、抓取關係、力的量測與階段化的停止處置。

抽屜是**被動**的：drive 五項為 0，本程式不發任何開度命令。開度只能由抓取關係
帶動。（施加外力的方向診斷在 isaac_drawer_probe.py，刻意放在另一個檔案。）

抓取模型兩版，由案例的 grasp_model 決定，**不混稱**：
    fixed_attachment  engage 時在夾爪與抽屜之間建一個固定關節。關節的兩端框架
                      取自**當下**的相對位姿，所以連接瞬間沒有位置跳變。
                      手指維持全開 —— 再去夾住橫桿就不是「理想固定連接」了。
    friction          只有接觸與摩擦（本檔支援，但屬於下一輪）。

力的量測（見案例 force 區塊）：
    get_measured_joint_forces() 回傳「該連桿上游關節」的反作用力，且在**子連桿
    座標系**裡。本程式取 joint6 那一列，用 link6 的世界旋轉轉到世界座標，
    再沿抽屜軸投影得 F_pull。**夾爪慣性項沒有扣除**，所以那是「手腕沿抽屜軸
    傳遞的力」，不是純把手拉力。
    座標慣例用靜止段自我核對：無接觸時 |F| 應該等於 joint6 之後所有連桿的重量。

停止處置分兩種（凍結設定點**不保證**力會下降，位置驅動會持續施力）：
    一般異常  凍結最後一次已套用的設定點
    超力      先解除耦合（拆掉固定關節／張開手指），再凍結，才停止
"""
import argparse, hashlib, json, math, os, sys, threading, time
import numpy as np
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='drawer_open_a_fixed')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--poses', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--headless', default='true')
ap.add_argument('--sim-limit', type=float, default=45.0)
ap.add_argument('--wall-limit', type=float, default=900.0)
ap.add_argument('--rtf', type=float, default=1.0, help='>0 時以此倍率對齊牆鐘')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
# 快照取得模式：只做到接近、停穩與取得同步快照，**不建立固定連接、不執行拉開**。
# 用途是在新的底盤固定方式下取得對準參考所需的實際抓取關係，
# 不必為此承受一次已知可能超力的未對準拉開段。
ap.add_argument('--snapshot-only', action='store_true',
                help='在 engage 時刻記錄快照但不連接，隨即安全收尾')
# --- 執行時錄影（**預設關閉；不帶旗標時本檔行為與未加此功能時逐行相同**）---
# 錄的是**這一趟真正執行的畫面**，不是事後的姿態重演。
# 只加光源（headless 場景沒有光就全黑），**不改任何幾何、材質顏色、質量或物理參數**，
# 也不疊任何文字。保護條件不受影響：中止判斷與監看都在錄影之外照常執行。
# --- 摩擦夾持：閉合放行閘 ---
# 閉合斜坡與 postengage 相位解耦，**但不與對中驗收解耦**。命令序列可以隨時
# 要求閉合；真正放行與否由這裡把關：實際對中量測達標並連續保持才放行。
ap.add_argument('--grip-gate-offset-mm', type=float, default=0.30,
                help='放行門檻：兩指中心沿閉合軸相對把手軸的偏移上限（mm）')
ap.add_argument('--grip-gate-hold-s', type=float, default=0.5,
                help='達標須連續保持的時間（秒）')
ap.add_argument('--grip-gate-timeout-s', type=float, default=8.0,
                help='命令要求閉合後，閘未放行的容許等待時間；逾時即中止。'
                     '等待期間手指維持全開，放行後由閘自行產生完整斜坡')
ap.add_argument('--grip-gate-pos-mm', type=float, default=20.0,
                help='放行條件：兩指中點到把手軸、垂直閉合軸的距離上限（mm）')
ap.add_argument('--grip-gate-rot-deg', type=float, default=1.0,
                help='放行條件：工具姿態相對設計姿態的夾角上限（度）')
ap.add_argument('--grip-gate-track-rad', type=float, default=0.005,
                help='放行條件：手臂關節追蹤誤差上限（rad）')
ap.add_argument('--grip-hold-contact-n', type=float, default=0.5,
                help='夾持驗收：每指接觸量下限（N）。**接觸存在門檻，不稱法向夾持力**')
ap.add_argument('--grip-hold-pos-mm', type=float, default=0.20,
                help='夾持驗收：把手相對夾爪的位置變化上限（mm，相對保持起點）')
ap.add_argument('--grip-hold-rot-deg', type=float, default=1.0,
                help='夾持驗收：把手相對夾爪的姿態變化上限（度，相對保持起點）')
ap.add_argument('--grip-hold-s', type=float, default=2.0,
                help='夾持驗收：須由**實際資料**連續滿足的時間（秒）')
ap.add_argument('--pull-slip-mm', type=float, default=2.0,
                help='拉動段：把手相對夾爪的位置變化上限（mm，相對**拉動起點**）')
ap.add_argument('--pull-rot-deg', type=float, default=2.0,
                help='拉動段：把手相對夾爪的姿態變化上限（度，相對**拉動起點**）')
ap.add_argument('--grip-close-ramp-s', type=float, default=3.0,
                help='放行後由閘自行產生的閉合斜坡長度（秒）')
ap.add_argument('--record-frames', default='', help='輸出 PNG 的目錄；空字串=不錄')
ap.add_argument('--record-fps', type=float, default=30.0)
ap.add_argument('--record-res', default='1280x720')
ap.add_argument('--record-focal', type=float, default=20.0)
ap.add_argument('--record-eye', default='')
ap.add_argument('--record-at', default='')
ap.add_argument('--record-warmup', type=int, default=40,
                help='開錄前先算繪幾幀：RTX 標註器首次取像會回 None')
ap.add_argument('--align-check', default='',
                help='對準版軌跡的 traj_meta.json；engage 時核對實際快照是否與'
                     '產生軌跡時用的那份相符')
# --- 獨立診斷用（**不計為正式操作成果**）---
# 僅取消**抽屜剛體**的重力作用；手臂與夾爪重力保留。
# 抽屜質量、慣量、阻尼、滑軌、固定連接、偏置策略與軌跡配時全部不變。
ap.add_argument('--drawer-no-gravity', action='store_true',
                help='只把抽屜剛體的 disableGravity 設為 True（診斷用）')
# 只跑原軌跡的前段：命令序列與完整版**逐點相同**，到指定開度即停，
# **不是**把整條軌跡壓縮到該行程（那會改變速度剖面）。
# 只重播命令、**不建立夾爪—抽屜固定關節**：回答「負載成長是否依賴這個連接」。
# engage 事件照常記錄（同步快照照取），但不建關節、也不把 coupled 設為 True
# （否則 grasp_lost 會因抽屜不動而誤觸發）。
ap.add_argument('--no-attach', action='store_true',
                help='engage 時記錄快照但**不建立固定關節**，繼續重播（診斷用）')
# 以**指令進度**收尾：不建立連接時抽屜不會動，不能等它開到某個開度。
ap.add_argument('--stop-at-cmd-t', type=float, default=0.0,
                help='>0 時已套用設定點的 t_sched 超過此值即停（診斷用）')
ap.add_argument('--stop-at-opening-m', type=float, default=0.0,
                help='>0 時開度超過此值即停（診斷用），停止原因 diagnostic_opening_limit')
ap.add_argument('--pull-target-m', type=float, default=0.0,
                help='>0 時覆寫本趟的目標開度（有界交接驗證用），須與軌跡一致')
ap.add_argument('--align-tol-m', type=float, default=0.002)
ap.add_argument('--align-tol-deg', type=float, default=0.2)
ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args()

import drawer_asset as DA                                           # noqa: E402
import drawer_align as DAL                                         # noqa: E402
from cpu_temp import read as cpu_temp_read
from grip_gate import GripGate, GripHold                          # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

WB_URDF = os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf')
KIN = WholeBodyKinematics.from_urdf_string(open(WB_URDF).read())
KIDX = [KIN.dof_names.index(f'joint{i}') for i in range(1, 7)]


def apply_grasp_friction(stage, grip, drawer_prim, finger_links):
    """把接觸摩擦材質綁到**把手橫桿**與**兩片指墊的碰撞形狀**，並讀回驗證。

    為什麼要讀回：設了不等於生效。材質可能綁到視覺模型而不是碰撞形狀，
    綁定也可能被更強的 binding 蓋掉。這裡逐一列出實際綁到哪些 prim，
    並把讀回值寫進輸出，讓「摩擦是 0.8」變成可查核的事實而不是宣稱。

    PhysX 的有效摩擦由**接觸雙方的材質與混合規則**共同決定，不是單方說了算 ——
    所以混合規則也一併明訂並記錄。
    """
    from pxr import UsdShade, UsdPhysics, PhysxSchema, Sdf

    mpath = '/World/PhysicsMaterials/grasp_friction'
    mat = UsdShade.Material.Define(stage, mpath)
    mp = mat.GetPrim()
    api = UsdPhysics.MaterialAPI.Apply(mp)
    api.CreateStaticFrictionAttr().Set(float(grip['friction_static']))
    api.CreateDynamicFrictionAttr().Set(float(grip['friction_dynamic']))
    api.CreateRestitutionAttr().Set(float(grip.get('friction_restitution', 0.0)))
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(mp)
    mode = str(grip['friction_combine_mode'])
    pxm.CreateFrictionCombineModeAttr().Set(mode)
    pxm.CreateRestitutionCombineModeAttr().Set(mode)

    # 目標 1：把手橫桿（build_usd 對它套了 CollisionAPI）
    targets = []
    bar = stage.GetPrimAtPath(f'{drawer_prim}/handle_bar')
    if bar and bar.IsValid() and bar.HasAPI(UsdPhysics.CollisionAPI):
        targets.append(str(bar.GetPath()))
    # 目標 2：兩片指墊底下**帶 CollisionAPI** 的 prim（不是視覺模型）
    #
    # 兩個坑：
    #  1) 指墊幾何是 USD instance，**預設的 PrimRange 不會走進 instance proxy**，
    #     不加 TraverseInstanceProxies 會一個都找不到（第一版就是這樣）。
    #  2) instance proxy **不可授寫**。要綁材質得先把外層 instance 取消 instanceable；
    #     這只改場景的實例化表示，不改幾何、質量或任何物理參數。
    def _scan(fl):
        root = stage.GetPrimAtPath(fl)
        found, tree = [], []
        if not (root and root.IsValid()):
            return found, tree
        for pr in Usd.PrimRange(root, Usd.TraverseInstanceProxies(
                Usd.PrimDefaultPredicate)):
            has = bool(pr.HasAPI(UsdPhysics.CollisionAPI))
            tree.append({'prim': str(pr.GetPath()), 'type': str(pr.GetTypeName()),
                         'collision': has, 'instance_proxy': bool(pr.IsInstanceProxy()),
                         'instance': bool(pr.IsInstance())})
            if has:
                found.append(pr)
        return found, tree

    subtree = []
    for fl in finger_links:
        found, tree = _scan(fl)
        # 有 instance proxy 就先解除外層 instanceable，再重掃一次
        if any(pr.IsInstanceProxy() for pr in found):
            n_un = 0
            for pr in Usd.PrimRange(stage.GetPrimAtPath(fl),
                                    Usd.TraverseInstanceProxies(
                                        Usd.PrimDefaultPredicate)):
                if pr.IsInstance():
                    pr.SetInstanceable(False); n_un += 1
            print(f'[drawer] {fl}：取消 {n_un} 個 instanceable 以便授寫材質',
                  flush=True)
            found, tree = _scan(fl)
        subtree.append({'link': fl, 'n_collision': len(found), 'tree': tree})
        for pr in found:
            targets.append(str(pr.GetPath()))

    bound = []
    for tp in targets:
        pr = stage.GetPrimAtPath(tp)
        b = UsdShade.MaterialBindingAPI.Apply(pr)
        b.Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
    # 讀回：實際解析到的 physics 材質路徑
    for tp in targets:
        pr = stage.GetPrimAtPath(tp)
        rel = UsdShade.MaterialBindingAPI(pr).GetDirectBindingRel('physics')
        got = [str(x) for x in (rel.GetTargets() or [])] if rel else []
        bound.append({'prim': tp, 'bound_to': got, 'ok': got == [mpath]})

    # --- 碰撞幾何與接觸 offset 讀回 ---
    # 「以 URDF 點雲 + stage 位姿算出的間隙」不等於物理引擎實際使用的碰撞形狀
    # 與接觸距離。FK 與 stage 一致**不能**推論碰撞幾何也已驗證，所以這裡分開讀。
    def _collider_info(tp):
        pr = stage.GetPrimAtPath(tp)
        d = {'prim': tp, 'type': str(pr.GetTypeName())}
        try:
            mc = UsdPhysics.MeshCollisionAPI(pr)
            at = mc.GetApproximationAttr() if mc else None
            d['approximation'] = str(at.Get()) if (at and at.Get()) else None
        except Exception as e:
            d['approximation'] = f'讀取失敗 {e!r}'
        try:
            pc = PhysxSchema.PhysxCollisionAPI(pr)
            if pc:
                for nm, fn in (('contact_offset', pc.GetContactOffsetAttr),
                               ('rest_offset', pc.GetRestOffsetAttr)):
                    at = fn()
                    v = at.Get() if at else None
                    d[nm] = None if v is None else float(v)
            else:
                d['contact_offset'] = d['rest_offset'] = None
                d['offset_note'] = 'prim 未套用 PhysxCollisionAPI'
        except Exception as e:
            d['contact_offset'] = d['rest_offset'] = f'讀取失敗 {e!r}'
        if d.get('contact_offset') in (None, -1.0) or d.get('rest_offset') in (None, -1.0):
            d.setdefault('offset_note', '')
            d['offset_note'] += '（-1 或 None 代表沿用場景預設，實際值需由場景讀）'
        return d

    collider_rb = [_collider_info(tp) for tp in targets]
    scene_rb = None
    for pr in stage.Traverse():
        if pr.HasAPI(PhysxSchema.PhysxSceneAPI):
            sc = PhysxSchema.PhysxSceneAPI(pr)
            scene_rb = {'prim': str(pr.GetPath())}
            for nm, fn in (('default_contact_offset', 'GetContactOffsetAttr'),
                           ('default_rest_offset', 'GetRestOffsetAttr')):
                at = getattr(sc, fn, lambda: None)()
                v = at.Get() if at else None
                scene_rb[nm] = None if v is None else float(v)
            break

    rb = {
        'material_prim': mpath,
        'colliders': collider_rb,
        'physx_scene': scene_rb,
        'gap_definition_gap': (
            '本專案離線算的「間隙」= URDF collision 網格取樣點雲到把手解析圓柱的'
            '最短距離。**物理引擎用的是該網格的凸包近似加上接觸 offset**，兩者'
            '不是同一個量；離線間隙為 0 之前就可能產生接觸力。'
            '未經對照前，不得以離線間隙推論接觸時機'),
        'static_friction': float(api.GetStaticFrictionAttr().Get()),
        'dynamic_friction': float(api.GetDynamicFrictionAttr().Get()),
        'restitution': float(api.GetRestitutionAttr().Get()),
        'friction_combine_mode': str(pxm.GetFrictionCombineModeAttr().Get()),
        'scope_note': ('只綁把手橫桿與指墊碰撞形狀；櫃體與抽屜其他面未設材質，'
                       '沿用 PhysX 預設'),
        'combine_note': ('有效摩擦由接觸雙方材質與混合規則共同決定；'
                         '與未設材質的一方接觸時，結果取決於預設材質與本規則'),
        'targets': bound,
        'n_targets': len(bound),
        'finger_subtrees': subtree,
        'all_bound_ok': bool(bound) and all(x['ok'] for x in bound),
    }
    print(f'[drawer] 夾持摩擦材質：靜 {rb["static_friction"]} / 動 '
          f'{rb["dynamic_friction"]}，混合規則 {rb["friction_combine_mode"]}', flush=True)
    for x in bound:
        print(f'    綁定 {"OK " if x["ok"] else "**失敗**"} {x["prim"]}', flush=True)
    return rb


def fk_tcp(qa, park):
    q = np.zeros(len(KIN.dof_names))
    q[0], q[1], q[2] = park
    q[KIDX] = np.asarray(qa, float)
    return KIN.fk(q, 'link_tcp')

CASES = yaml.safe_load(open(a.cases))
CASE = CASES['cases'][a.case]
SPEC = DA.load(a.spec)
POSES = yaml.safe_load(open(a.poses))
ARM = [f'joint{i}' for i in range(1, 7)]
Q_START = np.array([float(POSES[CASE['pregrasp']['start_config']][j]) for j in ARM])
PARK = (float(CASE['parking']['x']), float(CASE['parking']['y']),
        math.radians(float(CASE['parking']['yaw_deg'])))
POSE = (float(CASE['object']['pose'][0]), float(CASE['object']['pose'][1]))
TOL = CASE['tolerance']
TMO = CASE['timeout_s']
FRC = CASE['force']
AXIS = np.array(FRC['drawer_axis_world'], float)
AXIS = AXIS / np.linalg.norm(AXIS)
GRASP_MODEL = CASE['grasp_model']
GRIP_SPEC = CASE.get('grip') or {}
TARGET = float(CASE['drawer']['target_opening_m'])
TARGET_CASE = TARGET
if a.pull_target_m > 0:
    TARGET = float(a.pull_target_m)
CPU_LIMIT = float(CASE['cpu_limit_c'])
F_OPEN = float(SPEC['grasp_surface']['finger_joint_open'])
TCP_OFF = float(SPEC['grasp_surface']['tcp_offset_along_tool_z'])
BAR = SPEC['drawer']['handle']['bar']['center']

ALIGN_REF = None
if a.align_check:
    _m = json.load(open(a.align_check))
    if _m.get('aligned'):
        ALIGN_REF = _m['alignment']['snapshot']
        print(f'[drawer] 對準版：將核對 engage 快照（來源 '
              f'{ALIGN_REF["source_run"]}），容差 '
              f'{a.align_tol_m*1000:.1f} mm / {a.align_tol_deg:.2f}°')

os.makedirs(a.out, exist_ok=True)
SHA = {k: hashlib.sha256(open(v, 'rb').read()).hexdigest()[:16] for k, v in
       (('urdf', a.urdf), ('spec', a.spec), ('cases', a.cases), ('poses', a.poses))}
print(f'[drawer] 案例 {a.case}  抓取模型 {GRASP_MODEL}')
print(f'[drawer] sha {SHA}')
print(f'[drawer] 停放 ({PARK[0]:.4f}, {PARK[1]:.4f}) yaw {CASE["parking"]["yaw_deg"]}°')
print(f'[drawer] 目標開度 {TARGET:.3f} m ± {TOL["opening_m"]:.3f}，'
      f'保持 {TOL["opening_hold_s"]:.1f} s'
      + (f'  **本趟覆寫，案例值 {TARGET_CASE:.3f} m**' if a.pull_target_m > 0 else ''))

from isaacsim import SimulationApp                                  # noqa: E402
_cfg = {'headless': a.headless.lower() == 'true'}
if a.cpu_threads > 0:
    _cfg['limit_cpu_threads'] = a.cpu_threads
sim_app = SimulationApp(_cfg)

from isaacsim.core.api import World                                 # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane      # noqa: E402
from isaacsim.core.prims import SingleArticulation, RigidPrim       # noqa: E402
from isaacsim.core.utils.types import ArticulationAction            # noqa: E402
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, Usd   # noqa: E402
from isaac_common import import_urdf                                # noqa: E402

import rclpy                                                        # noqa: E402
from rclpy.node import Node                                         # noqa: E402
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)                            # noqa: E402
from rosgraph_msgs.msg import Clock                                 # noqa: E402
from sensor_msgs.msg import JointState                              # noqa: E402
from geometry_msgs.msg import PoseStamped                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String                  # noqa: E402

class MonitorFailure(RuntimeError):
    """監看量讀不到。**不是**「沒有超限」，兩者必須分開處置。"""

    def __init__(self, what, detail=''):
        super().__init__(f'{what}: {detail}')
        self.what = what
        self.detail = str(detail)


DRAWER = '/World/drawer_unit/drawer'
ROBOT = '/World/omni_bot'
JOINT_ATTACH = '/World/drawer_unit/grasp_attach'


def q_yaw(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class DrawerNode(Node):
    def __init__(self):
        super().__init__('isaac_drawer_sim')
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
        rel = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_ALL,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.tcp_pub = self.create_publisher(PoseStamped, '/manip/tcp_pose', be)
        self.dr_pub = self.create_publisher(Float64MultiArray, '/manip/drawer_state', be)
        self.gr_pub = self.create_publisher(Float64MultiArray, '/manip/grasp_state', be)
        self.ct_pub = self.create_publisher(Float64MultiArray, '/manip/contact', be)
        self.st_pub = self.create_publisher(String, '/manip/status', 10)
        # 一次指派的快照：讀取端一次取走整組，不會讀到半新半舊的 (seq, q)
        self.snap = None
        self.fsnap = None
        self.pending = []           # 尚未套用的事件
        self.phase = 'idle'
        self.arm_cb_n = self.arm_cb_rej = 0
        self.g_cb_n = self.p_cb_n = 0
        self.create_subscription(Float64MultiArray, '/arm/joint_position_cmd',
                                 self._cmd, 10)
        self.create_subscription(Float64MultiArray, '/manip/gripper_cmd',
                                 self._grip, 10)
        self.create_subscription(String, '/manip/phase_cmd', self._phase, rel)

    def _cmd(self, m):
        self.arm_cb_n += 1
        if len(m.data) != 9:
            self.arm_cb_rej += 1
            return
        self.snap = (int(m.data[0]), float(m.data[2]), np.array(m.data[3:9]))

    def _grip(self, m):
        self.g_cb_n += 1
        if len(m.data) != 3:
            return
        self.fsnap = (int(m.data[0]), float(m.data[2]))

    def _phase(self, m):
        self.p_cb_n += 1
        try:
            d = json.loads(m.data)
        except Exception:
            return
        self.phase = d.get('phase', self.phase)
        if d.get('event'):
            self.pending.append(d)

    def say(self, d):
        s = String(); s.data = json.dumps(d, ensure_ascii=False)
        self.st_pub.publish(s)

    def farr(self, pub, vals):
        m = Float64MultiArray(); m.data = [float(v) for v in vals]; pub.publish(m)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    if abs(float(world.get_rendering_dt()) - float(a.physics_dt)) > 1e-12:
        print('[drawer] rendering_dt != physics_dt，中止'); return 4
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    authored = DA.build_usd(world.stage, SPEC, POSE)
    # --- S3 固定底座版 ---
    # 舊版用「每個物理步呼叫 set_linear_velocity(0)/set_angular_velocity(0)」固定底盤。
    # 逐步診斷顯示那個覆寫本身會使保持偏差與 effort 增加、位置游移放大，
    # 所以改成匯入器的 fix_base：world→根 的單一固定關節。
    # **這代表底盤被外部固定支撐**，不代表輪子靠地面摩擦能承受相同的操作負載；
    # 移動底盤的操作要另外驗證。
    import_urdf(a.urdf, ROBOT, fix_base=True)

    stage = world.stage
    # root_joint 的 body0 是 /World/omni_bot 這個 Xform、錨點 localPos0 = 0，
    # 所以把**那個 Xform** 移到停放位姿即可；不動 root_joint，也不另建第二個
    # world 固定關節。之後**不呼叫 set_world_pose**（會與固定關節打架）。
    _root_fixed = []
    for _p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
            Usd.PrimDefaultPredicate)):
        if not str(_p.GetPath()).startswith(ROBOT) or not _p.IsA(UsdPhysics.FixedJoint):
            continue
        _j = UsdPhysics.Joint(_p)
        _b1 = [str(t) for t in (_j.GetBody1Rel().GetTargets() or [])]
        _b0 = [str(t) for t in (_j.GetBody0Rel().GetTargets() or [])]
        if _b1 and _b1[0].endswith('base_footprint') and (not _b0 or _b0[0] == ROBOT):
            _root_fixed.append(str(_p.GetPath()))
    print(f'[drawer] world→根 固定關節 {len(_root_fixed)} 個：{_root_fixed}')
    if len(_root_fixed) != 1:
        print('[drawer] **預期恰好 1 個 world→根 固定關節，中止**'); return 10
    # --- 診斷：只取消抽屜的重力 ---
    grav_state = {}
    if a.drawer_no_gravity:
        _d = stage.GetPrimAtPath(DRAWER)
        PhysxSchema.PhysxRigidBodyAPI.Apply(_d).CreateDisableGravityAttr(True)
        print('[drawer] **診斷模式：已把抽屜剛體的 disableGravity 設為 True**',
              flush=True)
    # 逐一檢查所有剛體，確認取消重力只作用於抽屜
    for _p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
            Usd.PrimDefaultPredicate)):
        if not _p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        _v = None
        if _p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            _at = PhysxSchema.PhysxRigidBodyAPI(_p).GetDisableGravityAttr()
            _v = bool(_at.Get()) if _at and _at.Get() is not None else None
        grav_state[str(_p.GetPath())] = _v
    _off = [k for k, v in grav_state.items() if v is True]
    print(f'[drawer] 剛體 {len(grav_state)} 個；disableGravity=True 的有 '
          f'{len(_off)} 個：{_off}')
    if a.drawer_no_gravity and _off != [DRAWER]:
        print(f'[drawer] **取消重力的範圍不等於只有抽屜，中止**'); return 11
    if not a.drawer_no_gravity and _off:
        print(f'[drawer] **非診斷模式卻有剛體被取消重力，中止**'); return 11

    _xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    _xf.ClearXformOpOrder()
    _xf.AddTranslateOp().Set(Gf.Vec3d(PARK[0], PARK[1], 0.0))
    _rq = q_yaw(PARK[2])
    _xf.AddOrientOp().Set(Gf.Quatf(_rq[0], _rq[1], _rq[2], _rq[3]))
    print(f'[drawer] 已把 {ROBOT} 移到停放位姿；root_joint 未更動')

    fingers = [str(p.GetPath()) for p in stage.Traverse()
               if p.GetName() in ('uflite_finger1', 'uflite_finger2')]
    grip_link = [str(p.GetPath()) for p in stage.Traverse()
                 if p.GetName() == 'uflite_gripper_link']
    print(f'[drawer] 手指 prim {fingers}')
    print(f'[drawer] 夾爪 prim {grip_link}')
    if not grip_link:
        print('[drawer] 找不到 uflite_gripper_link，中止'); return 5
    GRIP = grip_link[0]

    # --- friction 版：接觸摩擦材質（fixed 版完全不走這段）---
    fric_rb = None
    if GRASP_MODEL == 'friction':
        fric_rb = apply_grasp_friction(stage, GRIP_SPEC, DRAWER, fingers)
        if not fric_rb['all_bound_ok']:
            print('[drawer] **摩擦材質未全部綁到碰撞形狀，中止**'); return 13
        if not fric_rb['n_targets'] >= 3:
            print(f'[drawer] **綁定目標只有 {fric_rb["n_targets"]} 個'
                  '（預期至少把手 1 + 指墊 2），中止**')
            for st in fric_rb.get('finger_subtrees', []):
                print(f'  子樹 {st["link"]}（帶 CollisionAPI 的有 '
                      f'{st["n_collision"]} 個）：')
                for e in st['tree'][:40]:
                    print(f'    {"C" if e["collision"] else " "}'
                          f'{"P" if e["instance_proxy"] else " "}'
                          f'{"I" if e["instance"] else " "} '
                          f'{e["type"]:<14s} {e["prim"]}')
            return 13

    # **view 必須在 world.reset() 之前建立**：prepare_contact_sensors 要早於
    # PhysX 場景建好才有效。reset 之後才建，接觸力一律回傳 0（已實測）。
    print('[drawer] 建立抽屜 view ...', flush=True)
    drawer_v = RigidPrim(prim_paths_expr=DRAWER, name='drawer_v',
                         track_contact_forces=True, max_contact_count=128,
                         prepare_contact_sensors=True)
    print('[drawer] 建立手指 view ...', flush=True)
    # 夾爪殼對抽屜的接觸也要追蹤：**不建立固定關節不等於沒有其他接觸傳力**
    grip_v = RigidPrim(prim_paths_expr=GRIP, name='grip_v',
                       track_contact_forces=True, max_contact_count=128,
                       prepare_contact_sensors=True,
                       contact_filter_prim_paths_expr=[DRAWER])
    finger_v = None
    if fingers:
        # 用字元類別而不是 (A|B) 的全路徑交替：兩條路徑只差最後一個字元
        finger_v = RigidPrim(prim_paths_expr=fingers[0][:-1] + '[12]',
                             name='finger_v', track_contact_forces=True,
                             max_contact_count=128, prepare_contact_sensors=True,
                             contact_filter_prim_paths_expr=[DRAWER])
    # --- 執行時錄影：相機與光源要在 reset 之前建好 ---
    rec_cam = None
    rec_every = 0
    rec_dir = a.record_frames
    rec_index = []
    if rec_dir:
        from isaacsim.sensors.camera import Camera            # noqa: E402
        from pxr import UsdLux                                # noqa: E402
        os.makedirs(rec_dir, exist_ok=True)
        _k = UsdLux.DistantLight.Define(stage, '/World/rec_key')
        _k.CreateIntensityAttr(3000.0)
        UsdGeom.Xformable(_k).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
        UsdLux.DomeLight.Define(stage, '/World/rec_dome').CreateIntensityAttr(450.0)
        _foc = np.array([POSE[0], POSE[1] - 0.38, 0.45])
        _eye = (np.array([float(v) for v in a.record_eye.split(',')])
                if a.record_eye else _foc + np.array([1.70, -2.45, 0.92]))
        _at = (np.array([float(v) for v in a.record_at.split(',')])
               if a.record_at else _foc)
        _w, _h = (int(v) for v in a.record_res.split('x'))
        rec_cam = Camera(prim_path='/World/rec_cam', resolution=(_w, _h))
        _cx = UsdGeom.Xformable(stage.GetPrimAtPath('/World/rec_cam'))
        _cx.ClearXformOpOrder()
        _cx.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*[float(v) for v in _eye]), Gf.Vec3d(*[float(v) for v in _at]),
            Gf.Vec3d(0, 0, 1)).GetInverse())
        _cg = UsdGeom.Camera(stage.GetPrimAtPath('/World/rec_cam'))
        _cg.GetFocalLengthAttr().Set(float(a.record_focal))
        _cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 200.0))
        rec_every = max(1, int(round(1.0 / (a.record_fps * a.physics_dt))))
        print(f'[drawer] **執行時錄影開啟** -> {rec_dir}；{_w}x{_h} '
              f'每 {rec_every} 個物理步取一幀（約 {1.0/(rec_every*a.physics_dt):.1f} fps）'
              f'；相機 eye={np.round(_eye,3).tolist()} at={np.round(_at,3).tolist()}',
              flush=True)

    # --- 接觸量測介面自省：**列對應哪根手指、過濾哪些物件，要有證據** ---
    # 「指2為零」若其實是列序搞反、或過濾沒涵蓋把手，結論就整個反過來。
    contact_introspect = None
    if GRASP_MODEL == 'friction' and finger_v is not None:
        def _paths(v):
            for at in ('prim_paths', 'prims', '_prim_paths'):
                x = getattr(v, at, None)
                if x is None:
                    continue
                try:
                    return [str(getattr(e, 'GetPath', lambda: e)()) for e in x]
                except Exception:
                    return [str(e) for e in x]
            return None
        contact_introspect = {
            'finger_view_expr': fingers[0][:-1] + '[12]',
            'finger_view_prim_paths_in_row_order': _paths(finger_v),
            'contact_filter_prim_paths': [DRAWER],
            'drawer_rigid_body_prim': DRAWER,
            'handle_bar_collider': f'{DRAWER}/handle_bar',
            'filter_covers_bar_note': (
                '過濾對象是抽屜剛體；把手橫桿是它底下的碰撞形狀。'
                '**此處只記錄設定，涵蓋與否由本趟的接觸讀數驗證**'),
            'available_contact_methods': sorted(
                m for m in dir(finger_v) if 'contact' in m.lower()),
            'force_semantics_gap': (
                'get_contact_force_matrix 回傳的是合力向量；'
                '**本趟未證實它是否包含摩擦分量，也未取得接觸點與法向**。'
                '列為缺口，不以分量投影冒充接觸法向／摩擦力'),
        }
        print('[drawer] 手指接觸 view 列序：'
              f'{contact_introspect["finger_view_prim_paths_in_row_order"]}', flush=True)
        print(f'[drawer] 可用接觸方法：{contact_introspect["available_contact_methods"]}',
              flush=True)

    print('[drawer] world.reset() ...', flush=True)
    world.reset()
    print('[drawer] reset 完成', flush=True)
    if rec_cam is not None:
        import imageio.v2 as _imageio                          # noqa: E402
        rec_cam.initialize()
        for _ in range(max(1, a.record_warmup)):
            world.render()
            if rec_cam.get_rgba() is not None and len(rec_cam.get_rgba()):
                break
        else:
            print('[drawer] **相機暖機後仍取不到影像，中止**'); return 12
        for _ in range(max(1, a.record_warmup)):
            world.render()
        print('[drawer] 相機暖機完成', flush=True)
    robot = SingleArticulation(prim_path=ROBOT, name='omni_bot')
    robot.initialize()
    print(f'[drawer] articulation DOF {robot.num_dof}', flush=True)
    names = list(robot.dof_names)
    idx = {n: i for i, n in enumerate(names)}
    missing = [j for j in ARM if j not in idx]
    if missing:
        print(f'[drawer] URDF 缺關節 {missing}，中止'); return 5

    # --- 手指 drive 出力上限：**執行期**讀回，不以 USD 上的值冒充 ---
    finger_force_rb = {'source': None, 'values': None,
                       'gap': '未嘗試'} if GRASP_MODEL == 'friction' else None
    if GRASP_MODEL == 'friction':
        FJ_probe = [j for j in ('finger_joint1', 'finger_joint2') if j in idx]
        got = None; src = None
        for nm in ('get_max_efforts', 'get_max_joint_efforts'):
            fn = getattr(robot, nm, None)
            if fn is None:
                continue
            try:
                v = np.asarray(fn()).reshape(-1)
                got = {j: float(v[idx[j]]) for j in FJ_probe}; src = f'robot.{nm}()'
                break
            except Exception:
                continue
        if got is None:
            av = getattr(robot, '_articulation_view', None)
            fn = getattr(av, 'get_max_efforts', None) if av is not None else None
            if fn is not None:
                try:
                    v = np.asarray(fn()).reshape(-1)
                    got = {j: float(v[idx[j]]) for j in FJ_probe}
                    src = 'articulation_view.get_max_efforts()'
                except Exception:
                    got = None
        if got is None:
            finger_force_rb = {
                'source': None, 'values': None,
                'gap': ('**執行期讀不到 drive 出力上限**；USD 上 '
                        'drive:linear:physics:maxForce = 5，但本趟未能在執行期證實。'
                        '此為明列的缺口，不以 USD 值代替量測')}
            print('[drawer] **手指 drive 出力上限：執行期讀不到，列為缺口**', flush=True)
        else:
            finger_force_rb = {'source': src, 'values': got, 'gap': None}
            print(f'[drawer] 手指 drive 出力上限（執行期讀回，{src}）：{got}', flush=True)

    # --- 兩指關節本身的設定讀回（含 mimic）---
    # **看到 newton:mimicJoint 屬性不等於目前物理後端有採用**，所以分開記：
    # 一是 USD 上宣告了什麼，二是執行期能不能證實它生效。
    finger_joint_rb = None
    if GRASP_MODEL == 'friction':
        from pxr import UsdPhysics as _UP
        finger_joint_rb = {'joints': {}, 'mimic_declared': {},
                           'mimic_runtime_evidence': (
                               '**本趟無法由單一組對稱命令證實 mimic 是否生效**：'
                               '兩指收到相同命令，跟隨與獨立驅動的結果無法區分。'
                               '可用的間接證據是逐步的兩指實際位置 —— '
                               '一指被接觸擋住時另一指是否同步停止。列為缺口')}
        for jn in ('finger_joint1', 'finger_joint2'):
            jp = next((pr for pr in Usd.PrimRange.Stage(
                stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
                if pr.GetName() == jn and pr.IsA(_UP.PrismaticJoint)), None)
            if jp is None:
                continue
            J_ = _UP.PrismaticJoint(jp)
            d = {'prim': str(jp.GetPath()), 'axis': str(J_.GetAxisAttr().Get()),
                 'lower': float(J_.GetLowerLimitAttr().Get()),
                 'upper': float(J_.GetUpperLimitAttr().Get()),
                 'localRot0': [float(v) for v in (
                     [J_.GetLocalRot0Attr().Get().GetReal()]
                     + list(J_.GetLocalRot0Attr().Get().GetImaginary()))],
                 'api_schemas': [str(x) for x in jp.GetAppliedSchemas()]}
            dr = _UP.DriveAPI.Get(jp, 'linear')
            if dr:
                mf = dr.GetMaxForceAttr().Get()
                d['drive_max_force_usd'] = None if mf is None else float(mf)
            finger_joint_rb['joints'][jn] = d
            rel = jp.GetRelationship('newton:mimicJoint')
            if rel and rel.GetTargets():
                finger_joint_rb['mimic_declared'][jn] = {
                    'target': [str(t) for t in rel.GetTargets()],
                    'schema': 'NewtonMimicAPI',
                    'gearing': None, 'offset': None,
                    'note': ('宣告於 USD；**本模擬選用的是 physx variant，'
                             'Newton 專用 schema 是否被 PhysX 後端採用未經證實**')}
        print(f'[drawer] 手指關節讀回：{json.dumps(finger_joint_rb["joints"], ensure_ascii=False)[:260]}',
              flush=True)
        print(f'[drawer] mimic 宣告：{json.dumps(finger_joint_rb["mimic_declared"], ensure_ascii=False)[:200]}',
              flush=True)

    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for k, j in enumerate(ARM):
        q[idx[j]] = Q_START[k]; kp[idx[j]] = a.kp; kd[idx[j]] = a.kd
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in idx]
    for j in FJ:
        q[idx[j]] = F_OPEN; kp[idx[j]] = a.finger_kp; kd[idx[j]] = a.finger_kd
    robot.set_joint_positions(q)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
    # 固定底座下不呼叫 set_world_pose：位姿由 Xform 與 root_joint 決定。
    print('[drawer] view initialize ...', flush=True)
    # joint6 的反作用力列號要在暖機監看之前就決定
    j6_row_pre = None
    try:
        j6_row_pre = int(robot._articulation_view._metadata.joint_indices['joint6']) + 1
    except Exception:
        j6_row_pre = idx['joint6'] + 1

    drawer_v.initialize()
    grip_v.initialize()
    if finger_v is not None:
        finger_v.initialize()

    # --- 啟動前：確認每一個監看量都讀得到。讀不到就不要開始跑。---
    mon0 = {}
    tc0, tsrc0 = cpu_temp_read()
    mon0['cpu_temp'] = {'ok': bool(tc0 is not None),
                        'value': float(tc0) if tc0 is not None else None,
                        'source': str(tsrc0)}
    try:
        F_chk = np.array(robot.get_measured_joint_forces())
        mon0['wrist_force'] = {
            'ok': bool(F_chk.ndim == 2 and np.all(np.isfinite(F_chk))),
            'shape': [int(v) for v in F_chk.shape]}
    except Exception as e:
        mon0['wrist_force'] = {'ok': False, 'error': repr(e)}
    try:
        Mc = np.array(finger_v.get_contact_force_matrix(dt=a.physics_dt)) \
            if finger_v is not None else np.zeros((1, 1, 3))
        mon0['finger_contact'] = {'ok': bool(np.all(np.isfinite(Mc))),
                                  'shape': [int(v) for v in Mc.shape]}
    except Exception as e:
        mon0['finger_contact'] = {'ok': False, 'error': repr(e)}
    try:
        Mg = np.array(grip_v.get_contact_force_matrix(dt=a.physics_dt))
        mon0['gripper_contact'] = {'ok': bool(np.all(np.isfinite(Mg))),
                                   'shape': [int(v) for v in Mg.shape]}
    except Exception as e:
        mon0['gripper_contact'] = {'ok': False, 'error': repr(e)}
    bad_mon = [k for k, v in mon0.items() if not v.get('ok')]
    print(f'[drawer] 啟動前監看檢查 {json.dumps(mon0, ensure_ascii=False)}', flush=True)
    if bad_mon:
        print(f'[drawer] **監看量讀不到 {bad_mon}，不開始執行**', flush=True)
        json.dump({'schema': 'drawer_run/2', 'case': a.case,
                   'stop_reason': 'monitor_unavailable_at_start',
                   'monitors': mon0, 'unavailable': bad_mon},
                  open(os.path.join(a.out, 'drawer_run.json'), 'w'),
                  ensure_ascii=False, indent=2)
        return 7

    print('[drawer] view 就緒，暖機 30 步（暖機同樣納入監看）', flush=True)
    warm_fail = None
    for _ in range(30):
        world.step(render=False)
        # 暖機階段就開始監看：不能等進了正式迴圈才開始看溫度與力
        tcw, _srcw = cpu_temp_read()
        if tcw is None:
            warm_fail = ('monitor_failed_cpu_temp', '暖機中溫度讀不到')
            break
        if tcw >= CPU_LIMIT:
            warm_fail = ('cpu_temp', f'暖機中 {tcw:.1f} °C')
            break
        try:
            Fw = np.array(robot.get_measured_joint_forces())
            fw = np.linalg.norm(Fw[j6_row_pre][:3]) if j6_row_pre is not None else 0.0
        except Exception as e:
            warm_fail = ('monitor_failed_wrist_force', repr(e))
            break
        if fw > float(FRC['abort_threshold_n']):
            warm_fail = ('contact_force', f'暖機中 |F| {fw:.2f} N')
            break
    if warm_fail is not None:
        print(f'[drawer] **暖機階段停止：{warm_fail[0]}（{warm_fail[1]}）**', flush=True)
        json.dump({'schema': 'drawer_run/2', 'case': a.case,
                   'stop_reason': warm_fail[0], 'detail': warm_fail[1],
                   'monitors': mon0, 'phase': 'warmup'},
                  open(os.path.join(a.out, 'drawer_run.json'), 'w'),
                  ensure_ascii=False, indent=2)
        return 8

    p0, qq0 = robot.get_world_pose()
    BASE0 = (float(p0[0]), float(p0[1]), yaw_of(qq0))
    dp0, _ = drawer_v.get_world_poses()
    DY0 = float(dp0[0][1])
    print(f'[drawer] 初始底盤 ({BASE0[0]:.4f}, {BASE0[1]:.4f}) '
          f'yaw {math.degrees(BASE0[2]):.3f}°，抽屜 y0 {DY0:.5f}')

    prims = {}
    for nm in ('link_tcp', 'link6') + (
            ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link')
            if GRASP_MODEL == 'friction' else ()):
        pr = next((p for p in stage.Traverse() if p.GetName() == nm), None)
        if pr is None:
            print(f'[drawer] 找不到 {nm}，中止'); return 6
        prims[nm] = pr

    def world_T(pr):
        M = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = M.ExtractTranslation(); r = M.ExtractRotationQuat(); i = r.GetImaginary()
        R = np.array([[M[ri][ci] for ci in range(3)] for ri in range(3)])
        sc = np.linalg.norm(R, axis=1)
        R = R / sc[:, None]
        # USD 是列向量慣例（v' = v·M），要的旋轉矩陣是它的轉置
        return np.array([t[0], t[1], t[2]]), R.T, \
            np.array([r.GetReal(), i[0], i[1], i[2]])

    # joint6 在 measured_joint_forces 裡的列號（該連桿上游關節 = joint_index + 1）
    j6_row, j6_src = None, 'none'
    try:
        md = robot._articulation_view._metadata
        j6_row = int(md.joint_indices['joint6']) + 1
        j6_src = 'metadata.joint_indices'
        jnames = list(md.joint_names)
    except Exception as e:
        jnames = []
        print(f'[drawer] 讀不到 articulation metadata（{e}）')
    if j6_row is None:
        j6_row = idx['joint6'] + 1; j6_src = 'dof_names 順序（後備）'
    # 固定根部會改變 articulation 索引，**不沿用舊的列號**：一律由 metadata 重新
    # 對應。該列已在 evaluation/results/fixedbase_* 的短測中以已知 20 N 外力核對
    # 通過（Δ|F| = 20.0000 N）。
    print(f'[drawer] joint6 反作用力列號 {j6_row}（來源 {j6_src}，由 metadata 重新對應）')
    if jnames:
        print(f'[drawer] articulation joint 順序 {jnames}')

    def frames():
        """連接當下的同步快照（純讀取，**不建立任何關節**）。

        快照取得模式與正式連接共用這一段，確保兩者記錄的是同一組量。
        """
        Ma = UsdGeom.Xformable(stage.GetPrimAtPath(GRIP)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        Mb = UsdGeom.Xformable(stage.GetPrimAtPath(DRAWER)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        Mrel = Ma * Mb.GetInverse()
        ta = Ma.ExtractTranslation(); ra = Ma.ExtractRotationQuat()
        tb = Mb.ExtractTranslation(); rb = Mb.ExtractRotationQuat()
        t = Mrel.ExtractTranslation(); r = Mrel.ExtractRotationQuat()
        i = r.GetImaginary()
        qz = lambda q: [float(q.GetReal())] + [float(v) for v in q.GetImaginary()]
        return {'local_pos1': [float(t[0]), float(t[1]), float(t[2])],
                'local_rot1_wxyz': qz(r),
                'gripper_world_pos': [float(v) for v in ta],
                'gripper_world_rot_wxyz': qz(ra),
                'drawer_world_pos': [float(v) for v in tb],
                'drawer_world_rot_wxyz': qz(rb)}, Mrel

    def attach():
        _f, Mrel = frames()
        j = UsdPhysics.FixedJoint.Define(stage, JOINT_ATTACH)
        j.CreateBody0Rel().SetTargets([Sdf.Path(GRIP)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(DRAWER)])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        t = Mrel.ExtractTranslation(); r = Mrel.ExtractRotationQuat()
        i = r.GetImaginary()
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(float(r.GetReal()),
                                             float(i[0]), float(i[1]), float(i[2])))
        j.CreateJointEnabledAttr().Set(True)
        return _f

    def detach():
        pr = stage.GetPrimAtPath(JOINT_ATTACH)
        if pr.IsValid():
            UsdPhysics.Joint(pr).GetJointEnabledAttr().Set(False)
            return True
        return False

    # --- 力的量測：列號、符號與旋轉慣例 ---
    # 列號規則 joint_index + 1 已用**已知外力**校正過（evaluation/results 的
    # force_known 診斷）：在夾爪施世界 (0,0,−20) N，row 100 的 Δ|F| = 20.109 N
    # （誤差 0.5 %），finger_joint1 那列只變 0.003 N。
    # 符號：回報的是**父對子的反作用力**，與外加負載反號。
    # 偏置：靜止無接觸時該列約 5.2 N，而 joint6 以下的重量只有 4.05 N，
    #       約 1.1 N 的常數偏置未解釋 —— 所以這個量**不能直接當把手拉力**，
    #       只當「手腕傳遞力」的安全監看（30 N 門檻離 5 N 基線還有 25 N）。
    dist_mass_chk = 0.4126          # joint6 以下總質量（含 link6），由 URDF 樹算得
    _, l6R0, _ = world_T(prims['link6'])
    try:
        F0 = np.array(robot.get_measured_joint_forces())
        fw0 = l6R0 @ F0[j6_row][:3]
        force_rows = int(F0.shape[0])
    except Exception as e:
        fw0 = np.zeros(3); force_rows = -1
        print(f'[drawer] 讀不到 measured_joint_forces：{e}')
    exp_w = dist_mass_chk * 9.81
    # 旋轉慣例：stage 讀出的 TCP 姿態要與獨立 FK 算出的一致，否則轉到世界座標
    # 的力就是錯的。全零位時 R 與 R^T 幾乎相同，分辨不出來，所以要在**運動中**
    # 持續比對（見主迴圈的 rot_err）。
    qa0 = np.array([float(robot.get_joint_positions()[idx[j]]) for j in ARM])
    R_fk0 = fk_tcp(qa0, PARK)[:3, :3]
    _, R_st0, _ = world_T(prims['link_tcp'])
    rot_err0 = float(np.degrees(np.arccos(np.clip(
        (np.trace(R_st0.T @ R_fk0) - 1) / 2, -1, 1))))
    chk = {'rows': force_rows, 'row_used': j6_row,
           'f_world_static': [float(v) for v in fw0],
           'norm_N': float(np.linalg.norm(fw0)),
           'distal_weight_N': exp_w,
           'offset_N': float(np.linalg.norm(fw0)) - exp_w,
           'rot_conv_err_deg_at_start': rot_err0}
    print(f'[drawer] 力：列 {j6_row}/{force_rows}，靜止 |F| '
          f'{np.linalg.norm(fw0):.4f} N，joint6 以下重量 {exp_w:.4f} N，'
          f'偏置 {chk["offset_N"]:+.4f} N', flush=True)
    print(f'[drawer] 旋轉慣例：stage 與 FK 的 TCP 姿態差 {rot_err0:.4f}°（起始姿態）',
          flush=True)

    print('[drawer] 啟動 ROS 節點 ...', flush=True)
    rclpy.init()
    node = DrawerNode()
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor(); ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    LOG_COLS = ['t', 'phase', 'applied_seq', 'opening', 'opening_v',
                'slip', 'f_norm', 'f_pull', 'fc_norm', 'gc_norm', 'f_drawer',
                'tq_x', 'tq_y', 'tq_z', 'tq_norm',
                'rot_conv_err_deg', 'stage_vs_fk_err', 'cmd_lag',
                'e_par', 'e_perp', 'base_drift',
                'base_dyaw_deg', 'track_err', 'limit_margin'] + ARM + [
                # friction 版的量測欄；fixed 版恆為 0（**0 代表未套用該模型，
                # 不代表量到零接觸**，兩者由 grasp_model 區分）
                'fc1_norm', 'fc2_norm',
                # **不是接觸面的法向／摩擦力**：只是把接觸合力投影到
                # 「兩指原點連線」這條閉合軸及其垂直方向。把手是圓柱，
                # 接觸法向未必沿該軸，所以照分量本義命名。
                'f_along_close_axis', 'f_perp_close_axis',
                # 量測補強：兩指的命令與實際關節位置
                # 手指三路分開記錄，**不可互相代替**：
                'fj_req',          # 軌跡請求值
                'fj_sent',         # 閘輸出、實際送入 articulation 的值
                'fj1_act', 'fj2_act']   # 量測關節位置
    log, events, stop_reason = [], [], 'sim_limit'
    prev_ov, prev_ov_t = None, None
    rot_err_max, stage_fk_err_max = 0.0, 0.0
    TEMP_EVERY = 50                    # 每這麼多樣本量一次溫度（0.5 s @100 Hz）
    last_temp = tc0
    temp_max, temp_max_t = tc0, 0.0    # 本趟實測峰值（回報用）
    f_over_n = 0                       # 連續超過門檻的樣本數
    # 手指接觸中止：門檻、判準基礎、持續時間**全部由案例明訂**，
    # 與手腕傳遞力的規則各自獨立，不共用計數器也不共用持續時間。
    FG_ABORT_N = float(GRIP_SPEC.get('finger_contact_abort_n', 0.0) or 0.0)
    FG_SUSTAIN = int(round(float(GRIP_SPEC.get('finger_contact_abort_sustained_s', 0.0) or 0.0)
                           / a.physics_dt))
    FG_BASIS = str(GRIP_SPEC.get('finger_contact_abort_basis', ''))
    fg_over_n = 0
    fg_peak, fg_peak_t = 0.0, None
    F_SUSTAIN = int(round(float(FRC.get('abort_sustained_s', 0.0))
                          / a.physics_dt))
    f_peak, f_peak_t, f_peak_ph = 0.0, None, None
    DR_M = float(SPEC['drawer']['physics']['mass_kg'])
    DR_D = float(SPEC['drawer']['physics']['linear_damping'])
    applied_seq = None
    frozen_q = None
    coupled = False
    hold_ok_t = None
    arrived_t = None
    phase_t0 = {'': 0.0}
    last_phase = None
    w0 = time.monotonic()
    nxt = time.monotonic()
    print('[drawer] 進入物理主迴圈', flush=True)
    dist_mass = 0.25 + 2 * 0.0163      # 夾爪殼 + 兩指，用於座標慣例自我核對

    monitor_fail = None
    align_mismatch = False
    snapshot_done = False
    post_stop = []
    n_step = 0
    # --- 閉合放行閘（friction 版）---
    # 邏輯在 evaluation/grip_gate.py，已由 evaluation/test_grip_gate.py
    # **不開模擬器**覆蓋：初始姿態投影碰巧為零、只有位置合格但姿態錯誤、
    # 保持途中失效、延遲放行後仍完整執行斜坡、逾時中止。
    GATE_BAR = np.array([float(BAR[0]) + POSE[0], float(BAR[1]) + POSE[1],
                         float(BAR[2])])
    # 把手參考點在**抽屜剛體座標系**的固定位置（= drawer_unit.yaml handle.bar.center）
    BAR_LOCAL = np.array([float(BAR[0]), float(BAR[1]), float(BAR[2])])
    R_DES = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    grip_gate = GripGate(
        offset_limit_mm=a.grip_gate_offset_mm, pos_perp_max_mm=a.grip_gate_pos_mm,
        rot_max_deg=a.grip_gate_rot_deg, track_max_rad=a.grip_gate_track_rad,
        hold_s=a.grip_gate_hold_s, timeout_s=a.grip_gate_timeout_s,
        ramp_s=a.grip_close_ramp_s) if GRASP_MODEL == 'friction' else None
    gate_metrics = []          # [t, pos_perp_mm, rot_deg, offset_mm]
    grip_hold = GripHold(contact_min_n=a.grip_hold_contact_n,
                         rel_pos_max_mm=a.grip_hold_pos_mm,
                         rel_rot_max_deg=a.grip_hold_rot_deg,
                         hold_s=a.grip_hold_s) if GRASP_MODEL == 'friction' else None
    # 同步記錄夾爪與抽屜的實際位置、四元數與模擬時間。
    # 相對量一律由 T_gripper→drawer 計算，**不用世界座標差代替**。
    pose_trace = []
    pull_ref = None            # 拉動起點的相對位姿
    pull_metrics = []          # [t, slip_mm, rot_deg]
    pull_block = None
    last_fj_sent = float(F_OPEN)
    finger_pose_trace = []          # [t, p1xyz, p2xyz]：兩指墊世界位置逐步紀錄
    while True:
      try:
        world.step(render=False)
        t = float(world.current_time)
        # 執行時錄影：**這一趟真正在跑的畫面**。算繪只讀場景、不寫回任何狀態，
        # 也不介入中止判斷；代價是牆鐘變慢（模擬時間語意不變）。
        if rec_cam is not None and (n_step % rec_every) == 0:
            world.render()
            _img = rec_cam.get_rgba()
            if _img is not None and len(_img):
                _imageio.imwrite(
                    os.path.join(rec_dir, f'f{len(rec_index):06d}.png'),
                    np.asarray(_img)[:, :, :3].astype(np.uint8))
                rec_index.append([len(rec_index), round(t, 4)])
        n_step += 1
        sec = int(t); nsec = int(round((t - sec) * 1e9))
        c = Clock(); c.clock.sec = sec; c.clock.nanosec = min(nsec, 999999999)
        node.clock_pub.publish(c)

        # 底盤由 world→根 的固定關節固定；**正常迴圈不做任何底盤位置／速度覆寫**。

        ph = node.phase
        if ph != last_phase:
            phase_t0[ph] = t; last_phase = ph

        # --- 事件（在套用設定點之前處理，這樣連接就發生在該時刻的物理步）---
        while node.pending:
            e = node.pending.pop(0)
            ev = e['event']
            dpa, _ = drawer_v.get_world_poses()
            tcp_a, _, _ = world_T(prims['link_tcp'])
            info = {'event': ev, 'sim_t': t, 'seq': e.get('seq'),
                    'drawer_y_before': float(dpa[0][1]),
                    'tcp_before': tcp_a.tolist()}
            if ev == 'engage':
                if a.snapshot_only:
                    # **只記錄，不連接**：同一個物理時刻的同步快照
                    info['attach_frames'], _ = frames()
                    info['snapshot_only'] = True
                    snapshot_done = True
                    print('[drawer] 快照取得模式：已記錄同步快照，**未建立連接**',
                          flush=True)
                elif a.no_attach:
                    # 照常記錄同步快照，**但不建立固定關節**，也不設 coupled
                    info['attach_frames'], _ = frames()
                    info['no_attach'] = True
                    print('[drawer] 診斷模式：已記錄同步快照，'
                          '**未建立夾爪—抽屜固定關節**，繼續重播', flush=True)
                elif GRASP_MODEL == 'fixed_attachment':
                    info['attach_frames'] = attach()
                    if ALIGN_REF is not None:
                        af = info['attach_frames']
                        dp = float(np.linalg.norm(
                            np.array(af['gripper_world_pos'])
                            - np.array(ALIGN_REF['gripper_world_pos'])))
                        da = DAL.ang_deg(
                            DAL.quat_R(af['gripper_world_rot_wxyz']),
                            DAL.quat_R(ALIGN_REF['gripper_world_rot_wxyz']))
                        dd = float(np.linalg.norm(
                            np.array(af['drawer_world_pos'])
                            - np.array(ALIGN_REF['drawer_world_pos'])))
                        info['align_check'] = {
                            'ref_run': ALIGN_REF['source_run'],
                            'gripper_pos_diff_m': dp, 'gripper_rot_diff_deg': da,
                            'drawer_pos_diff_m': dd,
                            'tol_m': a.align_tol_m, 'tol_deg': a.align_tol_deg,
                            'ok': bool(dp <= a.align_tol_m
                                       and da <= a.align_tol_deg
                                       and dd <= a.align_tol_m)}
                        print(f'[drawer] 對準快照核對：夾爪位置差 {dp*1000:.4f} mm、'
                              f'姿態差 {da:.5f}°、抽屜位置差 {dd*1000:.4f} mm '
                              f'→ {"相符" if info["align_check"]["ok"] else "**不符**"}',
                              flush=True)
                        align_mismatch = not info['align_check']['ok']
                    coupled = True
                else:
                    coupled = True      # friction：靠手指命令，不建關節
            elif ev == 'release':
                info['detached'] = detach() if GRASP_MODEL == 'fixed_attachment' else True
                coupled = False
            world.step(render=False)    # 讓連接／解除在下一步生效後再量
            dpb, _ = drawer_v.get_world_poses()
            tcp_b, _, _ = world_T(prims['link_tcp'])
            info['drawer_y_after'] = float(dpb[0][1])
            info['drawer_jump_mm'] = (info['drawer_y_after']
                                      - info['drawer_y_before']) * 1000.0
            info['tcp_jump_mm'] = float(np.linalg.norm(tcp_b - tcp_a)) * 1000.0
            events.append(info)
            node.say({'event_applied': ev, 'sim_t': t,
                      'drawer_jump_mm': info['drawer_jump_mm'],
                      'tcp_jump_mm': info['tcp_jump_mm']})
            print(f'[drawer] 事件 {ev} @ sim {t:.3f}  抽屜跳動 '
                  f'{info["drawer_jump_mm"]:+.4f} mm  TCP 跳動 '
                  f'{info["tcp_jump_mm"]:+.4f} mm', flush=True)

        # --- 套用設定點 ---
        snap = node.snap
        if frozen_q is not None:
            tgt = robot.get_joint_positions()
            for k, j in enumerate(ARM):
                tgt[idx[j]] = frozen_q[k]
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))
        elif snap is not None:
            applied_seq, _, q_cmd = snap
            tgt = robot.get_joint_positions()
            for k, j in enumerate(ARM):
                tgt[idx[j]] = q_cmd[k]
            fs = node.fsnap
            if fs is not None:
                _fv = fs[1]
                if grip_gate is not None:
                    _fv = grip_gate.finger_command(t, float(fs[1]), F_OPEN,
                                                   float(GRIP_SPEC.get(
                                                       'finger_cmd_closed', 0.0)))
                last_fj_sent = float(_fv)
                for j in FJ:
                    tgt[idx[j]] = _fv
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))

        # --- 量測 ---
        qm = robot.get_joint_positions()
        qa = np.array([float(qm[idx[j]]) for j in ARM])
        tcp_p, tcp_R, tcp_q = world_T(prims['link_tcp'])
        _, l6_R, _ = world_T(prims['link6'])
        dp, dq_w = drawer_v.get_world_poses()
        dv = drawer_v.get_velocities()[0]
        opening = DY0 - float(dp[0][1])
        opening_v = -float(dv[1])
        # 夾持關係：TCP 對「當下開度下的預定夾持點」的偏差
        exp_tcp = DA.grasp_tcp_world(SPEC, POSE, opening)
        slip = tcp_p - exp_tcp
        slip_n = float(np.linalg.norm(slip))
        # 讀不到就拋例外，**不補零**。補零等於把「監看失效」偽裝成「力為 0」。
        try:
            F = np.array(robot.get_measured_joint_forces())
            if F.ndim != 2 or j6_row >= F.shape[0]:
                raise MonitorFailure('wrist_force', f'shape {F.shape} row {j6_row}')
            f_local = F[j6_row][:3]
            tq_local = F[j6_row][3:6]
            if not np.all(np.isfinite(F[j6_row])):
                raise MonitorFailure('wrist_force', f'非有限值 {F[j6_row]}')
        except MonitorFailure:
            raise
        except Exception as e:
            raise MonitorFailure('wrist_force', repr(e))
        f_world = l6_R @ f_local
        f_norm = float(np.linalg.norm(f_world))
        f_pull = float(AXIS @ f_world)
        # 力矩分量：力隨開度成長而與速度無關時，要靠力矩才分得出是哪一種負載
        # （抽屜外伸造成的傾覆力矩 vs 沿軸的推拉）。只記力分不出來。
        # **參考點是 link6 原點**（reaction 本來就在該連桿座標系原點取），
        # 這裡只把分量旋轉到世界座標軸，**沒有把參考點移到把手或抽屜**。
        # 所以它不是抽屜的傾覆力矩，兩者不可互稱。
        tq_world = l6_R @ tq_local
        tq_norm = float(np.linalg.norm(tq_world))
        fc = np.zeros(3)
        per_finger = None
        if finger_v is not None:
            try:
                M = np.array(finger_v.get_contact_force_matrix(dt=a.physics_dt))
                fc = M.reshape(-1, 3).sum(axis=0)
                if not np.all(np.isfinite(fc)):
                    raise MonitorFailure('finger_contact', f'非有限值 {fc}')
                per_finger = M.reshape(M.shape[0], -1, 3).sum(axis=1)
            except MonitorFailure:
                raise
            except Exception as e:
                raise MonitorFailure('finger_contact', repr(e))
        elif GRASP_MODEL == 'friction':
            # 摩擦夾持靠接觸傳力；**讀不到接觸資料不能當成零**
            raise MonitorFailure('finger_contact', 'friction 版缺少手指接觸 view')
        fc_n = float(np.linalg.norm(fc))
        # 兩指的命令與實際關節位置（**量測，不影響控制**）：
        # 「手指停止移動」不等於夾住，要看實際位置與命令的關係。
        # --- 更新閉合放行閘（條件同時檢查，不只單一投影）---
        if grip_gate is not None:
            _ok = True
            try:
                _pf1, _Rg, _ = world_T(prims['uflite_finger1'])
                _pf2, _, _ = world_T(prims['uflite_finger2'])
                _ok = bool(np.all(np.isfinite(_pf1)) and np.all(np.isfinite(_pf2)))
            except Exception:
                _pf1 = _pf2 = np.zeros(3); _Rg = np.eye(3); _ok = False
            _ctr = 0.5 * (_pf1 + _pf2)
            _ax = _Rg @ np.array([0.0, 1.0, 0.0])
            _v = _ctr - GATE_BAR
            _off = float(np.dot(_v, _ax)) * 1000.0
            _perp = float(np.linalg.norm(_v - _ax * np.dot(_v, _ax))) * 1000.0
            _rot = float(DAL.ang_deg(_Rg, R_DES))
            # 追蹤誤差在本迴圈稍後才算，這裡用**同一定義**自行取值，
            # 不把尚未賦值的變數帶進閘（第一版就是這個 UnboundLocalError）
            _trk = (float(np.abs(qa - snap[2]).max())
                    if snap is not None else 0.0)
            grip_gate.update(t, phase=ph, data_ok=_ok, pos_perp_mm=_perp,
                             rot_deg=_rot, offset_mm=_off, track_rad=_trk)
            gate_metrics.append([round(t, 4), round(_perp, 4), round(_rot, 5),
                                 round(_off, 4)])
            if grip_gate.released and grip_gate.release_t == t:
                print(f'[drawer] **閉合放行閘：通過** @ sim {t:.3f}'
                      f'（位置 {_perp:.3f} mm、姿態 {_rot:.4f}°、'
                      f'對中 {_off:+.4f} mm、追蹤 {_trk:.6f} rad）', flush=True)

        _fs = node.fsnap
        fj_req = float(_fs[1]) if _fs is not None else float('nan')
        fj_sent = float(last_fj_sent)
        fj_act = [float(qm[idx[j]]) if j in idx else 0.0
                  for j in ('finger_joint1', 'finger_joint2')]
        # 逐指：模長（中止判準）＋沿閉合軸的分量分解（**非接觸法向／摩擦**）
        fpn = fpt = 0.0
        fc_each = [0.0, 0.0]
        if per_finger is not None and GRASP_MODEL == 'friction':
            if per_finger.shape[0] < 2:
                raise MonitorFailure(
                    'finger_contact', f'逐指矩陣只有 {per_finger.shape[0]} 列')
            fc_each = [float(np.linalg.norm(per_finger[0])),
                       float(np.linalg.norm(per_finger[1]))]
            p1, _, _ = world_T(prims['uflite_finger1'])
            p2, _, _ = world_T(prims['uflite_finger2'])
            finger_pose_trace.append([round(t, 4), *[round(float(v), 6) for v in p1],
                                      *[round(float(v), 6) for v in p2]])
            ax = p1 - p2
            nrm = float(np.linalg.norm(ax))
            if nrm > 1e-9:
                ax = ax / nrm
                # 正向 = 沿閉合軸的分量（逐指各自投影後取絕對值再相加）
                fpn = float(abs(np.dot(per_finger[0], ax))
                            + abs(np.dot(per_finger[1], ax)))
                tv = [per_finger[i] - ax * np.dot(per_finger[i], ax) for i in (0, 1)]
                fpt = float(np.linalg.norm(tv[0]) + np.linalg.norm(tv[1]))

        # --- 同步記錄夾爪與抽屜位姿；相對量用 T_gripper→drawer ---
        if GRASP_MODEL == 'friction':
            _pg, _Rg2, _qg = world_T(prims['uflite_gripper_link'])
            _pd = np.asarray(dp[0], float)
            _qd = np.asarray(dq_w[0], float)
            _Rd = DAL.quat_R(_qd)
            _relR = _Rg2.T @ _Rd
            # **驗收位置在把手參考點**（grip_hold_criteria v2）：
            #   p_GH = R_G^T (p_D + R_D · p_DH − p_G)
            # 抽屜剛體原點距夾爪約 654 mm、把手約 69 mm；用原點會把夾爪的
            # 微小轉動經力臂放大成視在位移（實測 0.4385 vs 0.0450 mm）。
            _relp = _Rg2.T @ ((_pd + _Rd @ BAR_LOCAL) - _pg)
            # 抽屜剛體原點：**保留為診斷量**，不作為把手滑移判準
            _relp_org = _Rg2.T @ (_pd - _pg)
            pose_trace.append([round(t, 4)]
                              + [round(float(v), 7) for v in _pg]
                              + [round(float(v), 7) for v in _qg]
                              + [round(float(v), 7) for v in _pd]
                              + [round(float(v), 7) for v in _qd]
                              + [round(float(np.linalg.norm(_relp_org)), 7)])
            # 夾持驗收：**閉合斜坡實際完成後**才開始累積（不用請求軌跡的時間）
            _cs = grip_gate.close_start_t if grip_gate is not None else None
            if (_cs is not None
                    and t >= _cs + grip_gate.cfg['ramp_s'] - 1e-9):
                grip_hold.update(t, f1_n=fc_each[0], f2_n=fc_each[1],
                                 rel_p=_relp, rel_R=_relR,
                                 ang_deg_fn=DAL.ang_deg)
            # 拉動段：相對變化自**拉動起點**計算
            if ph == 'pull':
                if pull_ref is None:
                    pull_ref = (_relp.copy(), _relR.copy())
                _sl = float(np.linalg.norm(_relp - pull_ref[0])) * 1000.0
                _rr = float(DAL.ang_deg(_relR, pull_ref[1]))
                pull_metrics.append([round(t, 4), round(_sl, 5), round(_rr, 5)])

        try:
            gcm = np.array(grip_v.get_contact_force_matrix(dt=a.physics_dt))
            gc = gcm.reshape(-1, 3).sum(axis=0)
            if not np.all(np.isfinite(gc)):
                raise MonitorFailure('gripper_contact', f'非有限值 {gc}')
        except MonitorFailure:
            raise
        except Exception as e:
            raise MonitorFailure('gripper_contact', repr(e))
        gc_n = float(np.linalg.norm(gc))
        # **依抽屜動力學重建的軸向外力估計** F̂ = m·â + m·d·v。
        # 這是重建值，不是拉力的直接量測：它依賴阻尼模型（線性阻尼 d）、
        # 速度的差分，以及「軸向沒有其他作用力」這個假設。
        # 它的好處是不依賴 articulation 的列號對應，可與手腕反作用力互相對照。
        #
        # 時間基準：engage / release 會額外呼叫一次 world.step()，那些樣本的
        # 兩次速度量測相隔的模擬時間**不是** physics_dt。固定除以 physics_dt
        # 會讓那幾個樣本的加速度項偏小，連帶影響峰值，所以改用實際時間差。
        if prev_ov is not None and prev_ov_t is not None and t > prev_ov_t:
            dt_ov = t - prev_ov_t
            a_dr = (opening_v - prev_ov) / dt_ov
        else:
            dt_ov, a_dr = a.physics_dt, 0.0
        prev_ov, prev_ov_t = opening_v, t
        f_drawer = DR_M * a_dr + DR_M * DR_D * opening_v
        # 旋轉慣例的持續比對：運動中 R 與 R^T 會分開，差角應保持在 0 附近
        T_fk = fk_tcp(qa, PARK)
        R_fk = T_fk[:3, :3]
        rot_err = float(np.degrees(np.arccos(np.clip(
            (np.trace(tcp_R.T @ R_fk) - 1) / 2, -1, 1))))
        rot_err_max = max(rot_err_max, rot_err)
        # Isaac stage 讀出的 TCP 位置，與**同一組量測關節角**的離線 FK 之差。
        # **不是 URDF 模型差異**：離線核對過，omni_bot_manip.urdf 與展開的 9-DOF
        # 檔在夾爪連桿上運動學完全相同（0.0000 mm / 0.00000°）。
        # 這裡量到的 0.26–0.52 mm 成因未確立，先如實記錄、不下標籤。
        # 與「命令還沒追上」的動態落後是兩回事，不能混在 slip 裡一起看。
        stage_fk_err = float(np.linalg.norm(tcp_p - T_fk[:3, 3]))
        stage_fk_err_max = max(stage_fk_err_max, stage_fk_err)
        # 命令對應的 TCP 與實際 TCP 的差 = 追蹤落後，**要分解**：
        #   e_par  = â^T e        沿滑動軸（抽屜可以讓開的方向）
        #   e_perp = e − â e_par  垂直滑動軸（滑動關節會剛性抵抗的方向）
        # 總落後 5–6 mm **不等於**橫向偏離 5–6 mm；要判斷高力是否隨橫向誤差
        # 增加，只能看 e_perp，不能用總誤差推論。
        if snap is not None:
            e_vec = tcp_p - fk_tcp(snap[2], PARK)[:3, 3]
            e_par = float(AXIS @ e_vec)
            e_perp_v = e_vec - AXIS * e_par
            e_perp = float(np.linalg.norm(e_perp_v))
            lag = float(np.linalg.norm(e_vec))
        else:
            e_par, e_perp, lag = 0.0, 0.0, 0.0
        bp, bq = robot.get_world_pose()
        drift = math.hypot(float(bp[0]) - BASE0[0], float(bp[1]) - BASE0[1])
        dyaw = abs(yaw_of(bq) - BASE0[2])
        trk = float(np.abs(qa - (snap[2] if snap is not None else qa)).max()) \
            if snap is not None else 0.0
        lm = min(min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k])
                 for k in range(6))

        node.farr(node.dr_pub, [t, opening, opening_v])
        node.farr(node.gr_pub, [t, slip[0], slip[1], slip[2], slip_n,
                                1.0 if coupled else 0.0])
        node.farr(node.ct_pub, [t, f_world[0], f_world[1], f_world[2], f_norm,
                                f_pull, fc[0], fc[1], fc[2], fc_n,
                                gc[0], gc[1], gc[2], gc_n, f_drawer])
        js = JointState(); js.header.stamp.sec = sec
        js.header.stamp.nanosec = min(nsec, 999999999)
        js.name = names; js.position = [float(v) for v in qm]
        node.js_pub.publish(js)
        ps = PoseStamped(); ps.header.stamp = js.header.stamp
        ps.header.frame_id = 'world'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = [float(v) for v in tcp_p]
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, \
            ps.pose.orientation.z = [float(v) for v in tcp_q]
        node.tcp_pub.publish(ps)

        # 欄名與資料列長度必須一致 —— 曾經只加欄名沒加值，讓其後欄位全部錯位，
        # 事後讀到的「track_err」其實是 limit_margin。這裡每步檢查一次。
        log.append([round(t, 4), ph, applied_seq, round(opening, 6),
                    round(opening_v, 5), round(slip_n, 6), round(f_norm, 4),
                    round(f_pull, 4), round(fc_n, 4), round(gc_n, 4),
                    round(f_drawer, 5),
                    round(tq_world[0], 4), round(tq_world[1], 4),
                    round(tq_world[2], 4), round(tq_norm, 4),
                    round(rot_err, 5), round(stage_fk_err, 6), round(lag, 6),
                    round(e_par, 6), round(e_perp, 6), round(drift, 5),
                    round(math.degrees(dyaw), 4), round(trk, 5), round(lm, 5)]
                   + [round(float(v), 6) for v in qa]
                   + [round(fc_each[0], 4), round(fc_each[1], 4),
                      round(fpn, 4), round(fpt, 4),
                      round(fj_req, 6), round(fj_sent, 6),
                      round(fj_act[0], 6), round(fj_act[1], 6)])

        if len(log) == 1 and len(log[0]) != len(LOG_COLS):
            raise RuntimeError(f'log 欄名 {len(LOG_COLS)} 個 vs 資料列 '
                               f'{len(log[0])} 個 —— 欄位錯位，中止')

        # --- 到位判定 ---
        if abs(opening - TARGET) <= float(TOL['opening_m']):
            if hold_ok_t is None:
                hold_ok_t = t
            elif arrived_t is None and t - hold_ok_t >= float(TOL['opening_hold_s']):
                arrived_t = t
                print(f'[drawer] 開度到位並保持 {TOL["opening_hold_s"]:.1f} s @ sim {t:.3f}',
                      flush=True)
        else:
            hold_ok_t = None

        # --- 停止條件 ---
        # 手指接觸中止（friction 版才啟用）：判準是**每一指各自**的接觸力模長，
        # 任一指超過門檻即計數；持續時間為 0 代表單一樣本即中止。
        if GRASP_MODEL == 'friction' and FG_ABORT_N > 0:
            fg_max = max(fc_each)
            if fg_max > fg_peak:
                fg_peak, fg_peak_t = fg_max, t
            if fg_max > FG_ABORT_N:
                fg_over_n += 1
            else:
                fg_over_n = 0
        if f_norm > f_peak:
            f_peak, f_peak_t, f_peak_ph = f_norm, t, ph
        if f_norm > float(FRC['abort_threshold_n']):
            f_over_n += 1
        else:
            f_over_n = 0
        stop = None
        if (a.stop_at_cmd_t > 0 and snap is not None
                and snap[1] >= a.stop_at_cmd_t):
            # 依**指令進度**收尾（t_sched），不依賴抽屜是否真的移動
            stop = 'diagnostic_cmd_progress'
        elif a.stop_at_opening_m > 0 and opening >= a.stop_at_opening_m:
            # 診斷用的區間上限：命令序列與完整版相同，只是提前收尾
            stop = 'diagnostic_opening_limit'
        elif snapshot_done:
            # 沒有建立任何連接，沒有負載要卸，直接以預設處置收尾
            stop = 'snapshot_captured'
        elif align_mismatch:
            # 實際快照與產生軌跡時用的那份不符 ⇒ 後面的參考是對錯位姿算的。
            # 不拿錯位的參考硬跑，直接停。
            stop = 'align_snapshot_mismatch'
        # 門檻 30 N 未變；要**連續**超過 abort_sustained_s 才中止，
        # 這樣建立約束的單步暫態不會被當成過載（見案例設定的說明）。
        if f_over_n > F_SUSTAIN:
            stop = 'contact_force'
        elif GRASP_MODEL == 'friction' and FG_ABORT_N > 0 and fg_over_n > FG_SUSTAIN:
            stop = 'finger_contact_force'
        elif (grip_hold is not None and ph == 'pull' and pull_block is None
              and not grip_hold.satisfied):
            # **不以預排時間代替**：拉動只有在實際資料連續滿足夾持判準後才觸發
            pull_block = {'sim_t': t, 'held_s': grip_hold.held_s(t),
                          'last_fail': grip_hold.last_fail,
                          'worst': dict(grip_hold.worst)}
            print(f'[drawer] **夾持驗收未由實際資料滿足，不拉動** @ sim {t:.3f}'
                  f'（已連續 {grip_hold.held_s(t):.3f} s < '
                  f'{grip_hold.cfg["hold_s"]}；{grip_hold.last_fail}）', flush=True)
            stop = 'grip_hold_not_met'
        elif (grip_hold is not None and ph == 'pull' and pull_ref is not None
              and pull_metrics and pull_metrics[-1][1] > a.pull_slip_mm):
            stop = 'grip_slip'
        elif (grip_hold is not None and ph == 'pull' and pull_ref is not None
              and pull_metrics and pull_metrics[-1][2] > a.pull_rot_deg):
            stop = 'grip_rotation'
        elif grip_gate is not None and grip_gate.abort:
            # 命令要求閉合但對中始終未達標 ⇒ 停，不在未對中的狀態下硬夾
            stop = grip_gate.abort
        elif drift > float(TOL['base_drift_m']):
            stop = 'base_drift'
        elif math.degrees(dyaw) > float(TOL['base_yaw_drift_deg']):
            stop = 'base_drift'
        elif lm < float(TOL['joint_limit_margin_rad']):
            stop = 'joint_limit_margin'
        elif trk > float(TOL['joint_track_err_rad']):
            stop = 'joint_track_err'
        elif coupled and slip_n > float(TOL['grasp_slip_m']):
            stop = 'grasp_lost'
        elif opening > float(SPEC['drawer']['joint']['upper']) - 0.005:
            stop = 'drawer_limit'
        elif ph in TMO and t - phase_t0.get(ph, t) > float(TMO[ph]):
            stop = 'phase_timeout'
        elif t > a.sim_limit:
            stop = 'sim_limit'
        elif time.monotonic() - w0 > a.wall_limit:
            stop = 'wall_timeout'
        elif len(log) % TEMP_EVERY == 0:
            tc, tsrc_now = cpu_temp_read()
            if tc is None:
                # 讀不到不是「沒超溫」。啟動時已確認可讀，中途讀不到是監看失效。
                raise MonitorFailure('cpu_temp', f'來源 {tsrc_now} 回傳 None')
            last_temp = tc
            if tc > temp_max:
                temp_max, temp_max_t = tc, t
            if tc >= CPU_LIMIT:
                stop = 'cpu_temp'

        if stop is not None:
            handling = ('release_coupling_then_freeze'
                        if stop in ('diagnostic_opening_limit',
                                    'diagnostic_cmd_progress')
                        else CASE['stop_handling'].get(
                            stop, CASE['stop_handling']['default']))
            print(f'[drawer] 停止：{stop}（處置 {handling}）@ sim {t:.3f}', flush=True)
            if handling == 'release_coupling_then_freeze':
                # 凍結設定點**不會**卸力：位置驅動會持續施力。先解除耦合。
                if GRASP_MODEL == 'fixed_attachment':
                    detach()
                coupled = False
                if snap is not None:
                    tgt = robot.get_joint_positions()
                    for j in FJ:
                        tgt[idx[j]] = F_OPEN
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=tgt))
            frozen_q = qa.copy()
            stop_reason = stop
            node.say({'stop': stop, 'handling': handling, 'sim_t': t})
            # 停止後的步進**也要留紀錄**：否則「釋放／解除連接後物體是否停穩」
            # 在資料裡沒有依據（前一趟就是這樣，只知道最終開度、不知道過程）。
            post_stop = []
            for _ in range(60):
                world.step(render=False)
                _p, _ = drawer_v.get_world_poses()
                _v = drawer_v.get_velocities()[0]
                post_stop.append([round(float(world.current_time), 4),
                                  round(DY0 - float(_p[0][1]), 6),
                                  round(-float(_v[1]), 6)])
            break

        if a.rtf > 0:
            nxt += a.physics_dt / a.rtf
            sl = nxt - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                nxt = time.monotonic()
      except MonitorFailure as mf:
        # 監看失效：我們**不知道**力或溫度是多少，所以走保守處置（先解除耦合），
        # 與「量到超限」分開記錄，兩者不能混為一談。
        monitor_fail = {'what': mf.what, 'detail': mf.detail,
                        'sim_t': float(world.current_time)}
        stop_reason = f'monitor_failed_{mf.what}'
        print(f'[drawer] **監看失效：{mf.what}（{mf.detail}）** '
              f'→ 保守處置：解除耦合後凍結', flush=True)
        if GRASP_MODEL == 'fixed_attachment':
            detach()
        coupled = False
        try:
            qa_now = np.array([float(robot.get_joint_positions()[idx[j]]) for j in ARM])
            frozen_q = qa_now.copy()
            tgt = robot.get_joint_positions()
            for j in FJ:
                tgt[idx[j]] = F_OPEN
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))
        except Exception:
            pass
        node.say({'stop': stop_reason, 'handling': 'release_coupling_then_freeze',
                  'monitor_failure': monitor_fail})
        for _ in range(20):
            world.step(render=False)
        break

    tc, tsrc = cpu_temp_read()
    dp, _ = drawer_v.get_world_poses()
    final_open = DY0 - float(dp[0][1])
    tcp_p, _, _ = world_T(prims['link_tcp'])
    out = {
        'schema': 'drawer_run/1', 'case': a.case, 'grasp_model': GRASP_MODEL,
        'sha256_16': SHA, 'physics_dt': a.physics_dt, 'rtf': a.rtf,
        'park': list(PARK), 'pose': list(POSE), 'drawer_y0': DY0,
        'authored': authored,
        'force_measure': {'row': j6_row, 'row_source': j6_src,
                          'joint_names': jnames,
                          'frame': 'link6 世界旋轉 @ 子連桿座標系的反作用力',
                          'axis_world': AXIS.tolist(),
                          'distal_mass_kg': dist_mass,
                          'distal_weight_N': dist_mass * 9.81,
                          'static_check': chk,
                          'note': ('F_pull 未扣除夾爪慣性項，是「手腕沿抽屜軸傳遞'
                                   '的力」，不是純把手拉力')},
        'grip_spec': GRIP_SPEC if GRASP_MODEL == 'friction' else None,
        'grasp_friction_readback': fric_rb,
        'contact_introspect': contact_introspect,
        'grip_gate': (grip_gate.summary() if grip_gate is not None else None),
        'grip_gate_metrics': gate_metrics if grip_gate is not None else None,
        'grip_gate_metric_cols': ['t', 'pos_perp_mm', 'rot_deg', 'offset_mm'],
        'grip_hold': (grip_hold.summary() if grip_hold is not None else None),
        'grip_hold_blocked_pull': pull_block,
        'pose_trace': pose_trace if GRASP_MODEL == 'friction' else None,
        'pose_trace_cols': ['t', 'gx', 'gy', 'gz', 'gqw', 'gqx', 'gqy', 'gqz',
                            'dx', 'dy', 'dz', 'dqw', 'dqx', 'dqy', 'dqz',
                            'drawer_origin_rel_norm_m'],
        'grip_hold_criteria': {
            'spec': 'evaluation/results/specs/grip_hold_criteria_v2.yaml',
            'version': 'v2',
            'acceptance_point': 'drawer_handle_bar_center',
            'acceptance_point_frame': 'drawer_body_local',
            'acceptance_point_xyz_m': [float(BAR[0]), float(BAR[1]), float(BAR[2])],
            'formula': 'p_GH = R_G^T (p_D + R_D * p_DH - p_G)',
            'note': ('抽屜剛體原點的位移僅為診斷量，見 pose_trace 末欄；'
                     '**不得當作把手滑移**')},
        'pull_metrics': pull_metrics if GRASP_MODEL == 'friction' else None,
        'pull_metric_cols': ['t', 'slip_mm', 'rot_deg'],
        'pull_criteria': {'slip_max_mm': a.pull_slip_mm,
                          'rot_max_deg': a.pull_rot_deg,
                          'reference': '拉動起點的 T_gripper→drawer',
                          'note': '拉動後不要求把手在世界座標不動'},
        'finger_pose_trace': finger_pose_trace if GRASP_MODEL == 'friction' else None,
        'finger_pose_cols': ['t', 'f1x', 'f1y', 'f1z', 'f2x', 'f2y', 'f2z'],
        'finger_joint_readback': finger_joint_rb,
        'finger_drive_force_readback': finger_force_rb,
        'finger_abort_rule': ({
            'threshold_N': FG_ABORT_N, 'basis': FG_BASIS,
            'sustained_s': float(GRIP_SPEC.get('finger_contact_abort_sustained_s', 0.0) or 0.0),
            'sustain_samples_required': FG_SUSTAIN,
            'version': str(GRIP_SPEC.get('finger_contact_abort_rule_version', '')),
            'peak_per_finger_N': fg_peak, 'peak_sim_t': fg_peak_t,
            'note': ('與手腕傳遞力中止**各自獨立**：門檻、判準基礎與持續時間'
                     '都由案例明訂，未沿用手腕規則的 50 ms。'
                     '讀不到接觸資料走監看失效，不寫 0 繼續跑')}
            if GRASP_MODEL == 'friction' else None),
        'abort_rule': {
            'version': ('v1_instantaneous' if F_SUSTAIN <= 0
                        else 'v2_sustained'),
            'threshold_N': float(FRC['abort_threshold_n']),
            'sustained_s': float(FRC.get('abort_sustained_s', 0.0)),
            'sustain_samples_required': F_SUSTAIN,
            'note': ('v2 相對 v1 是**放寬**中止規則（不只是改判讀）：v1 單一樣本'
                     '超過即停，v2 要連續超過才停。兩者不是同一個判準，'
                     '不同版本的趟次不可直接相提並論。'
                     'v1 那次量到的 65.4 N 峰值不因為只出現一次就無害。')},
        'f_norm_peak': {'N': f_peak, 'sim_t': f_peak_t, 'phase': f_peak_ph},
        'diagnostic': {
            'is_diagnostic': bool(a.drawer_no_gravity or a.stop_at_opening_m > 0
                                  or a.no_attach or a.stop_at_cmd_t > 0),
            'drawer_no_gravity': bool(a.drawer_no_gravity),
            'no_attach': bool(a.no_attach),
            'stop_at_cmd_t': a.stop_at_cmd_t,
            'stop_at_opening_m': a.stop_at_opening_m,
            'gravity_disabled_prims': [k for k, v in grav_state.items() if v is True],
            'rigid_body_count': len(grav_state),
            'note': ('**不計為正式操作成果**。命令序列與完整版逐點相同，'
                     '只是到指定開度提前收尾，非把軌跡壓縮到該行程')},
        'base_fixation': {
            'mode': 'importer_fix_base',
            'root_fixed_joints': _root_fixed,
            'placed_by': 'Xform transform on ' + ROBOT,
            'per_step_base_override': False,
            'caveat': ('底盤由外部固定支撐；不代表輪子靠地面摩擦能承受相同負載，'
                       '移動底盤的操作要另外驗證')},
        'align_ref': ALIGN_REF,
        'monitors_at_start': mon0,
        'monitor_failure': monitor_fail,
        'temp_check_every_samples': TEMP_EVERY,
        'last_temp_c': last_temp,
        'temp_max_c': temp_max, 'temp_max_sim_t': temp_max_t,
        'temp_start_c': tc0,
        'post_stop_log': post_stop,
        'post_stop_cols': ['t', 'opening', 'opening_v'],
        'events': events, 'stop_reason': stop_reason,
        'sim_time_s': float(world.current_time),
        'wall_s': time.monotonic() - w0,
        'target_opening_used_m': TARGET,
        'target_opening_case_m': TARGET_CASE,
        'final_opening_m': final_open,
        'opening_err_m': final_open - TARGET,
        'arrived_sim_t': arrived_t,
        'tcp_final_world': tcp_p.tolist(),
        'cb': {'arm': node.arm_cb_n, 'arm_rejected': node.arm_cb_rej,
               'gripper': node.g_cb_n, 'phase': node.p_cb_n},
        'cpu_temp_c': tc, 'cpu_temp_source': tsrc, 'cpu_limit_c': CPU_LIMIT,
        'log_cols': LOG_COLS,
        'rot_conv_err_max_deg': rot_err_max,
        'stage_vs_fk_err_max_m': stage_fk_err_max,
        'log': log,
    }
    if rec_dir and rec_index:
        import csv as _csv
        with open(os.path.join(rec_dir, 'frames_index.csv'), 'w', newline='') as _f:
            _w = _csv.writer(_f)
            _w.writerow(['frame', 'sim_t'])
            _w.writerows(rec_index)
        out['recording'] = {
            'dir': os.path.abspath(rec_dir), 'frames': len(rec_index),
            'fps_nominal': a.record_fps, 'every_n_physics_steps': rec_every,
            'resolution': a.record_res, 'focal_mm': a.record_focal,
            'sim_t_first': rec_index[0][1], 'sim_t_last': rec_index[-1][1],
            'nature': ('本趟執行當下由模擬器內相機感測器取像，非事後回放、非桌面擷取；'
                       '場景只加光源，未改幾何、材質顏色、質量或物理參數'),
            'caveat': ('RTX 標註器有數幀延遲，畫面相對物理步可能有固定的小延遲；'
                       'frames_index.csv 記的是**取像當下的模擬時間**')}
        print(f'[drawer] 錄影 {len(rec_index)} 幀 -> {rec_dir}', flush=True)
    json.dump(out, open(os.path.join(a.out, 'drawer_run.json'), 'w'),
              ensure_ascii=False)
    print(f'[drawer] 溫度：起 {tc0} °C，本趟峰值 {temp_max} °C @ sim {temp_max_t:.2f} s，'
          f'中止線 {CPU_LIMIT:.0f} °C')
    print(f'[drawer] 結束 {stop_reason}  最終開度 {final_open*1000:.2f} mm '
          f'(誤差 {(final_open-TARGET)*1000:+.2f} mm)  CPU {tc} °C（{tsrc}）')
    print(f'[drawer] -> {os.path.join(a.out, "drawer_run.json")}')
    node.destroy_node(); rclpy.shutdown()
    return 0


rc = 1
try:
    rc = main()
except Exception:
    import traceback
    tb = traceback.format_exc()
    print(tb, flush=True)
    try:
        open(os.path.join(a.out, 'traceback.txt'), 'w').write(tb)
    except Exception:
        pass
    rc = 9
finally:
    sim_app.close()
sys.exit(rc)
