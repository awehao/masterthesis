#!/usr/bin/env python3
"""遠距乾跑的運動學世界節點（取代 Isaac；Codex 20261116 核可的「隔離真節點乾跑」）。**合成資料來源照實**：
  /clock      ：本節點的模擬時鐘（牆鐘計時器驅動，每步 0.01 s）。
  /scan_raw   ：2D 射線投射——住宅場景登錄檔（20261112_m8_house_scene.json）的外牆／內牆／家具（高度涵蓋雷射高度者）＋關閉的門扇（門資產幾何）；360 束、10 Hz、無雜訊。未知物與行人不在場。
  /odom_raw   ：由 /cmd_vel 積分的底盤位姿（世界座標＝odom 座標；理想里程，無滑移、無延遲模型）。
  /joint_states：手臂六關節（一階設定點模型，α 取辨識檔；設定點由 /far/arm_cmd 的關節速率命令積分，命令逾時 0.2 s 歸零）＋手指。
真值另發 /far/truth（只供紀錄）。結束時把時間序列存成 npz。不發任何導航命令。"""
import json
import math
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(HERE)
DT = 0.01
D_STEPS = 8
SCAN_N, SCAN_MIN, SCAN_MAX = 360, -3.14159, 3.14159
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
RANGE_MIN, RANGE_MAX = 0.12, 10.0
ARM = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def load_boxes(lidar_z):
    reg = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
    B = []
    for it in reg['walls'] + reg['furniture']:
        c, s = it['c'], it['s']
        if c[2] - s[2] / 2 <= lidar_z <= c[2] + s[2] / 2:
            B.append((it['name'], c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2))
    import yaml
    d = yaml.safe_load(open(os.path.join(WS, reg['door']['asset'])))
    return B, reg, d


def ray_cast(p, yaw, boxes):
    th = yaw + SCAN_MIN + np.arange(SCAN_N) * SCAN_INC
    dx, dy = np.cos(th), np.sin(th)
    best = np.full(SCAN_N, np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for _, x0, x1, y0, y1 in boxes:
            tx0, tx1 = (x0 - p[0]) / dx, (x1 - p[0]) / dx
            ty0, ty1 = (y0 - p[1]) / dy, (y1 - p[1]) / dy
            tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
            tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
            hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
            t = np.where(tmin > 0, tmin, np.inf)
            best = np.where(hit, np.minimum(best, t), best)
    r = np.where(np.isfinite(best) & (best >= RANGE_MIN) & (best <= RANGE_MAX), best, np.inf)
    return r


class World(Node):
    def __init__(self):
        super().__init__('far_dry_world')
        a = sys.argv
        self.out = a[1]
        self.q = np.array([float(x) for x in a[2].split(',')], float)          # 起點 9 維（底盤 3＋手臂 6）
        self.t_end = float(a[3])
        gm = json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))   # 事先算好的門扇與雷射掛載（本節點不匯入專案控制模組：它們會裝 rclpy 替身）
        self.lidar_xy, self.lidar_z = np.array(gm['lidar_xy'], float), float(gm['lidar_z'])
        self.boxes, reg, door = load_boxes(self.lidar_z)
        if gm['leaf_z'][0] <= self.lidar_z <= gm['leaf_z'][1]:
            self.boxes.append(tuple(gm['leaf']))
        ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
        self.al = np.asarray(ident['alpha'], float)
        self.s = self.q[3:].copy()
        self.t, self.n = 0.0, 0
        self.cmd, self.cmd_t = np.zeros(3), -1.0
        self.acmd, self.acmd_t = np.zeros(6), -1.0
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_truth = self.create_publisher(Float64MultiArray, '/far/truth', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.create_subscription(Float64MultiArray, '/far/arm_cmd', self._acmd, 10)
        # 取像模式的接線驗證（argv[4]=='obs'）：以歷史影像的估計合成 /far/handle_obs（5 Hz，全部標 valid；只驗接線，不是移動視覺）
        self.fake_obs = len(a) > 4 and a[4] == 'obs'
        if self.fake_obs:
            from std_msgs.msg import Bool, String
            self.T_est = np.array(json.load(open(os.path.join(HERE, 'far_dry', 'handle_est_far_live.json')))['T_WH_est'], float)
            self.p_obs = self.create_publisher(Float64MultiArray, '/far/handle_obs', 10)
            self.p_xs = self.create_publisher(String, '/far/exec_state', 10)
            self._String = String
            self.done, self.obs_seq = False, 0
            self.create_subscription(Bool, '/far/done', lambda m: setattr(self, 'done', self.done or bool(m.data)), 10)
            self.create_subscription(String, '/far/abort', lambda m: setattr(self, 'done', True), 10)   # 模擬工程中止：不再步進
        self.log = []
        self.pipe = [np.zeros(9) for _ in range(D_STEPS)]              # 命令到受控對象的 8 步在途管線（同門任務控制器的延遲模型；底盤與手臂都經過）
        self.v_world, self.wz = np.zeros(2), 0.0
        self.meta = dict(boxes=[b[0] for b in self.boxes], lidar_z=self.lidar_z, lidar_xy=self.lidar_xy.tolist(), sources=dict(scan='合成射線投射（已知場景幾何＋關閉門扇；無雜訊）', odom='/cmd_vel 理想積分', clock='本節點', joint_states='一階設定點模型', delay='命令經 8 個物理步（0.08 s）在途管線才作用（底盤與手臂）'))
        self.timer = self.create_timer(DT, self.step)

    def _cmd(self, m):
        self.cmd, self.cmd_t = np.array([m.linear.x, m.linear.y, m.angular.z]), self.t

    def _acmd(self, m):
        if len(m.data) == 6 and np.isfinite(m.data).all():
            self.acmd, self.acmd_t = np.array(m.data, float), self.t

    def stamp(self):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(self.t)
        s.nanosec = int(round((self.t - int(self.t)) * 1e9))
        return s

    def step(self):
        u_in = self.cmd if self.t - self.cmd_t <= 0.5 else np.zeros(3)
        ua_in = self.acmd if self.t - self.acmd_t <= 0.2 else np.zeros(6)
        self.pipe.append(np.r_[u_in, ua_in])
        ue = self.pipe.pop(0)
        u, ua = ue[:3], ue[3:]
        c, s_ = math.cos(self.q[2]), math.sin(self.q[2])
        self.v_world, self.wz = np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]]), float(u[2])
        self.q[:2] += DT * self.v_world
        self.q[2] += DT * self.wz
        self.s = self.s + DT * ua
        self.q[3:] = self.q[3:] + self.al * (self.s - self.q[3:])
        self.t = round(self.t + DT, 10)
        self.n += 1
        ck = Clock()
        ck.clock = self.stamp()
        self.p_clock.publish(ck)
        if self.n % 2 == 0:
            m = Odometry()
            m.header.stamp = self.stamp()
            m.header.frame_id, m.child_frame_id = 'odom', 'base_footprint'
            m.pose.pose.position.x, m.pose.pose.position.y = float(self.q[0]), float(self.q[1])
            m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(self.q[2] / 2), math.sin(self.q[2] / 2)
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = float(u[0]), float(u[1]), float(u[2])
            self.p_odom.publish(m)
            js = JointState()
            js.header.stamp = self.stamp()
            js.name = ARM + ['finger_joint1', 'finger_joint2']
            js.position = [float(x) for x in self.q[3:]] + [0.0089, 0.0089]
            self.p_js.publish(js)
            s_end = self.s + DT * sum(p_[3:] for p_ in self.pipe)
            self.p_truth.publish(Float64MultiArray(data=[self.t, *[float(x) for x in self.q], *[float(x) for x in self.s], *[float(x) for x in s_end], *[float(x) for x in np.concatenate(self.pipe)]]))   # t, q9, s6, s_end6, 在途命令 8×9
        if self.n % 10 == 0:
            cy, sy = math.cos(self.q[2]), math.sin(self.q[2])
            pl = self.q[:2] + np.array([cy * self.lidar_xy[0] - sy * self.lidar_xy[1], sy * self.lidar_xy[0] + cy * self.lidar_xy[1]])
            r = ray_cast(pl, self.q[2], self.boxes)
            m = LaserScan()
            m.header.stamp = self.stamp()
            m.header.frame_id = 'lidar_link'
            m.angle_min, m.angle_max, m.angle_increment = SCAN_MIN, SCAN_MAX, SCAN_INC
            m.range_min, m.range_max = RANGE_MIN, RANGE_MAX
            m.ranges = [float(x) for x in r]
            self.p_scan.publish(m)
            self.scan_min = float(np.min(r))
        if getattr(self, 'fake_obs', False):
            if self.n % 20 == 0:
                c, s_ = math.cos(self.q[2]), math.sin(self.q[2])
                Tb = np.eye(4); Tb[:2, :2] = [[c, -s_], [s_, c]]; Tb[:2, 3] = self.q[:2]
                self.obs_seq += 1
                ok_ = float(np.linalg.norm(self.q[:2] - self.T_est[:2, 3])) < 1.6      # 合成假設：1.6 m 內才有有效估計（只為了讓「途中才有第一格有效觀察」的接線被走到；不是偵測距離的量測）
                Tp = self.T_est.copy()
                rng_ = float(np.linalg.norm(self.q[:2] - self.T_est[:2, 3]))
                bias = os.environ.get('OBS_FAR_BIAS_MM')                # 合成測試：遠距（>0.8 m）的觀察加固定偏差、近距不加 ⇒ 固定參考帶偏差，近距更新應把它拉回（只驗更新入口的閉迴路接線）
                if bias and rng_ > 0.8:
                    Tp[:2, 3] += np.array([float(x) for x in bias.split(',')]) * 1e-3
                self.p_obs.publish(Float64MultiArray(data=[float(self.obs_seq), self.t, 1.0 if ok_ else 0.0, *[float(x) for x in self.q[:3]], *([float(x) for x in (np.linalg.inv(Tb) @ Tp).reshape(-1)] if ok_ else [float('nan')] * 16), self.t]))
                self.p_xs.publish(self._String(data='run'))
            if self.done:
                self.save()
                raise SystemExit(0)
        self.log.append(np.r_[self.t, self.q, self.s, u, ua, getattr(self, 'scan_min', np.nan), u_in, ua_in])
        if self.t >= self.t_end:
            self.save()
            raise SystemExit(0)

    def save(self):
        np.savez_compressed(self.out + '_world.npz', L=np.array(self.log), cols=np.array(['t'] + [f'q{i}' for i in range(9)] + [f's{i}' for i in range(6)] + ['ux', 'uy', 'uw'] + [f'ua{i}' for i in range(6)] + ['scan_min', 'uin_x', 'uin_y', 'uin_w'] + [f'uain{i}' for i in range(6)]))
        json.dump(self.meta, open(self.out + '_world_meta.json', 'w'), ensure_ascii=False, indent=1)


def main():
    rclpy.init()
    w = World()
    try:
        rclpy.spin(w)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        try:
            w.save()
        except Exception:                                             # noqa: BLE001
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
