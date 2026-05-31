#!/usr/bin/env python3
"""
全向輪（Omni Wheel）十字形佈局 Gazebo 控制器
- 訂閱 /cmd_vel (Twist)
- 計算四輪速度並發布到各輪 Gazebo JointController topic
- 發布 /odom (dead-reckoning)

運動學（四輪十字形，全向輪，輪心距中心 d）：
  左輪  (0, +d)：軸沿 Y，正轉 → 機器人 +X
  右輪  (0, -d)：軸沿 Y，正轉 → 機器人 +X
  前輪 (+d,  0)：軸沿 X，正轉 → 機器人 -Y
  後輪 (-d,  0)：軸沿 X，正轉 → 機器人 -Y

  v_left  =  (vx - wz·d) / r
  v_right =  (vx + wz·d) / r
  v_front = -(vy + wz·d) / r
  v_back  = -(vy - wz·d) / r
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

WHEEL_DIST   = 0.22   # 輪心距底盤中心 (m) — 必須與 URDF 的 wheel_offset 一致
WHEEL_RADIUS = 0.08   # 輪半徑 (m)


class OmniDriveController(Node):
    def __init__(self):
        super().__init__('omni_drive_controller')
        # use_sim_time 由 launch 透過 parameters 傳入，不在此重複 declare

        # 發布各輪速度到 Gazebo JointController
        self.pub_left  = self.create_publisher(Float64, '/gz/left_wheel_vel',  10)
        self.pub_right = self.create_publisher(Float64, '/gz/right_wheel_vel', 10)
        self.pub_front = self.create_publisher(Float64, '/gz/front_wheel_vel', 10)
        self.pub_back  = self.create_publisher(Float64, '/gz/back_wheel_vel',  10)

        # 發布 odometry
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_br    = TransformBroadcaster(self)

        # 訂閱 cmd_vel
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

        # 機器人狀態
        self._vx  = 0.0
        self._vy  = 0.0
        self._wz  = 0.0
        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0
        self._last_t = None   # 等第一個 clock 才初始化

        # 20 Hz odometry timer
        self.create_timer(0.05, self._odom_cb)

        # 訂閱 /clock，收到第一個就發布初始 TF 確保 odom frame 存在
        from rosgraph_msgs.msg import Clock
        self._clock_ready = False
        self.create_subscription(Clock, '/clock', self._clock_cb, 1)

        self.get_logger().info('OmniDriveController started')

    # ------------------------------------------------------------------
    def _clock_cb(self, msg):
        if not self._clock_ready:
            self._clock_ready = True
            self._last_t = self.get_clock().now()
            self._publish_tf_odom()   # 立刻發布一次確保 odom frame 存在
            self.get_logger().info('Clock received, odom TF initialized')

    def _publish_tf_odom(self):
        """發布目前位姿的 TF（odom → base_footprint）"""
        q_z = math.sin(self._yaw / 2.0)
        q_w = math.cos(self._yaw / 2.0)
        stamp = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_footprint'
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.translation.z = 0.0
        t.transform.rotation.z    = q_z
        t.transform.rotation.w    = q_w
        self.tf_br.sendTransform(t)

    # ------------------------------------------------------------------
    def _cmd_cb(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z
        d  = WHEEL_DIST
        r  = WHEEL_RADIUS

        # 十字形全向輪運動學
        v_left  = Float64(data= (vx - wz * d) / r)
        v_right = Float64(data= (vx + wz * d) / r)
        v_front = Float64(data=-(vy + wz * d) / r)
        v_back  = Float64(data=-(vy - wz * d) / r)

        self.pub_left.publish(v_left)
        self.pub_right.publish(v_right)
        self.pub_front.publish(v_front)
        self.pub_back.publish(v_back)

        self._vx = vx
        self._vy = vy
        self._wz = wz

    # ------------------------------------------------------------------
    def _odom_cb(self):
        now = self.get_clock().now()
        if self._last_t is None:
            # clock 還沒來，用 wall time 暫時發布
            self._last_t = now
            self._publish_tf_odom()
            return
        dt = (now - self._last_t).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self._last_t = now

        # Dead-reckoning（世界座標系積分）
        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)
        self._x   += (self._vx * cos_yaw - self._vy * sin_yaw) * dt
        self._y   += (self._vx * sin_yaw + self._vy * cos_yaw) * dt
        self._yaw += self._wz * dt

        q_z = math.sin(self._yaw / 2.0)
        q_w = math.cos(self._yaw / 2.0)
        stamp = now.to_msg()

        # Odometry message
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_footprint'
        odom.pose.pose.position.x    = self._x
        odom.pose.pose.position.y    = self._y
        odom.pose.pose.orientation.z = q_z
        odom.pose.pose.orientation.w = q_w
        odom.twist.twist.linear.x    = self._vx
        odom.twist.twist.linear.y    = self._vy
        odom.twist.twist.angular.z   = self._wz
        self.odom_pub.publish(odom)

        self._publish_tf_odom()


def main():
    rclpy.init()
    node = OmniDriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
