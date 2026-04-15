#!/usr/bin/env python3
"""
啟動後自動發送預設導航目標 (8.5, 8.5)
用法：ros2 run ammr_bringup send_goal
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import time


class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # 等待 Nav2 完全啟動
        self.timer = self.create_timer(2.0, self.send_goal)
        self.sent = False

    def send_goal(self):
        if self.sent:
            return
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = 8.5
        msg.pose.position.y = 8.5
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub.publish(msg)
        self.get_logger().info('已發送目標點: (8.5, 8.5)')
        self.sent = True


def main():
    rclpy.init()
    node = GoalSender()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
