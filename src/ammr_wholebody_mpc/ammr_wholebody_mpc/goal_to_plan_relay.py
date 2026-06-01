"""Bridge node: convert /goal_pose to a Nav2 ComputePathToPose request.

Nav2's `planner_server` exposes the planner as a ROS2 action
(`/compute_path_to_pose`, action type `nav2_msgs/action/ComputePathToPose`)
but does NOT automatically react to RViz's `/goal_pose` topic — in a full
stack that bridging happens inside `bt_navigator`.

We're running without `bt_navigator` (we replace `controller_server` with
our own GMPC node), so we need this tiny relay to glue the two ends:

   /goal_pose  ──►  goal_to_plan_relay  ──action──►  planner_server
                                                              │
                                                              ▼
                                                       /plan (Path)
                                                              │
                                                              ▼
                                                       gmpc_controller ──► /cmd_vel

The relay republishes the path itself as a belt-and-braces — that way the
GMPC node sees the latest plan regardless of whether `planner_server` chose
to broadcast it on its own publisher.
"""

from __future__ import annotations

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg      import Path

import tf2_ros

try:
    from nav2_msgs.action import ComputePathToPose
except ImportError as e:                                  # pragma: no cover
    raise RuntimeError(
        'nav2_msgs not available — install ros-jazzy-nav2-msgs'
    ) from e


class GoalToPlanRelay(Node):

    def __init__(self):
        super().__init__('goal_to_plan_relay')

        self.declare_parameter('global_frame',     'map')
        self.declare_parameter('robot_base_frame', 'base_footprint')
        self.declare_parameter('planner_id',       'GridBased')
        self.declare_parameter('action_name',      'compute_path_to_pose')
        self.declare_parameter('plan_topic',       '/plan')
        self.declare_parameter('goal_topic',       '/goal_pose')
        self.declare_parameter('server_timeout',   5.0)

        self.global_frame = str(self.get_parameter('global_frame').value)
        self.base_frame   = str(self.get_parameter('robot_base_frame').value)
        self.planner_id   = str(self.get_parameter('planner_id').value)
        self.srv_timeout  = float(self.get_parameter('server_timeout').value)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.action_client = ActionClient(
            self, ComputePathToPose,
            str(self.get_parameter('action_name').value),
        )

        # Re-publish path with transient-local so a slow gmpc_node still gets it
        plan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history    =QoSHistoryPolicy.KEEP_LAST,
            depth      =1,
            durability =QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.plan_pub = self.create_publisher(
            Path, str(self.get_parameter('plan_topic').value), plan_qos,
        )

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter('goal_topic').value),
            self._goal_cb, 10,
        )

        self.get_logger().info(
            f'goal_to_plan_relay up: planner_id={self.planner_id!r}, '
            f'frame {self.global_frame}->{self.base_frame}'
        )

    # ----------------------------------------------------------------------
    def _current_pose_as_stamped(self) -> PoseStamped | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame,
                rclpy.time.Time(), Duration(seconds=0.5),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF {self.global_frame}->{self.base_frame}: {e}')
            return None

        ps = PoseStamped()
        ps.header.frame_id      = self.global_frame
        ps.header.stamp         = self.get_clock().now().to_msg()
        ps.pose.position.x      = tf.transform.translation.x
        ps.pose.position.y      = tf.transform.translation.y
        ps.pose.position.z      = tf.transform.translation.z
        ps.pose.orientation     = tf.transform.rotation
        return ps

    def _goal_cb(self, msg: PoseStamped):
        if not self.action_client.wait_for_server(timeout_sec=self.srv_timeout):
            self.get_logger().error(
                'ComputePathToPose action server unavailable — '
                'is planner_server running and lifecycle-active?'
            )
            return

        start = self._current_pose_as_stamped()
        if start is None:
            return

        req = ComputePathToPose.Goal()
        req.start      = start
        req.goal       = msg
        req.use_start  = True
        req.planner_id = self.planner_id

        self.get_logger().info(
            f'Plan request: ({start.pose.position.x:.2f}, {start.pose.position.y:.2f}) '
            f'-> ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

        send_future = self.action_client.send_goal_async(req)
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn('Planner rejected goal')
            return
        result_future = gh.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        path = result.path
        if path is None or len(path.poses) == 0:
            self.get_logger().warn(
                f'Planner returned empty path (error_code={result.error_code}, '
                f'msg={result.error_msg!r})'
            )
            return
        self.plan_pub.publish(path)
        self.get_logger().info(f'Path published: {len(path.poses)} poses')


def main():
    rclpy.init()
    node = GoalToPlanRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
