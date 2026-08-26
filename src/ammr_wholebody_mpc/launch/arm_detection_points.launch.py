"""Arm-avoidance observation pipeline: 12 detection points + closest-obstacle
vectors.

Replaces two ROS 1 launch files from the MastersThesis_ros tree:

  detection_points_tf.launch        12 static_transform_publisher nodes
  collision_detection_points.launch the node, plus a hand-written obstacle YAML

Neither is needed here. The detection frames now come out of the robot
description (arm_detect_points.xacro, included by both omni_bot.urdf.xacro and
omni_bot_wholebody.urdf.xacro), so robot_state_publisher puts them on TF and
they move with the arm. Nothing static to publish.

The obstacle list is NOT hand-written here. The ROS 1 version shipped a
213-line indoor_small_obstacles.yaml plus a utils_scripts/extract_sdf_obstacles.py
to regenerate it whenever the world changed -- two artefacts that silently go
stale. We parse the world SDF at launch instead, so the obstacle set is the
world by construction.

Only dyn_obs_* are included. Walls are static and already handled by the
chassis-level CBF and the raw-scan shield; feeding them to the arm as well
would double-count. Add them here if the arm ever needs to reach past a wall.

    ros2 launch ammr_wholebody_mpc arm_detection_points.launch.py \
        world:=<abs path to .sdf>

The detection-point offsets are fixed by the policy that consumes them: point
order and per-link offsets have to match the training rig. They live in
arm_detect_points.xacro; the list below is only the ORDER the node reads them
in, which has to match that file.
"""
import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Order is the observation-vector order and must match arm_detect_points.xacro.
DETECTION_FRAMES = [
    'detect0_1', 'detect0_2', 'detect1',
    'detect2_1', 'detect2_2', 'detect2_3',
    'detect3_1', 'detect3_2',
    'detect4_1', 'detect4_2',
    'detect5', 'detect6',
]


def _obstacles_from_world(path: str) -> list[str]:
    """dyn_obs_* -> the node's flat 'name:model:kind:dims' spec strings.

    The model pose published on /model/<name>/pose is already the collision
    centre in these worlds (the SDF puts the box centre at z = height/2 and
    gives the link no further offset), so the collision offset stays zero.
    """
    if not path or not os.path.exists(path):
        return []
    sdf = open(path).read()
    out = []
    for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', sdf, re.S):
        name, body = m.group(1), m.group(2)
        box = re.search(r'<box><size>([^<]+)</size>', body)
        cyl = re.search(r'<cylinder><radius>([\d.]+)</radius>\s*<length>([\d.]+)',
                        body)
        if box:
            sx, sy, sz = box.group(1).split()
            out.append(f'{name}:{name}:box:{sx},{sy},{sz}')
        elif cyl:
            out.append(f'{name}:{name}:cylinder:{cyl.group(1)},{cyl.group(2)}')
    return out


def _setup(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)
    obstacles = _obstacles_from_world(world)
    if not obstacles:
        print(f'[arm_detection_points] WARNING: no dyn_obs_* found in {world!r} '
              f'-- all 12 points will publish NaN')

    return [Node(
        package='ammr_wholebody_mpc', executable='arm_detection_points',
        name='arm_detection_points', output='screen',
        parameters=[{
            'robot_base_frame': LaunchConfiguration('base_frame'),
            'world_frame': LaunchConfiguration('world_frame'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'use_gz_base_pose': LaunchConfiguration('use_gz_base_pose'),
            'gz_base_model': LaunchConfiguration('gz_base_model'),
            'detection_frames': DETECTION_FRAMES,
            'obstacles': obstacles,
        }])]


def generate_launch_description() -> LaunchDescription:
    default_world = os.path.join(
        get_package_share_directory('ammr_bringup'), 'worlds', 'bigarena.sdf')
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument('world_frame', default_value='map'),
        DeclareLaunchArgument('publish_rate', default_value='30.0'),
        # true reproduces the ROS 1 rig, which read the base pose from
        # /gazebo/link_states, i.e. ground truth. Off by default: anything we
        # report has to run on the estimated pose.
        DeclareLaunchArgument('use_gz_base_pose', default_value='false'),
        DeclareLaunchArgument('gz_base_model', default_value='omni_bot'),
        OpaqueFunction(function=_setup),
    ])
