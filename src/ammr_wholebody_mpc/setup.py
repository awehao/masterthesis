import glob
import os

from setuptools import find_packages, setup

package_name = 'ammr_wholebody_mpc'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test', 'offline_prototype', 'offline_prototype.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'gmpc_node           = ammr_wholebody_mpc.gmpc_node:main',
            'test_path_publisher = ammr_wholebody_mpc.test_path_publisher:main',
            'goal_to_plan_relay  = ammr_wholebody_mpc.goal_to_plan_relay:main',
            'obstacle_aggregator = ammr_wholebody_mpc.obstacle_aggregator:main',
            'scan_obstacle_tracker = ammr_wholebody_mpc.scan_obstacle_tracker:main',
            'scan_safety_shield  = ammr_wholebody_mpc.scan_safety_shield:main',
        ],
    },
)
