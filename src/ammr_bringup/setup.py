import os
import glob
from setuptools import find_packages, setup

package_name = 'ammr_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),  glob.glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'),    glob.glob('urdf/*')),
        (os.path.join('share', package_name, 'maps'),    glob.glob('maps/*')),
        (os.path.join('share', package_name, 'config'),  glob.glob('config/*')),
        (os.path.join('share', package_name, 'worlds'),  glob.glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'map_publisher = ammr_bringup.map_publisher:main',
            'teleop = ammr_bringup.teleop:main',
        ],
    },
)
