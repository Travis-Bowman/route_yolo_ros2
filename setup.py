from setuptools import setup
from glob import glob

package_name = 'route_yolo_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/models', glob('route_yolo_ros2/models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='YOLOv8 detection node with ROS 2 service interface.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect = route_yolo_ros2.detect:main',
            'detect_service_client = route_yolo_ros2.detect_service_client:main',
        ],
    },
)

