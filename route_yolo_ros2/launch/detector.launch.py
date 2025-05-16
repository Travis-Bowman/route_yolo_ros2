from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='route_yolo_ros2',
            executable='detect',
            name='yolo_v8_detector',
            output='screen',
            parameters=[
                {'enable': True},
                {'flip_image': False},
                {'image_resize': 640},
            ],
            remappings=[
                ('/camera/image_raw', '/your/camera/image_raw'),
                ('/detect/trigger', '/your/yolo_trigger_command'),
            ]
        )
    ])
