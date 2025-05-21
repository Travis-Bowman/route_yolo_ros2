from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

image_topic_arg = DeclareLaunchArgument("image_topic", default_value="/routecam")
coco_model_arg = DeclareLaunchArgument("coco_model_path", default_value="~/ros2_ws/src/route_yolo_ros2/models/yolo12m.pt")
tire_model_arg = DeclareLaunchArgument("tire_model_path", default_value="~/ros2_ws/src/route_yolo_ros2/models/best-05142025.pt")

def detect():
    return Node(
        package='route_yolo_ros2',
        executable='detect',
        name='yolo_v8_detector',
        output='screen',
        parameters=[
            {'image_topic': LaunchConfiguration("image_topic")},
            {'coco_model_path': LaunchConfiguration("coco_model_path")},
            {'tire_model_path': LaunchConfiguration("tire_model_path")},
            {'flip_image': False},
            {'image_resize': 640},
        ],
    )

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
