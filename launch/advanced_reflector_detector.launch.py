from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_share_dir = get_package_share_directory('advanced_reflector_detector')
    params_file = os.path.join(package_share_dir, 'config', 'params.yaml')

    node = Node(
        package='advanced_reflector_detector',
        executable='advanced_reflector_detector_node',
        name='advanced_reflector_detector_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        node
    ])