from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare('quins').find('quins')
    
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'gazebo.launch.py')
        )
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='odometry_bridge',
        arguments=[
            '/model/quadruped/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        ],
        output='screen'
    )

    control_node = Node(
        package='quins',
        executable='control',
        name='quins_tuner',
        output='screen'
    )

    gui_node = Node(
        package='quins',
        executable='gui',
        name='quins_tuner_gui',
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        bridge_node,
        control_node, 
        gui_node
    ])
