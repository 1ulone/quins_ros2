import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    pkg_path = get_package_share_directory('quins')
    urdf_file = os.path.join(pkg_path, 'urdf', 'quadruped.urdf')
    controller_config = os.path.join(pkg_path, 'config', 'controllers.yaml')

    robot_desc = Command([
        'xacro ', urdf_file,
        ' controller_config:=', controller_config
    ])

    os.environ['GZ_SIM_RESOURCE_PATH'] = os.path.join(pkg_path, '..')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # 1. STARTS IMMEDIATELY: Exposes the parameter server before Gazebo needs it.
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher', # CRITICAL: Explicitly named for the plugin to find it
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_desc, value_type=str),
            'use_sim_time': True
        }]
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    # 2. DELAYED 5 SECONDS: Spawns the robot ONLY after the parameter server is alive.
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_desc,
            '-name', 'quadruped',
            '-z', '5.0',
        ],
        output='screen'
    )
    delayed_spawn = TimerAction(period=5.0, actions=[spawn_entity])

    # 3. DELAYED 7 & 9 SECONDS: Load controllers only after the robot safely exists.
    load_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'],
        output='screen',
    )
    delayed_jsb = TimerAction(period=7.0, actions=[load_jsb])

    load_jtc = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager-timeout', '60'],
        output='screen',
    )
    delayed_jtc = TimerAction(period=9.0, actions=[load_jtc])

    return LaunchDescription([
        gz_sim,
        clock_bridge,
        node_robot_state_publisher,
        delayed_spawn,
        delayed_jsb,
        delayed_jtc,
    ])
