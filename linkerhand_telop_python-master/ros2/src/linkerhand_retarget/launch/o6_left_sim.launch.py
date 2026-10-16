from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def launch_node(context):
    thumb_flex_min = float(
        LaunchConfiguration("thumb_flex_min").perform(context)
    )
    limits_file = LaunchConfiguration("limits_file")
    return [Node(
        package="linkerhand_retarget",
        executable="o6_left_sim",
        name="o6_left_sim_node",
        output="screen",
        parameters=[
            limits_file,
            {"topic": LaunchConfiguration("topic")},
            ({"command_min": [
                thumb_flex_min, 0.0, 0.0, 0.0, 0.0, 0.0
            ]} if thumb_flex_min > 0.0 else {}),
        ],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "topic",
            default_value="/cb_left_hand_control_cmd",
            description="O6 left-hand JointState command topic",
        ),
        DeclareLaunchArgument(
            "limits_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("linkerhand_retarget"),
                "config",
                "o6_command_limits.yaml",
            ]),
            description="ROS parameter file containing simulation limits",
        ),
        DeclareLaunchArgument(
            "thumb_flex_min",
            default_value="0",
            description=(
                "Minimum channel-1 command; higher values reduce maximum "
                "thumb flexion"
            ),
        ),
        OpaqueFunction(function=launch_node),
    ])
