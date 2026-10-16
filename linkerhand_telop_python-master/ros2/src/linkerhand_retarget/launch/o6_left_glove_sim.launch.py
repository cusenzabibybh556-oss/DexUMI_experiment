from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def launch_nodes(context):
    port = LaunchConfiguration("port").perform(context)
    thumb_flex_min = float(
        LaunchConfiguration("thumb_flex_min").perform(context)
    )
    limits_file = LaunchConfiguration("limits_file")

    return [
        Node(
            package="linkerhand_retarget",
            executable="handretarget",
            name="handretarget_node",
            output="screen",
            parameters=[{
                "ports": [port],
                "baudrate": ParameterValue(
                    LaunchConfiguration("baudrate"), value_type=int
                ),
                "auto_scan": ParameterValue(
                    LaunchConfiguration("auto_scan"), value_type=bool
                ),
                "calibration": ParameterValue(
                    LaunchConfiguration("calibration"), value_type=bool
                ),
            }],
        ),
        Node(
            package="linkerhand_retarget",
            executable="o6_left_sim",
            name="o6_left_sim_node",
            output="screen",
            parameters=[
                limits_file,
                {"topic": "/cb_left_hand_control_cmd"},
                ({"command_min": [
                    thumb_flex_min, 0.0, 0.0, 0.0, 0.0, 0.0
                ]} if thumb_flex_min > 0.0 else {}),
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "port",
            default_value=(
                "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
            ),
            description="Left LinkerForce glove serial device",
        ),
        DeclareLaunchArgument(
            "baudrate",
            default_value="2000000",
            description="LinkerForce glove serial baudrate",
        ),
        DeclareLaunchArgument(
            "auto_scan",
            default_value="true",
            description="Probe the candidate port to identify the glove side",
        ),
        DeclareLaunchArgument(
            "calibration",
            default_value="false",
            description="Run the interactive three-pose glove calibration",
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
        OpaqueFunction(function=launch_nodes),
    ])
