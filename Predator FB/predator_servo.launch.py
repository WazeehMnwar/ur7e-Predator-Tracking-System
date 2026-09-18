"""MoveIt Servo for camera pointing on a UR7e (ROS 2 Humble).

Start the UR driver with forward_velocity_controller before this file. This
launch starts the stock MoveIt node and a velocity-output Servo node.
"""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

from ament_index_python.packages import (
    get_package_prefix, get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro
import yaml


def make_servo(context):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    if ur_type != "ur7e":
        raise ValueError("This camera Servo setup is configured for ur7e only")

    description = Path(get_package_share_directory("ur_description"))
    moveit = Path(get_package_share_directory("ur_moveit_config"))
    model = description / "config" / ur_type
    mappings = {
        "name": "ur",
        "ur_type": ur_type,
        "robot_ip": "xxx.yyy.zzz.www",  # Descriptive URDF only; the driver owns the IP.
        "joint_limit_params": str(model / "joint_limits.yaml"),
        "kinematics_params": str(model / "default_kinematics.yaml"),
        "physical_params": str(model / "physical_parameters.yaml"),
        "visual_params": str(model / "visual_parameters.yaml"),
        "safety_limits": "true",
        "safety_pos_margin": "0.15",
        "safety_k_position": "20",
        "script_filename": "ros_control.urscript",
        "input_recipe_filename": "rtde_input_recipe.txt",
        "output_recipe_filename": "rtde_output_recipe.txt",
        "prefix": "",
    }
    urdf = xacro.process_file(
        str(description / "urdf" / "ur.urdf.xacro"), mappings=mappings
    ).toxml()
    # A small Cartesian input can otherwise turn into a large joint velocity.
    # Limit Servo's actual robot model, including its filtered joint output.
    poses_path = Path(LaunchConfiguration("poses_file").perform(context))
    with poses_path.open(encoding="utf-8") as stream:
        tracking = yaml.safe_load(stream).get("tracking", {})
    max_speed = float(tracking.get("max_joint_speed", 0.10))
    max_acceleration = float(tracking.get("max_joint_acceleration", 0.20))
    if (not math.isfinite(max_speed) or not 0.01 <= max_speed <= 0.20
            or not math.isfinite(max_acceleration) or not 0.01 <= max_acceleration <= 0.5):
        raise ValueError("Invalid tracking joint speed/acceleration limits in pose YAML")
    robot = ET.fromstring(urdf)
    joint_limits = {}
    for joint in robot.findall("joint"):
        if joint.get("type") in ("revolute", "continuous"):
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Missing joint limit: {joint.get('name')}")
            capped = min(float(limit.get("velocity")), max_speed)
            limit.set("velocity", str(capped))
            joint_limits[joint.get("name")] = {
                "has_velocity_limits": True, "max_velocity": capped,
                "has_acceleration_limits": True, "max_acceleration": max_acceleration,
            }
    urdf = ET.tostring(robot, encoding="unicode")
    srdf = xacro.process_file(
        str(moveit / "srdf" / "ur.srdf.xacro"),
        mappings={"name": "ur", "prefix": ""},
    ).toxml()

    with (moveit / "config" / "ur_servo.yaml").open(encoding="utf-8") as stream:
        servo = yaml.safe_load(stream)
    if not isinstance(servo, dict):
        raise ValueError("ur_moveit_config/config/ur_servo.yaml is invalid")
    servo.update({
        "move_group_name": "ur_manipulator",
        "planning_frame": "base_link",
        "ee_frame_name": "tool0",  # Humble parameter name.
        "ee_frame": "tool0",
        "robot_link_command_frame": "tool0",
        "command_in_type": "speed_units",
        "cartesian_command_in_topic": "/servo_node/delta_twist_cmds",
        "joint_command_in_topic": "/servo_node/delta_joint_cmds",
        "command_out_topic": "/forward_velocity_controller/commands",
        "command_out_type": "std_msgs/Float64MultiArray",
        "publish_joint_positions": False,
        "publish_joint_velocities": True,
        "publish_joint_accelerations": False,
        "publish_period": 0.02,
        "incoming_command_timeout": 0.2,
        "num_outgoing_halt_msgs_to_publish": 20,
        "joint_topic": "/joint_states",
        "check_collisions": True,
        "collision_check_rate": 10.0,
        "override_velocity_scaling_factor": 0.0,
        "halt_all_joints_in_joint_mode": True,
    })
    servo_bin = Path(get_package_prefix("moveit_servo")) / "lib" / "moveit_servo"
    if (servo_bin / "servo_node_main").exists():
        executable = "servo_node_main"  # Humble
    elif (servo_bin / "servo_node").exists():
        executable = "servo_node"  # Newer MoveIt releases
    else:
        raise FileNotFoundError(f"MoveIt Servo executable missing in {servo_bin}")
    return [Node(
        package="moveit_servo",
        executable=executable,
        name="servo_node",
        output="screen",
        parameters=[
            {"robot_description": urdf},
            {"robot_description_semantic": srdf},
            {"robot_description_planning": {"joint_limits": joint_limits}},
            {"moveit_servo": servo},
        ],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur7e"),
        DeclareLaunchArgument("poses_file", default_value=str(
            Path(__file__).with_name("predator_servo_poses.yaml"))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("ur_moveit_config"), "launch", "ur_moveit.launch.py",
            ])),
            launch_arguments={
                "ur_type": LaunchConfiguration("ur_type"),
                "launch_rviz": "false",
                "launch_servo": "false",
            }.items(),
        ),
        OpaqueFunction(function=make_servo),
    ])
