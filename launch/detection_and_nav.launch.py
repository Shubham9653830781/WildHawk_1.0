#!/usr/bin/env python3
"""
detection_and_nav.launch.py

Launch file to start detection, tracking, PX4 bridge and mission coordinator for WildHawk.

Usage examples:
# Launch with default args (assumes models under models/ and camera on /camera/image_raw)
ros2 launch wildhawk_bringup detection_and_nav.launch.py

# Custom model path and camera topic:
ros2 launch wildhawk_bringup detection_and_nav.launch.py \
    model_path:=/home/pi/wildhawk/models/mobilenet_animals.tflite \
    label_path:=/home/pi/wildhawk/models/labels.txt \
    camera_topic:=/camera/image_raw \
    use_camera:=True \
    camera_index:=0

# Use simulated PX4 SITL (if bridging over UDP 14540/14550)
ros2 launch wildhawk_bringup detection_and_nav.launch.py px4_connect_url:=udp://:14540 use_sim_time:=True

"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # Launch arguments
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(os.getcwd(), 'models', 'mobilenet_animals.tflite'),
        description='Path to the TFLite model file'
    )

    label_arg = DeclareLaunchArgument(
        'label_path',
        default_value=os.path.join(os.getcwd(), 'models', 'labels.txt'),
        description='Path to the label map file'
    )

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Image topic to subscribe to'
    )

    use_camera_arg = DeclareLaunchArgument(
        'use_camera',
        default_value='True',
        description='If true, launch local camera node (v4l2) when no external camera topic available'
    )

    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='Local camera index for v4l2 camera node'
    )

    px4_connect_url_arg = DeclareLaunchArgument(
        'px4_connect_url',
        default_value='udp://:14540',
        description='PX4 connection URL used by px4_bridge (e.g. udp://:14540)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation time (True when using SITL/Gazebo)'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='5.0',
        description='Publish rate (Hz) for detection & tracker nodes'
    )

    # LaunchConfigurations
    model_path = LaunchConfiguration('model_path')
    label_path = LaunchConfiguration('label_path')
    camera_topic = LaunchConfiguration('camera_topic')
    use_camera = LaunchConfiguration('use_camera')
    camera_index = LaunchConfiguration('camera_index')
    px4_connect_url = LaunchConfiguration('px4_connect_url')
    use_sim_time = LaunchConfiguration('use_sim_time')
    publish_rate = LaunchConfiguration('publish_rate')

    # Node: detection_node
    detection_node = Node(
        package='wildhawk_nodes',            # UPDATE: replace with your package name
        executable='detection_node.py',      # If packaged as an executable, use the executable name; otherwise use module entry
        name='wildhawk_detection_node',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'labelmap': label_path,
            'score_threshold': 0.5,
            'camera_topic': camera_topic,
            'publish_rate': publish_rate,
            'input_width': 300,
            'input_height': 300,
            'use_camera_if_no_topic': use_camera,
            'camera_index': camera_index,
            'use_sim_time': use_sim_time
        }],
        remappings=[
            # ensure detection publishes to expected topics
            ('/wildhawk/detections', '/wildhawk/detections'),
            ('/wildhawk/detection_image', '/wildhawk/detection_image')
        ]
    )

    # Node: tracker_node
    tracker_node = Node(
        package='wildhawk_nodes',            # UPDATE: replace with your package name
        executable='tracker_node.py',
        name='wildhawk_tracker_node',
        output='screen',
        parameters=[{
            'max_age': 3,
            'min_hits': 1,
            'iou_threshold': 0.3,
            'publish_rate': publish_rate,
            'draw_image': True,
            'detection_topic': '/wildhawk/detections',
            'detection_image_topic': '/wildhawk/detection_image',
            'use_sim_time': use_sim_time
        }],
        remappings=[
            ('/wildhawk/tracks', '/wildhawk/tracks'),
            ('/wildhawk/tracks_image', '/wildhawk/tracks_image')
        ]
    )

    # Node: px4_bridge_node (MAVLink <-> ROS2 bridge)
    # Replace 'px4_bridge' package & executable with the name you implement. This is a placeholder.
    px4_bridge_node = Node(
        package='wildhawk_nodes',            # UPDATE: replace with the package implementing px4_bridge_node
        executable='px4_bridge_node.py',
        name='wildhawk_px4_bridge',
        output='screen',
        parameters=[{
            'connection_url': px4_connect_url,
            'use_sim_time': use_sim_time
        }],
        remappings=[
            # remap MAVLink topics to the conventions used by the rest of the stack if needed
            # ('/mavros/…, '/mavlink/…') etc.
        ]
    )

    # Node: mission_coordinator (consumes /wildhawk/tracks and decides deterrence/waypoints)
    mission_coordinator_node = Node(
        package='wildhawk_nodes',            # UPDATE: replace with your package name
        executable='mission_coordinator.py',
        name='wildhawk_mission_coordinator',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'default_altitude': 10.0,
            'deterrence_duration': 5.0,
            'detection_topic': '/wildhawk/detections',
            'tracks_topic': '/wildhawk/tracks'
        }],
    )

    # Optional camera node: v4l2_camera (if you want to launch a local V4L2 camera)
    # This example uses the commonly available v4l2_camera ROS2 package. Replace with your camera driver if different.
    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera_node',
        output='screen',
        parameters=[{
            'image_size': [640, 480],
            'camera_frame_id': 'camera_link',
            'camera_index': camera_index,
            'use_sim_time': use_sim_time
        }],
        condition=PythonExpression([use_camera, " == 'True'"])
    )

    # Optional rviz2 for visualization (comment out if not needed)
    rviz_config_file = os.path.join(os.getcwd(), 'rviz', 'wildhawk.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else []
    )

    # Info messages to screen
    info_msg = LogInfo(msg=PythonExpression(["'Launching WildHawk stack — model: ' + '", model_path, \" + ', camera: ' + '\", camera_topic]))

    ld = LaunchDescription()

    # Add arguments
    ld.add_action(model_arg)
    ld.add_action(label_arg)
    ld.add_action(camera_topic_arg)
    ld.add_action(use_camera_arg)
    ld.add_action(camera_index_arg)
    ld.add_action(px4_connect_url_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(publish_rate_arg)

    # Add nodes
    ld.add_action(info_msg)
    ld.add_action(detection_node)
    ld.add_action(tracker_node)
    ld.add_action(px4_bridge_node)
    ld.add_action(mission_coordinator_node)
    # Camera node only if requested
    ld.add_action(camera_node)
    # RViz node (optional)
    ld.add_action(rviz_node)

    return ld


if __name__ == '__main__':
    generate_launch_description()
