#!/usr/bin/env python3
"""
px4_bridge_node.py

ROS2 node that bridges PX4 (MAVLink) <-> ROS2 for basic telemetry and command exchange.

Features:
- Connects to PX4 via pymavlink.mavutil (supports udp, tcp, serial URLs).
- Listens for MAVLink telemetry (HEARTBEAT, GLOBAL_POSITION_INT, ATTITUDE, SYS_STATUS, GPS_RAW_INT, HIGHRES_IMU).
- Publishes telemetry JSON on /px4/telemetry (std_msgs/String) and more-specific endpoints:
    - /px4/attitude
    - /px4/global_position
    - /px4/sys_status
- Subscribes to /wildhawk/px4_command (std_msgs/String) to receive JSON commands:
    Examples:
      {"cmd": "arm"}
      {"cmd": "disarm"}
      {"cmd": "set_mode", "mode": "GUIDED"}         # supported modes: STABILIZED, MANUAL, ALTCTL, GUIDED, AUTO, RTL
      {"cmd": "takeoff", "altitude": 10}
      {"cmd": "goto", "lat": 28.6, "lon": 77.2, "alt": 10}
      {"cmd": "rtl"}

- Sends corresponding MAVLink messages (using command_long_send or set_mode_send).
- Runs the MAVLink receive loop in a background thread, and publishes ROS messages on message receipt and periodic heartbeat.

Requirements:
- pymavlink: pip install pymavlink
- rclpy (ROS 2)
- std_msgs, sensor_msgs etc. (we use std_msgs/String for simplicity)

Usage:
    ros2 run <your_package> px4_bridge_node.py --ros-args -p connection_url:="udp:127.0.0.1:14540" -p target_system:=1

Or run directly:
    python3 px4_bridge_node.py --connection_url udp:127.0.0.1:14540

Author: Shaneshraje Kadu (WildHawk 1.0)
"""

import argparse
import json
import threading
import time
import math
import sys
import traceback

from typing import Optional, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# pymavlink
try:
    from pymavlink import mavutil
except Exception as e:
    raise RuntimeError("pymavlink is required. Install with: pip install pymavlink") from e


# ----------------- Helper functions -----------------


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


# Map friendly mode names to PX4 custom_mode values (simple mapping)
# NOTE: PX4 uses custom_mode integer encoding; mapping below is illustrative and may need adjustments per PX4 version.
PX4_MODE_MAP = {
    "MANUAL": 0,
    "ALTCTL": 2,
    "POSCTL": 3,
    "OFFBOARD": 6,
    "AUTO_MISSION": 10,
    "RTL": 5,
    "GUIDED": 4,  # GUIDED is not exact PX4 term but used for convenience here
    "STABILIZED": 1,
    "AUTO": 10,
    "HOLD": 7
}


# ----------------- PX4BridgeNode -----------------


class PX4BridgeNode(Node):
    def __init__(self, connection_url: str = 'udp:127.0.0.1:14540', target_system: int = 1, target_component: int = 1,
                 heartbeat_period: float = 1.0, heartbeat_timeout: float = 5.0):
        super().__init__('px4_bridge_node')
        self.get_logger().info('Starting PX4 Bridge Node...')
        self.connection_url = connection_url
        self.target_system = int(target_system)
        self.target_component = int(target_component)
        self.heartbeat_period = float(heartbeat_period)
        self.heartbeat_timeout = float(heartbeat_timeout)

        # Publishers
        self.telemetry_pub = self.create_publisher(String, '/px4/telemetry', 10)
        self.attitude_pub = self.create_publisher(String, '/px4/attitude', 10)
        self.position_pub = self.create_publisher(String, '/px4/global_position', 10)
        self.status_pub = self.create_publisher(String, '/px4/sys_status', 5)

        # Command subscriber
        self.cmd_sub = self.create_subscription(String, '/wildhawk/px4_command', self.cmd_callback, 10)

        # Internal state
        self.mav = None  # mavutil connection
        self.mav_thread = None
        self.mav_thread_stop = threading.Event()
        self.last_heartbeat_time = 0.0

        # Connect
        self.connect_to_mavlink()

        # Start background processing thread
        self.mav_thread = threading.Thread(target=self.mavloop, daemon=True)
        self.mav_thread.start()

        # Periodic heartbeat publisher to indicate liveliness
        self.create_timer(self.heartbeat_period, self.publish_heartbeat_status)

    def connect_to_mavlink(self):
        """Establish mavlink connection using pymavlink.mavutil"""
        self.get_logger().info(f'Connecting to MAVLink at {self.connection_url} ...')
        try:
            # mavutil.mavlink_connection supports: udp:HOST:PORT, tcp:HOST:PORT, serial:/dev/ttyUSB0:57600
            self.mav = mavutil.mavlink_connection(self.connection_url, autoreconnect=True, baud=115200)
            # wait heartbeat (non-blocking with timeout)
            self.get_logger().info('Waiting for heartbeat from system...')
            heartbeat = self.mav.wait_heartbeat(timeout=5)
            if heartbeat is None:
                self.get_logger().warning('No heartbeat received within timeout; continuing and will attempt reconnects.')
            else:
                self.last_heartbeat_time = time.time()
                self.get_logger().info(f'Heartbeat received from system (type={heartbeat.type}, sysid={self.mav.target_system})')
            # Set target_system/component from connection if not provided
            try:
                if self.mav.target_system is not None:
                    self.target_system = int(self.mav.target_system)
            except Exception:
                pass
        except Exception as e:
            self.get_logger().error(f'Failed to connect to MAVLink: {e}\n{traceback.format_exc()}')
            self.mav = None

    # ---------- MAVLink send helpers ----------

    def send_command_long(self, command: int, params: Optional[list] = None):
        """
        Send a MAV_CMD (command_long) with up to 7 params.
        params: list of up to 7 float params
        """
        if params is None:
            params = [0.0] * 7
        params = list(params) + [0.0] * (7 - len(params))
        try:
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                int(command),
                0,  # confirmation
                float(params[0]),
                float(params[1]),
                float(params[2]),
                float(params[3]),
                float(params[4]),
                float(params[5]),
                float(params[6])
            )
            self.get_logger().info(f'Sent COMMAND_LONG {command} params={params}')
        except Exception as e:
            self.get_logger().error(f'Failed to send command_long {command}: {e}')

    def arm_disarm(self, arm: bool):
        """Arm or disarm the vehicle using MAV_CMD_COMPONENT_ARM_DISARM (400)"""
        try:
            param = 1 if arm else 0
            self.send_command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, [param])
        except Exception as e:
            self.get_logger().error(f'arm_disarm error: {e}')

    def set_mode_px4(self, mode: str):
        """
        Attempt to set PX4 mode using MAV_CMD_DO_SET_MODE or set_mode_send.
        Note: PX4's mode API can vary; this is a best-effort approach.
        """
        mode_upper = mode.upper()
        if mode_upper not in PX4_MODE_MAP:
            self.get_logger().error(f'Mode {mode} not in PX4_MODE_MAP; supported: {list(PX4_MODE_MAP.keys())}')
            return

        try:
            # Try set_mode_send (some pymavlink builds support it)
            base_mode = 0  # PX4 base_mode usage simplistic here
            custom_mode = int(PX4_MODE_MAP[mode_upper])
            try:
                self.mav.set_mode_send(self.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, custom_mode)
                self.get_logger().info(f'Requested set_mode: {mode_upper} (custom_mode={custom_mode}) via set_mode_send')
                return
            except Exception:
                # fallback to MAV_CMD_DO_SET_MODE if available
                self.send_command_long(mavutil.mavlink.MAV_CMD_DO_SET_MODE, [0, custom_mode])
                self.get_logger().info(f'Requested set_mode: {mode_upper} (custom_mode={custom_mode}) via command_long')
        except Exception as e:
            self.get_logger().error(f'Failed to set mode {mode}: {e}')

    def takeoff(self, altitude_m: float = 10.0):
        """Send NAV_TAKEOFF command (MAV_CMD_NAV_TAKEOFF, 22). Note: PX4 may require being in GUIDED/ALTCTL."""
        try:
            # params: min_pitch, yaw, latitude, longitude, altitude
            # Using param5 as altitude; other params left zero (takeoff at current location)
            params = [0, 0, 0, 0, float(altitude_m)]
            self.send_command_long(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, params)
        except Exception as e:
            self.get_logger().error(f'Takeoff command failed: {e}')

    def goto(self, lat: float, lon: float, alt: float):
        """Send NAV_WAYPOINT (MAV_CMD_NAV_WAYPOINT) with coordinates — not a mission push; triggers immediate waypoint."""
        try:
            params = [0, 0, 0, 0, float(lat), float(lon), float(alt)]
            # MAV_CMD_NAV_WAYPOINT is 16, but direct CMD_NAV_WAYPOINT may be interpreted as mission item
            self.send_command_long(mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, params)
        except Exception as e:
            self.get_logger().error(f'Goto command failed: {e}')

    def rtl(self):
        """Return to launch using MAV_CMD_NAV_RETURN_TO_LAUNCH (20)"""
        try:
            self.send_command_long(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, [])
        except Exception as e:
            self.get_logger().error(f'RTL command failed: {e}')

    # ---------- ROS2 command callback ----------

    def cmd_callback(self, msg: String):
        """Receive JSON command messages on /wildhawk/px4_command"""
        try:
            payload = json.loads(msg.data)
            cmd = payload.get('cmd', '').lower()
            self.get_logger().info(f'Received px4_command: {payload}')
            if cmd == 'arm':
                self.arm_disarm(True)
            elif cmd == 'disarm':
                self.arm_disarm(False)
            elif cmd == 'set_mode':
                mode = payload.get('mode', '')
                if mode:
                    self.set_mode_px4(mode)
                else:
                    self.get_logger().error('set_mode missing "mode" parameter')
            elif cmd == 'takeoff':
                alt = float(payload.get('altitude', payload.get('alt', 10.0)))
                self.takeoff(alt)
            elif cmd == 'goto':
                lat = float(payload.get('lat'))
                lon = float(payload.get('lon'))
                alt = float(payload.get('alt', 10.0))
                self.goto(lat, lon, alt)
            elif cmd == 'rtl':
                self.rtl()
            else:
                self.get_logger().warn(f'Unknown px4 command: {cmd}')
        except Exception as e:
            self.get_logger().error(f'Error handling px4 command: {e}\n{traceback.format_exc()}')

    # ---------- MAVLink receive loop ----------

    def mavloop(self):
        """Background loop: read MAVLink messages, publish telemetry JSONs."""
        while rclpy.ok() and not self.mav_thread_stop.is_set():
            try:
                # Ensure connection
                if self.mav is None:
                    self.get_logger().info('MAVLink connection missing — attempting reconnect...')
                    try:
                        self.connect_to_mavlink()
                    except Exception as e:
                        self.get_logger().error(f'Reconnect attempt failed: {e}')
                        time.sleep(2.0)
                        continue

                # Blocking read with timeout
                m = self.mav.recv_match(blocking=True, timeout=1.0)
                if m is None:
                    # timed out; check heartbeat age
                    if time.time() - self.last_heartbeat_time > self.heartbeat_timeout:
                        self.get_logger().warning('No heartbeat recently; consider reconnecting.')
                    continue

                msg_name = m.get_type()
                now = time.time()

                if msg_name == 'HEARTBEAT':
                    self.last_heartbeat_time = now
                    hb = {
                        'type': int(m.type),
                        'autopilot': int(m.autopilot),
                        'base_mode': int(m.base_mode),
                        'custom_mode': int(m.custom_mode),
                        'system_status': int(m.system_status),
                        'timestamp': now
                    }
                    self.publish('/px4/telemetry', {'heartbeat': hb})
                elif msg_name == 'ATTITUDE':
                    att = {
                        'roll': float(m.roll),
                        'pitch': float(m.pitch),
                        'yaw': float(m.yaw),
                        'rollspeed': float(m.rollspeed),
                        'pitchspeed': float(m.pitchspeed),
                        'yawspeed': float(m.yawspeed),
                        'timestamp': now
                    }
                    self.publish('/px4/attitude', att)
                elif msg_name == 'GLOBAL_POSITION_INT':
                    pos = {
                        'lat': m.lat / 1e7,
                        'lon': m.lon / 1e7,
                        'alt': m.alt / 1000.0,  # mm -> m
                        'relative_alt': m.relative_alt / 1000.0,
                        'vx': m.vx / 100.0,
                        'vy': m.vy / 100.0,
                        'vz': m.vz / 100.0,
                        'hdg': m.hdg / 100.0,
                        'timestamp': now
                    }
                    self.publish('/px4/global_position', pos)
                elif msg_name == 'SYS_STATUS':
                    ss = {
                        'voltage_battery': float(m.voltage_battery) / 1000.0,
                        'current_battery': float(m.current_battery) / 100.0 if m.current_battery != -1 else None,
                        'battery_remaining': int(m.battery_remaining),
                        'onboard_control_sensors_present': int(m.onboard_control_sensors_present),
                        'timestamp': now
                    }
                    self.publish('/px4/sys_status', ss)
                elif msg_name == 'GPS_RAW_INT' or msg_name == 'GPS_STATUS' or msg_name == 'GPS':
                    # handle GPS messages generically
                    try:
                        gps = {}
                        if hasattr(m, 'lat') and hasattr(m, 'lon'):
                            gps['lat'] = getattr(m, 'lat') / 1e7
                            gps['lon'] = getattr(m, 'lon') / 1e7
                        if hasattr(m, 'alt'):
                            gps['alt'] = getattr(m, 'alt') / 1000.0
                        gps['timestamp'] = now
                        self.publish('/px4/telemetry', {'gps': gps})
                    except Exception:
                        pass
                elif msg_name == 'HIGHRES_IMU':
                    imu = {
                        'xacc': float(m.xacc),
                        'yacc': float(m.yacc),
                        'zacc': float(m.zacc),
                        'xgyro': float(m.xgyro),
                        'ygyro': float(m.ygyro),
                        'zgyro': float(m.zgyro),
                        'timestamp': now
                    }
                    self.publish('/px4/telemetry', {'highres_imu': imu})
                else:
                    # You may add handling for more message types here
                    pass

            except Exception as e:
                self.get_logger().error(f'Exception in mavloop: {e}\n{traceback.format_exc()}')
                # Sleep briefly to avoid busy-looping on persistent error
                time.sleep(0.5)

        self.get_logger().info('MAVLoop exiting.')

    def publish(self, topic: str, payload: Dict[str, Any]):
        """Publish JSON payload to telemetry topic(s)."""
        try:
            msg = String()
            msg.data = json.dumps({'topic': topic, 'payload': payload, 'ts': time.time()})
            # Primary telemetry
            self.telemetry_pub.publish(msg)
            # Also publish to specific topics if applicable
            if topic == '/px4/attitude':
                self.attitude_pub.publish(String(data=json.dumps(payload)))
            elif topic == '/px4/global_position':
                self.position_pub.publish(String(data=json.dumps(payload)))
            elif topic == '/px4/sys_status':
                self.status_pub.publish(String(data=json.dumps(payload)))
        except Exception as e:
            self.get_logger().error(f'Failed to publish telemetry: {e}')

    def publish_heartbeat_status(self):
        """Periodically publish a small heartbeat to indicate bridge alive."""
        alive = {
            'connected': bool(self.mav is not None),
            'last_heartbeat_age': time.time() - self.last_heartbeat_time if self.last_heartbeat_time else None,
            'timestamp': time.time()
        }
        try:
            msg = String()
            msg.data = json.dumps({'topic': '/px4/bridge_status', 'payload': alive})
            self.telemetry_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish bridge status: {e}')

    def destroy_node(self):
        # Stop thread
        self.mav_thread_stop.set()
        if self.mav is not None:
            try:
                self.mav.close()
            except Exception:
                pass
        super().destroy_node()


# ---------- Entry point ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description='PX4 MAVLink <-> ROS2 bridge node')
    parser.add_argument('--connection_url', '-c', default='udp:127.0.0.1:14540',
                        help='MAVLink connection URL, e.g., udp:127.0.0.1:14540 or tcp:127.0.0.1:5760 or serial:/dev/ttyUSB0:57600')
    parser.add_argument('--target_system', type=int, default=1, help='Target system ID (PX4 system id)')
    parser.add_argument('--target_component', type=int, default=1, help='Target component id')
    parser.add_argument('--heartbeat_period', type=float, default=1.0, help='Heartbeat publish period (s)')
    parser.add_argument('--heartbeat_timeout', type=float, default=5.0, help='Heartbeat timeout for warnings (s)')
    args, unknown = parser.parse_known_args(argv)

    rclpy.init(args=[])
    node = PX4BridgeNode(connection_url=args.connection_url,
                         target_system=args.target_system,
                         target_component=args.target_component,
                         heartbeat_period=args.heartbeat_period,
                         heartbeat_timeout=args.heartbeat_timeout)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('User requested shutdown.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
