#!/usr/bin/env python3
"""
mission_coordinator.py

ROS2 node that coordinates missions for WildHawk 1.0:
- Consumes tracked objects (/wildhawk/tracks) and PX4 telemetry (/px4/global_position, /px4/attitude)
- Implements a simple state machine to react to detections:
    IDLE -> DETECTED -> DETERRENCE -> GUIDANCE -> IDLE
- Sends high-level PX4 commands via /wildhawk/px4_command (std_msgs/String containing JSON)
    Supported commands published here: {"cmd":"arm"}, {"cmd":"takeoff","altitude":...}, {"cmd":"goto", "lat":..., "lon":..., "alt":...},
    {"cmd":"set_mode","mode":"GUIDED"}, {"cmd":"rtl"}, {"cmd":"disarm"}
- Designed to be safe for SITL testing. Real hardware use requires extra safety checks.

Usage:
    ros2 run wildhawk_1.0 mission_coordinator.py
or
    python3 mission_coordinator.py

Parameters:
- detection_topic (default: /wildhawk/tracks)
- telemetry_topic (default: /px4/global_position)
- min_track_age (frames) minimum hits/age required to consider a track real
- min_score (float) minimum detection score to consider
- detection_hold_time (sec) how long to keep reacting to same detection
- deterrence_duration (sec) how long to perform deterrence behavior (e.g., circle or goto)
- guidance_altitude (m) altitude to guide animals from (above ground)
- safe_altitude (m) altitude for transit
- home_rtl_on_finish (bool) send RTL after mission finish

Author: Shaneshraje Kadu (WildHawk 1.0)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import threading
from typing import Dict, Any, List, Optional

# Simple typing aliases
TrackDict = Dict[str, Any]


class MissionCoordinator(Node):
    def __init__(self):
        super().__init__('wildhawk_mission_coordinator')
        self.get_logger().info('Starting WildHawk Mission Coordinator...')

        # Declare parameters with defaults
        self.declare_parameter('detection_topic', '/wildhawk/tracks')
        self.declare_parameter('telemetry_topic', '/px4/global_position')
        self.declare_parameter('min_track_age', 1)         # minimal hits/age to trust a track
        self.declare_parameter('min_score', 0.5)
        self.declare_parameter('detection_hold_time', 10.0)  # seconds to hold a detection state
        self.declare_parameter('deterrence_duration', 8.0)   # seconds to perform deterrence
        self.declare_parameter('guidance_altitude', 6.0)     # meters
        self.declare_parameter('safe_altitude', 12.0)        # meters for transit
        self.declare_parameter('home_rtl_on_finish', True)
        self.declare_parameter('use_sim_time', False)

        # Read parameter values
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.telemetry_topic = self.get_parameter('telemetry_topic').get_parameter_value().string_value
        self.min_track_age = int(self.get_parameter('min_track_age').get_parameter_value().integer_value)
        self.min_score = float(self.get_parameter('min_score').get_parameter_value().double_value)
        self.detection_hold_time = float(self.get_parameter('detection_hold_time').get_parameter_value().double_value)
        self.deterrence_duration = float(self.get_parameter('deterrence_duration').get_parameter_value().double_value)
        self.guidance_altitude = float(self.get_parameter('guidance_altitude').get_parameter_value().double_value)
        self.safe_altitude = float(self.get_parameter('safe_altitude').get_parameter_value().double_value)
        self.home_rtl_on_finish = bool(self.get_parameter('home_rtl_on_finish').get_parameter_value().bool_value)

        # Internal state
        self.current_tracks: Dict[int, TrackDict] = {}  # keyed by track_id
        self.last_detection_time: Optional[float] = None
        self.active_track_id: Optional[int] = None
        self.state = 'IDLE'  # IDLE, DETECTED, DETERRENCE, GUIDING, RETURNING
        self.state_enter_time = time.time()

        # Telemetry state
        self.current_position = None  # dict with lat, lon, alt
        self.last_telemetry_time = None

        # Publishers / Subscribers
        self.cmd_pub = self.create_publisher(String, '/wildhawk/px4_command', 10)
        self.tracks_sub = self.create_subscription(String, self.tracks_callback, self.detection_topic, 10)
        self.telemetry_sub = self.create_subscription(String, self.telemetry_callback, self.telemetry_topic, 10)

        # Timer to run state machine
        self.timer = self.create_timer(0.5, self.tick)  # 2 Hz tick rate

        self.get_logger().info(f'MissionCoordinator initialized. Listening for tracks on "{self.detection_topic}"')

    # ----------------- Callbacks -----------------
    def tracks_callback(self, msg: String):
        """Receive tracked objects JSON and update internal track list."""
        try:
            payload = json.loads(msg.data)
            tracks: List[TrackDict] = payload.get('tracks', [])
            now = time.time()
            # Convert list to dict keyed by track_id
            tracks_map = {}
            for t in tracks:
                tid = int(t.get('track_id', -1))
                if tid < 0:
                    continue
                tracks_map[tid] = t
                tracks_map[tid]['_recv_ts'] = now

            # Update current tracks store
            self.current_tracks = tracks_map

            # If there are valid tracks, update last_detection_time
            # Only consider tracks that meet min criteria
            valid = self.find_valid_tracks()
            if valid:
                self.last_detection_time = now
                # Pick the highest score track as active candidate
                best = max(valid, key=lambda x: x.get('score', 0.0))
                prev_active = self.active_track_id
                self.active_track_id = int(best['track_id'])
                if prev_active != self.active_track_id:
                    self.get_logger().info(f'Active track changed to id={self.active_track_id}, label={best.get("label")}, score={best.get("score"):.2f}')
            else:
                # no valid tracks
                pass
        except Exception as e:
            self.get_logger().error(f'Failed to parse tracks JSON: {e}')

    def telemetry_callback(self, msg: String):
        """Receive telemetry JSON (from px4_bridge) and parse position fields."""
        try:
            payload = json.loads(msg.data)
            # payload may be of format {'payload': {'lat':..., 'lon':..., 'alt':...}, ...}
            p = payload.get('payload', payload)
            # If this message is already just the pos/dict (as px4_bridge publishes), try to extract lat/lon/alt
            lat = None
            lon = None
            alt = None
            # Common place: payload contains lat, lon, alt
            if isinstance(p, dict):
                lat = p.get('lat') or p.get('latitude')
                lon = p.get('lon') or p.get('longitude')
                alt = p.get('alt') or p.get('relative_alt') or p.get('altitude')
            # Update telemetry only if present
            if lat is not None and lon is not None:
                self.current_position = {'lat': float(lat), 'lon': float(lon), 'alt': float(alt) if alt is not None else None}
                self.last_telemetry_time = time.time()
        except Exception:
            # Sometimes the telemetry is nested; attempt to parse more generically
            try:
                payload2 = json.loads(msg.data)
                payload_inner = payload2.get('payload', {})
                if payload_inner and 'lat' in payload_inner:
                    self.current_position = {'lat': float(payload_inner['lat']), 'lon': float(payload_inner['lon']), 'alt': float(payload_inner.get('alt', 0))}
                    self.last_telemetry_time = time.time()
            except Exception as e:
                self.get_logger().debug('Telemetry parse fallback failed.')

    # ----------------- Helpers -----------------
    def find_valid_tracks(self) -> List[TrackDict]:
        """Return list of tracks that satisfy min_score and min_age criteria."""
        valid = []
        for tid, tr in self.current_tracks.items():
            score = float(tr.get('score', 0.0))
            age = int(tr.get('age', 0))
            if score >= self.min_score and age >= self.min_track_age:
                valid.append(tr)
        return valid

    def publish_px4_command(self, cmd: Dict[str, Any]):
        """Helper to publish a PX4 command (JSON string) to /wildhawk/px4_command."""
        try:
            msg = String()
            msg.data = json.dumps(cmd)
            self.cmd_pub.publish(msg)
            self.get_logger().info(f'Published PX4 command: {cmd}')
        except Exception as e:
            self.get_logger().error(f'Failed to publish px4_command: {e}')

    def send_guidance_point(self, lat: float, lon: float, alt: float):
        """Publish a goto command to guide the drone to specified lat/lon/alt."""
        cmd = {'cmd': 'goto', 'lat': float(lat), 'lon': float(lon), 'alt': float(alt)}
        self.publish_px4_command(cmd)

    def send_takeoff(self, altitude: float):
        """Request a takeoff to given altitude."""
        cmd = {'cmd': 'takeoff', 'altitude': float(altitude)}
        self.publish_px4_command(cmd)

    def send_set_mode(self, mode: str):
        cmd = {'cmd': 'set_mode', 'mode': mode}
        self.publish_px4_command(cmd)

    def send_rtl(self):
        cmd = {'cmd': 'rtl'}
        self.publish_px4_command(cmd)

    # ----------------- State machine tick -----------------
    def tick(self):
        now = time.time()
        # Expire old detection if not seen recently
        if self.last_detection_time and (now - self.last_detection_time) > self.detection_hold_time:
            # Clear active track
            if self.active_track_id is not None:
                self.get_logger().info('Detection hold expired — returning to IDLE')
            self.active_track_id = None
            self.current_tracks = {}
            self.last_detection_time = None
            # Transition to IDLE if not already
            if self.state != 'IDLE':
                self._change_state('IDLE')
            return

        # Decide behavior based on state
        if self.state == 'IDLE':
            # If valid detection exists, transition to DETECTED
            valid = self.find_valid_tracks()
            if valid:
                self.get_logger().info('Valid detection(s) found — transitioning to DETECTED')
                self._change_state('DETECTED')
        elif self.state == 'DETECTED':
            # On entering DETECTED, request guidance altitude and potentially takeoff or change mode
            # If currently not armed/airborne, we should request takeoff. We don't know armed state here; best-effort:
            # 1. Set GUIDED mode
            # 2. Takeoff to safe_altitude if telemetry absent or below threshold
            self._enter_detected_actions()
            # After initial actions, go to DETERRENCE
            self._change_state('DETERRENCE')
        elif self.state == 'DETERRENCE':
            # Perform deterrence maneuvers for configured duration: e.g., approach above animal then perform local waypoint
            elapsed = now - self.state_enter_time
            if elapsed < self.deterrence_duration:
                # Continue deterrence behavior (we may issue repeated goto to a point near animal)
                self._perform_deterrence_cycle()
            else:
                # finished deterrence duration
                self.get_logger().info('Deterrence complete — transitioning to GUIDING')
                self._change_state('GUIDING')
        elif self.state == 'GUIDING':
            # Try to guide animal safely away — set a guidance waypoint slightly away from detected animal
            if self.active_track_id is None:
                self.get_logger().info('No active track in GUIDING — returning to IDLE')
                self._change_state('IDLE')
            else:
                # Find active track bounding box and pick a point to guide away
                tr = self.current_tracks.get(self.active_track_id)
                if tr is None:
                    self.get_logger().info('Active track disappeared during GUIDING — returning to IDLE')
                    self._change_state('IDLE')
                    return
                # If we have drone telemetry, compute a guidance point along the vector from animal to home or away direction.
                if self.current_position:
                    # Basic strategy: move drone to a point that is (dx,dy) offset from animal to steer it away.
                    # Here we treat animal bbox center as a local direction; lacking world coords for animal (no tracking to lat/lon),
                    # we perform a simple "hover and create noise" action by moving above the area to guidance_altitude.
                    # If camera-to-world mapping exists, replace with real lat/lon guidance.
                    lat = self.current_position['lat']
                    lon = self.current_position['lon']
                    alt = max(self.guidance_altitude, self.current_position.get('alt') or self.guidance_altitude)
                    # Issue a goto above current position (simple behavior)
                    self.get_logger().info('Sending guidance goto above current position to encourage movement away from area.')
                    self.send_guidance_point(lat, lon, alt)
                else:
                    # No telemetry — just attempt an altitude change (takeoff to guidance altitude)
                    self.get_logger().info('No position telemetry; issuing takeoff to guidance altitude.')
                    self.send_takeoff(self.guidance_altitude)

                # After issuing guidance, transition to RETURNING or IDLE after some hold
                # We will hold guidance for detection_hold_time then finish
                if self.last_detection_time and (now - self.last_detection_time) > (self.detection_hold_time / 2.0):
                    self.get_logger().info('Guidance period complete — transitioning to RETURNING')
                    self._change_state('RETURNING')
        elif self.state == 'RETURNING':
            # Optionally RTL or move to safe loiter; then disarm if configured
            # We'll call RTL if home_rtl_on_finish True, else just disarm (careful!)
            if self.home_rtl_on_finish:
                self.get_logger().info('Issuing RTL to return to launch/home.')
                self.send_rtl()
            else:
                self.get_logger().info('Home RTL disabled; no automatic RTL will be sent.')
            # After issuing RTL, transition to IDLE
            self._change_state('IDLE')
        else:
            self.get_logger().warning(f'Unknown state: {self.state}; resetting to IDLE')
            self._change_state('IDLE')

    # ----------------- State helpers -----------------
    def _change_state(self, new_state: str):
        self.get_logger().info(f'State change: {self.state} -> {new_state}')
        self.state = new_state
        self.state_enter_time = time.time()
        # Optional entry actions
        if new_state == 'DETECTED':
            pass
        elif new_state == 'DETERRENCE':
            # start deterrence timer and maybe play a sound or perform aggressive move
            pass
        elif new_state == 'GUIDING':
            pass
        elif new_state == 'RETURNING':
            pass
        elif new_state == 'IDLE':
            self.active_track_id = None

    def _enter_detected_actions(self):
        """Actions upon initial detection: set guided mode and takeoff to safe altitude if needed."""
        # Set GUIDED or OFFBOARD mode to allow commands
        self.send_set_mode('GUIDED')
        # If we have telemetry, check altitude to determine if takeoff is needed
        if self.current_position and self.current_position.get('alt') is not None:
            cur_alt = float(self.current_position.get('alt') or 0.0)
            if cur_alt < (self.safe_altitude - 1.0):
                # Request takeoff to safe altitude
                self.get_logger().info(f'Current altitude {cur_alt:.1f} m < safe_altitude {self.safe_altitude} m -> takeoff')
                self.send_takeoff(self.safe_altitude)
            else:
                self.get_logger().info('Already above safe altitude, skipping takeoff.')
        else:
            # No telemetry: request takeoff as best-effort
            self.get_logger().info('Telemetry unavailable: issuing takeoff to safe_altitude as best-effort.')
            self.send_takeoff(self.safe_altitude)

    def _perform_deterrence_cycle(self):
        """
        Concrete deterrence behavior:
        - If telemetry available and track available, attempt to move to point slightly offset from current position
          to create visible presence (e.g., orbit or move along short waypoint).
        - Otherwise, repeatedly re-issue a takeoff/altitude command to produce motion.
        Note: for true deterrence you'd implement audio/visual cues; here we use movement commands as placeholders.
        """
        # Try to get current telemetry
        if self.current_position:
            lat = self.current_position['lat']
            lon = self.current_position['lon']
            alt = max(self.safe_altitude, self.current_position.get('alt') or self.safe_altitude)
            # Compute a small offset (approx lat/lon offset by meters)
            # Approx: 1 meter ~ 9e-6 degrees latitude; longitude scale depends on lat. We'll use conservative offsets.
            meters_offset = 8.0  # move 8 meters to side as deterrence
            lat_offset = (meters_offset / 111000.0)  # degrees
            lon_offset = (meters_offset / (111000.0 * max(0.0001, abs(lat))))  # crude; fallback if lat small
            # Alternate offsets to create small "patrol" pattern
            t = time.time()
            sign = 1 if int(t) % 2 == 0 else -1
            target_lat = lat + sign * lat_offset
            target_lon = lon + sign * lon_offset
            target_alt = alt
            self.get_logger().debug(f'Deterrence goto: lat={target_lat:.7f} lon={target_lon:.7f} alt={target_alt:.1f}')
            self.send_guidance_point(target_lat, target_lon, target_alt)
        else:
            # No telemetry; try to issue a nominal takeoff to create vertical motion
            self.get_logger().debug('Deterrence: no telemetry, re-issuing takeoff to safe_altitude')
            self.send_takeoff(self.safe_altitude)

    # ----------------- Shutdown -----------------
    def destroy_node(self):
        self.get_logger().info('Shutting down MissionCoordinator...')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('User requested shutdown.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```
