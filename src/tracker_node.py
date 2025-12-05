#!/usr/bin/env python3
"""
tracker_node.py

Lightweight multi-object tracker for WildHawk 1.0.

Features:
- Subscribes to detection JSON on /wildhawk/detections (std_msgs/String).
  Detection JSON format expected:
  {
    "timestamp": 1234567890.0,
    "detections": [
      {"label":"deer","class_id":1,"score":0.87,"bbox":{"x":10,"y":20,"w":100,"h":80}},
      ...
    ]
  }

- Optionally subscribes to annotated detection image on /wildhawk/detection_image (sensor_msgs/Image)
  to draw tracked IDs and publish an annotated image on /wildhawk/tracks_image.

- Publishes tracking results as JSON on /wildhawk/tracks:
  {
    "timestamp": 1234567890.0,
    "tracks": [
      {"track_id": 1, "label": "deer", "class_id": 1, "score": 0.87, "bbox": {"x":..., "y":..., "w":..., "h":...}, "age":2, "hits":5},
      ...
    ]
  }

Tracking method:
- Uses IOU-based greedy assignment (no external dependencies).
- Maintains Track objects with simple lifecycle (age, hits, time_since_update).
- New detections create new tracks; unmatched tracks are kept up to max_age before deletion.

Parameters (with defaults):
- max_age: 3          # how many missed frames to keep an unmatched track
- min_hits: 1         # minimal hits before reporting a track as confirmed
- iou_threshold: 0.3  # IOU threshold for matching
- publish_rate: 5.0   # Hz to publish tracks
- draw_image: True    # whether to subscribe to detection_image and publish annotated image

Author: Shaneshraje Kadu (for WildHawk 1.0)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import json
import time
import numpy as np
import threading
from typing import List, Tuple, Dict, Any, Optional

# ----------------- Helper utilities -----------------


def iou_bbox(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in format (x, y, w, h)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


# ----------------- Track class -----------------


class Track:
    _id_counter = 1
    _lock = threading.Lock()

    def __init__(self, bbox: Tuple[int, int, int, int], label: str = "", class_id: int = -1, score: float = 0.0):
        with Track._lock:
            self.track_id = Track._id_counter
            Track._id_counter += 1

        self.bbox = bbox  # (x, y, w, h)
        self.label = label
        self.class_id = class_id
        self.score = float(score)

        self.hits = 1  # number of total detection matches
        self.age = 1  # number of frames since creation
        self.time_since_update = 0  # frames since last matched
        self.history = [bbox]  # past bboxes (optional)
        self.active = True

    def predict(self):
        """
        Placeholder predict step (no motion model). Could be extended with velocity.
        Here: increment age and time_since_update.
        """
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: Tuple[int, int, int, int], label: Optional[str], class_id: Optional[int], score: Optional[float]):
        """
        Update the track with a new matched detection.
        """
        self.bbox = bbox
        if label is not None:
            self.label = label
        if class_id is not None:
            self.class_id = int(class_id)
        if score is not None:
            self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        self.active = True

    def to_dict(self) -> Dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "track_id": int(self.track_id),
            "label": str(self.label),
            "class_id": int(self.class_id) if self.class_id is not None else -1,
            "score": float(self.score),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "age": int(self.age),
            "hits": int(self.hits),
            "time_since_update": int(self.time_since_update)
        }


# ----------------- Tracker Node -----------------


class IOUTrackerNode(Node):
    def __init__(self):
        super().__init__('wildhawk_tracker_node')
        self.get_logger().info('Starting WildHawk Tracker Node...')

        # Parameters
        self.declare_parameter('max_age', 3)
        self.declare_parameter('min_hits', 1)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('draw_image', True)
        self.declare_parameter('detection_topic', '/wildhawk/detections')
        self.declare_parameter('detection_image_topic', '/wildhawk/detection_image')

        self.max_age = int(self.get_parameter('max_age').get_parameter_value().integer_value)
        self.min_hits = int(self.get_parameter('min_hits').get_parameter_value().integer_value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').get_parameter_value().double_value)
        self.publish_rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)
        self.draw_image = bool(self.get_parameter('draw_image').get_parameter_value().bool_value)
        self.detection_topic = str(self.get_parameter('detection_topic').get_parameter_value().string_value)
        self.detection_image_topic = str(self.get_parameter('detection_image_topic').get_parameter_value().string_value)

        # Internal state
        self.tracks: List[Track] = []
        self.last_detections = []  # latest detections parsed
        self.latest_frame = None  # cv2 frame if annotation used
        self.lock = threading.Lock()

        # Publishers
        self.tracks_pub = self.create_publisher(String, '/wildhawk/tracks', 10)
        self.tracks_image_pub = self.create_publisher(Image, '/wildhawk/tracks_image', 5)
        self.bridge = CvBridge()

        # Subscribers
        self.detections_sub = self.create_subscription(String, self.detection_topic, self.detections_callback, 10)
        if self.draw_image:
            # subscribe to detection image to draw labels and track IDs
            self.image_sub = self.create_subscription(Image, self.detection_image_topic, self.image_callback, 5)
        else:
            self.image_sub = None

        # Timer
        timer_period = 1.0 / max(1.0, self.publish_rate)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(f'IOU Tracker initialized (iou_threshold={self.iou_threshold}, max_age={self.max_age}, min_hits={self.min_hits})')

    # ---------- Callbacks ----------
    def detections_callback(self, msg: String):
        """
        Parse detection JSON and store detections for next tick.
        Expected format:
        {
            "timestamp": ...,
            "detections": [
                {"label":"deer","class_id":1,"score":0.87,"bbox":{"x":10,"y":20,"w":100,"h":80}},
                ...
            ]
        }
        """
        try:
            payload = json.loads(msg.data)
            detections = payload.get('detections', [])
            parsed = []
            for d in detections:
                bbox = d.get('bbox', {})
                x = int(bbox.get('x', 0))
                y = int(bbox.get('y', 0))
                w = int(bbox.get('w', 0))
                h = int(bbox.get('h', 0))
                label = d.get('label', '')
                class_id = d.get('class_id', -1)
                score = float(d.get('score', 0.0))
                parsed.append({
                    'bbox': (x, y, w, h),
                    'label': label,
                    'class_id': int(class_id) if class_id is not None else -1,
                    'score': score
                })
            with self.lock:
                self.last_detections = parsed
        except Exception as e:
            self.get_logger().error(f'Error parsing detections JSON: {e}')

    def image_callback(self, msg: Image):
        """Receive annotated detection image and store for drawing track IDs."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_frame = cv_img
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')

    # ---------- Core tracking loop ----------
    def timer_callback(self):
        with self.lock:
            detections = list(self.last_detections)  # shallow copy
            frame = None if self.latest_frame is None else self.latest_frame.copy()

        # Predict step for all tracks (no motion model beyond lifecycle)
        for tr in self.tracks:
            tr.predict()

        # Build IOU cost matrix between tracks and detections
        matched_indices = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, tr in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = iou_bbox(tr.bbox, det['bbox'])

            # Greedy matching: repeatedly pick highest IOU > threshold
            assigned_tracks = set()
            assigned_dets = set()
            while True:
                # find max
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[t_idx, d_idx]
                if max_iou < self.iou_threshold:
                    break
                # assign
                if (t_idx in assigned_tracks) or (d_idx in assigned_dets):
                    # if either already assigned, set to 0 and continue
                    iou_matrix[t_idx, d_idx] = 0.0
                    if np.max(iou_matrix) <= 0:
                        break
                    continue
                assigned_tracks.add(t_idx)
                assigned_dets.add(d_idx)
                matched_indices.append((t_idx, d_idx))
                # zero out row and column
                iou_matrix[t_idx, :] = 0.0
                iou_matrix[:, d_idx] = 0.0
                if np.max(iou_matrix) <= 0:
                    break

            # compute unmatched lists
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in assigned_tracks]
            unmatched_detections = [j for j in range(len(detections)) if j not in assigned_dets]

        # Update matched tracks
        for t_idx, d_idx in matched_indices:
            tr = self.tracks[t_idx]
            det = detections[d_idx]
            tr.update(det['bbox'], det['label'], det['class_id'], det['score'])

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_tr = Track(det['bbox'], label=det['label'], class_id=det['class_id'], score=det['score'])
            self.tracks.append(new_tr)

        # Age unmatched tracks and remove dead ones
        removed = []
        for idx in sorted(unmatched_tracks, reverse=True):
            tr = self.tracks[idx]
            # If time_since_update exceeds max_age, remove track
            if tr.time_since_update > self.max_age:
                removed.append(self.tracks.pop(idx))
            else:
                # keep track but it was not updated this cycle
                pass

        # Prepare output: only report tracks with hits >= min_hits
        tracks_out = []
        for tr in self.tracks:
            if tr.hits >= self.min_hits or tr.time_since_update == 0:
                tracks_out.append(tr.to_dict())

        # Publish tracks JSON
        out_payload = {
            "timestamp": time.time(),
            "tracks": tracks_out
        }
        try:
            msg = String()
            msg.data = json.dumps(out_payload)
            self.tracks_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish tracks: {e}')

        # Publish annotated image if available
        if frame is not None:
            # draw tracks on frame
            for tr in self.tracks:
                x, y, w, h = tr.bbox
                # color by id (simple deterministic color)
                color = (
                    (37 * tr.track_id) % 255,
                    (17 * tr.track_id) % 255,
                    (97 * tr.track_id) % 255
                )
                cv2 = __import__('cv2')  # import local to avoid top-level dependency detection issues
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label_text = f'ID:{tr.track_id} {tr.label} {tr.score:.2f}'
                cv2.putText(frame, label_text, (x, max(10, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                self.tracks_image_pub.publish(img_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge error publishing tracks image: {e}')
            except Exception as e:
                self.get_logger().error(f'Unexpected error publishing image: {e}')

        # Optionally log small summary
        self.get_logger().debug(f'Tracks published: {len(tracks_out)}, total_active_tracks: {len(self.tracks)}, removed: {len(removed)}')

    def destroy_node(self):
        with self.lock:
            self.tracks.clear()
        super().destroy_node()


# ----------------- Entry point -----------------

def main(args=None):
    rclpy.init(args=args)
    node = IOUTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down WildHawk Tracker Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
