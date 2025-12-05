#!/usr/bin/env python3
"""
detection_node.py

ROS2 node for running TensorFlow Lite (TFLite) MobileNet/SSD detection on a video stream
and publishing detection results and an annotated image.

Features:
- Uses tflite_runtime if available, otherwise falls back to TensorFlow's interpreter.
- Subscribes to a ROS2 image topic (default: /camera/image_raw), or uses local camera if not publishing.
- Publishes detections as JSON on /wildhawk/detections (std_msgs/String).
- Publishes an annotated image on /wildhawk/detection_image (sensor_msgs/Image) for visualization.
- Parameterized model path, label file, confidence threshold, input size, camera topic, publish rate.
- Efficient pre- and post-processing optimized for MobileNet SSD TFLite models.

Requirements (add to requirements.txt):
- rclpy
- opencv-python
- numpy
- Pillow
- cv_bridge
- tflite-runtime OR tensorflow

Usage (ROS 2):
$ ros2 run <your_package> detection_node --ros-args -p model_path:="/path/to/mobilenet.tflite" -p labelmap:="/path/to/labels.txt"

Or run directly:
$ python3 detection_node.py --model models/mobilenet_animals.tflite --labels models/labels.txt

"""

import sys
import os
import json
import time
import argparse
import logging
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

# ROS 2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Try to import tflite runtime first (lightweight for edge)
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    TFLITE_RUNTIME = True
except Exception:
    try:
        # Fall back to full tensorflow if tflite_runtime not available
        from tensorflow.lite.python.interpreter import Interpreter
        from tensorflow.lite.python.interpreter import load_delegate  # type: ignore
        TFLITE_RUNTIME = False
    except Exception as e:
        raise RuntimeError(
            "No TFLite interpreter available. Install tflite-runtime or tensorflow."
        ) from e


# ---------- Helper functions ----------

def load_labels(path: str) -> Dict[int, str]:
    """Load labels file — supports simple 'index label' or one-label-per-line formats."""
    labels = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    with open(path, 'r', encoding='utf-8') as fh:
        lines = [l.strip() for l in fh.readlines() if l.strip()]
    # If lines start with numbers, parse "0 person"
    for i, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            labels[i] = parts[0]
        else:
            # If left part is integer index
            if parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
            else:
                labels[i] = line
    return labels


def preprocess_frame(frame: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess frame to feed into TFLite model.
    Resizes and normalizes to float32 in [0,1] or [-1,1] depending on model typical usage.
    This implementation returns float32 in [0,1].
    """
    h, w = input_size
    img = cv2.resize(frame, (w, h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    # Expand dims to [1, h, w, 3]
    return np.expand_dims(img_norm, axis=0)


def scale_boxes(boxes: np.ndarray, frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """Scale normalized boxes (ymin, xmin, ymax, xmax) to pixel coordinates (x, y, w, h)."""
    h, w = frame_shape[:2]
    results = []
    for b in boxes:
        ymin, xmin, ymax, xmax = b
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        # Clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        results.append((x1, y1, x2 - x1, y2 - y1))
    return results


def parse_tflite_detections(interpreter: Interpreter, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Parse outputs from a TFLite detection model into a list of detections.
    This supports common SSD Mobilenet TFLite outputs:
    - 'TFLite_Detection_PostProcess' (boxes), 'TFLite_Detection_PostProcess:1' (classes), 'TFLite_Detection_PostProcess:2' (scores), 'TFLite_Detection_PostProcess:3' (num)
    or the alternate numeric output ordering.
    """
    output_details = interpreter.get_output_details()
    # Map name -> tensor
    out = {}
    for od in output_details:
        name = od.get('name')
        out[name] = od

    # Try common indices
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = interpreter.get_tensor(output_details[3]['index'])
    except Exception:
        # Try alternate ordering
        # Find which outputs have shapes matching expected detection shapes
        boxes = None
        scores = None
        classes = None
        num = None
        for od in output_details:
            shape = od['shape']
            # boxes often has shape [1, N, 4]
            if len(shape) == 3 and shape[-1] == 4:
                boxes = interpreter.get_tensor(od['index'])
            elif len(shape) == 2 and shape[-1] == 1:
                # could be classes or scores -> disambiguate via type (float vs int)
                tensor = interpreter.get_tensor(od['index'])
                if tensor.dtype == np.float32 and scores is None:
                    scores = tensor
                elif tensor.dtype in (np.int32, np.int64) and classes is None:
                    classes = tensor
            elif len(shape) == 1:
                # maybe num
                num = interpreter.get_tensor(od['index'])
        if boxes is None or scores is None:
            raise RuntimeError("Unexpected TFLite output format — cannot parse detections.")

    # Flatten arrays to expected shapes
    # boxes: [1, N, 4], classes: [1, N] or [1, N, 1], scores: [1, N]
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)
    if classes.ndim > 1:
        classes = classes.flatten()

    detections = []
    for i, s in enumerate(scores):
        if s < score_threshold:
            continue
        box = boxes[i]
        cls = int(classes[i])
        detections.append({
            'box': box.tolist(),  # normalized ymin,xmin,ymax,xmax
            'score': float(s),
            'class_id': cls
        })
    return detections


# ---------- ROS2 Node ----------

class DetectionNode(Node):
    def __init__(self, args):
        super().__init__('wildhawk_detection_node')
        self.get_logger().info('Starting WildHawk Detection Node...')

        # Parameters (also allow CLI overrides)
        self.declare_parameter('model_path', args.model)
        self.declare_parameter('labelmap', args.labels)
        self.declare_parameter('score_threshold', float(args.threshold))
        self.declare_parameter('camera_topic', args.camera_topic)
        self.declare_parameter('publish_rate', float(args.publish_rate))
        self.declare_parameter('input_width', int(args.input_width))
        self.declare_parameter('input_height', int(args.input_height))
        self.declare_parameter('use_camera_if_no_topic', bool(args.use_camera))
        self.declare_parameter('camera_index', int(args.camera_index))

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.labelmap_path = self.get_parameter('labelmap').get_parameter_value().string_value
        self.score_threshold = self.get_parameter('score_threshold').get_parameter_value().double_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.input_w = self.get_parameter('input_width').get_parameter_value().integer_value
        self.input_h = self.get_parameter('input_height').get_parameter_value().integer_value
        self.use_camera = self.get_parameter('use_camera_if_no_topic').get_parameter_value().bool_value
        self.camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value

        # Load labels
        try:
            self.labels = load_labels(self.labelmap_path)
            self.get_logger().info(f'Loaded {len(self.labels)} labels.')
        except Exception as e:
            self.get_logger().error(f'Failed to load labels: {e}')
            self.labels = {}

        # Load TFLite model
        self.get_logger().info(f'Loading model: {self.model_path}')
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Try to load GPU delegate if available (optional)
        delegates = []
        try:
            # Example: Edge TPU or GPU delegates could be loaded via load_delegate
            # For Raspberry Pi with Coral USB, you'd use the edgetpu delegate (if installed).
            # This code leaves delegate loading to the user to customize.
            pass
        except Exception as e:
            self.get_logger().warning(f'Could not create delegates: {e}')

        self.interpreter = Interpreter(model_path=self.model_path, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()

        # Get input details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Setup publishers
        self.detections_pub = self.create_publisher(String, '/wildhawk/detections', 10)
        self.image_pub = self.create_publisher(Image, '/wildhawk/detection_image', 5)
        self.bridge = CvBridge()

        # Subscribe to camera topic if available
        self.subscription = None
        if self.topic_exists(self.camera_topic):
            self.subscription = self.create_subscription(Image, self.camera_topic, self.image_callback, 5)
            self.get_logger().info(f'Subscribed to camera topic: {self.camera_topic}')
            self.capture = None
        else:
            if self.use_camera:
                self.get_logger().warning(f'Camera topic {self.camera_topic} not found. Falling back to local camera index {self.camera_index}.')
                self.capture = cv2.VideoCapture(self.camera_index)
                if not self.capture.isOpened():
                    raise RuntimeError(f'Cannot open local camera index {self.camera_index}')
            else:
                self.get_logger().error(f'Camera topic {self.camera_topic} not found and use_camera_if_no_topic is False.')
                raise RuntimeError('No camera available.')

        # Rate timer
        timer_period = 1.0 / max(1.0, self.publish_rate)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.latest_frame = None
        self.last_publish_time = 0.0

    def topic_exists(self, topic_name: str) -> bool:
        """Simple check for topic existence."""
        try:
            topics = self.get_topic_names_and_types()
            return any(topic_name == t[0] for t in topics)
        except Exception:
            return False

    def image_callback(self, msg: Image):
        """Save latest frame from ROS topic."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = cv_image
            # don't process immediately here — do in timer callback to control rate
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')

    def timer_callback(self):
        """Run detection at configured publish_rate."""
        if self.capture is not None:
            ret, frame = self.capture.read()
            if not ret:
                self.get_logger().warning('Failed to read frame from local camera.')
                return
            self.latest_frame = frame

        if self.latest_frame is None:
            # No frame yet
            return

        frame = self.latest_frame.copy()
        start_time = time.time()

        # Preprocess
        input_tensor = preprocess_frame(frame, (self.input_h, self.input_w)).astype(np.float32)

        # Prepare input tensor index
        try:
            input_index = self.input_details[0]['index']
            # Check if model expects float or uint8
            if self.input_details[0]['dtype'] == np.uint8:
                # quantized model: need to convert from float [0,1] -> [0,255] and cast to uint8
                input_tensor_uint8 = (input_tensor * 255).astype(np.uint8)
                self.interpreter.set_tensor(input_index, input_tensor_uint8)
            else:
                self.interpreter.set_tensor(input_index, input_tensor)
        except Exception as e:
            self.get_logger().error(f'Failed to set input tensor: {e}')
            return

        # Run inference
        try:
            self.interpreter.invoke()
        except Exception as e:
            self.get_logger().error(f'Interpreter invoke error: {e}')
            return

        # Parse detection results
        try:
            parsed = parse_tflite_detections(self.interpreter, score_threshold=self.score_threshold)
        except Exception as e:
            self.get_logger().error(f'Failed to parse detections: {e}')
            parsed = []

        # Convert normalized boxes to pixel coordinates
        boxes_norm = [d['box'] for d in parsed]
        boxes_px = scale_boxes(np.array(boxes_norm) if boxes_norm else np.zeros((0, 4)), frame.shape)

        detections_output = []
        for det, box_px in zip(parsed, boxes_px):
            class_id = det['class_id']
            label = self.labels.get(class_id, str(class_id))
            score = det['score']
            x, y, w, h = box_px
            detections_output.append({
                'label': label,
                'class_id': int(class_id),
                'score': float(score),
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
            # Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Publish detections JSON
        detections_msg = {
            'timestamp': time.time(),
            'detections': detections_output
        }
        try:
            ros_msg = String()
            ros_msg.data = json.dumps(detections_msg)
            self.detections_pub.publish(ros_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish detections: {e}')

        # Publish annotated image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error publishing image: {e}')

        elapsed = time.time() - start_time
        self.get_logger().debug(f'Inference done in {elapsed*1000:.1f} ms, found {len(detections_output)} detections.')

    def destroy_node(self):
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        super().destroy_node()


# ---------- Entry point ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description='WildHawk TFLite detection node')
    parser.add_argument('--model', '-m', required=False, default='models/mobilenet_animals.tflite', help='Path to TFLite model file')
    parser.add_argument('--labels', '-l', required=False, default='models/labels.txt', help='Path to label map file')
    parser.add_argument('--threshold', '-t', required=False, default=0.5, type=float, help='Score threshold for detections')
    parser.add_argument('--camera_topic', required=False, default='/camera/image_raw', help='ROS image topic to subscribe to (default /camera/image_raw)')
    parser.add_argument('--publish_rate', required=False, default=5.0, type=float, help='Publish rate (Hz)')
    parser.add_argument('--input_width', required=False, default=300, type=int, help='Model input width')
    parser.add_argument('--input_height', required=False, default=300, type=int, help='Model input height')
    parser.add_argument('--use_camera', required=False, default=True, type=lambda x: (str(x).lower() in ('true','1','yes')), help='Use local camera fallback if topic not present')
    parser.add_argument('--camera_index', required=False, default=0, type=int, help='Local camera index for fallback')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose debug logging')
    parsed = parser.parse_args(args=None if argv is None else argv)

    # Setup logging (ROS 2 handles log, but keep Python logging too)
    if parsed.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Start ROS node
    rclpy.init(args=sys.argv)
    node = None
    try:
        node = DetectionNode(parsed)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Interrupted by user, shutting down.')
    except Exception as e:
        print('Node error:', e, file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
