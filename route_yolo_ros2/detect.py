import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, UInt32, Image
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import torch
import gc
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from route_yolo_ros2.srv import DetectObject  


class YoloDetector(Node):
    def __init__(self):
        super().__init__('YoloDetector')
        self.bridge = CvBridge()
        self.cam_image = None
        self.last_count = 0
        self.last_size = 0

        # Declare parameters
        self.declare_parameter('enable', True)
        self.declare_parameter('flip_image', False)
        self.declare_parameter('image_resize', 640)

        # Load YOLO model paths from package share
        pkg_share = get_package_share_directory('route_yolo_ros2')
        self.model_coco_path = os.path.join(pkg_share, 'models', 'coco.pt')
        self.model_stop_path = os.path.join(pkg_share, 'models', 'stop.pt')
        self.model_tire_path = os.path.join(pkg_share, 'models', 'tire.pt')

        # Subscribers
        self.create_subscription(ROSImage, '/camera/image_raw', self.image_callback, 10)

        # Service
        self.create_service(DetectObject, 'detect_object', self.handle_detection_request)

        self.get_logger().info("YOLO Service-Based Detector Initialized")

    def image_callback(self, msg):
        if not self.get_parameter('enable').get_parameter_value().bool_value:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        resize_val = self.get_parameter('image_resize').get_parameter_value().integer_value
        img = self.letterbox_resize(img, (resize_val, resize_val))

        if self.get_parameter('flip_image').get_parameter_value().bool_value:
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)

        self.cam_image = img

    def handle_detection_request(self, request, response):
        if self.cam_image is None:
            self.get_logger().warn("No image received yet")
            response.count = 0
            response.size = 0
            return response

        if request.target not in ['stop', 'tire', 'person']:
            self.get_logger().warn(f"Unknown detection target: {request.target}")
            response.count = 0
            response.size = 0
            return response

        self.detect(request.target)
        response.count = self.last_count
        response.size = self.last_size
        return response

    def detect(self, mode):
        model_path = {
            'stop': self.model_stop_path,
            'tire': self.model_tire_path,
            'person': self.model_coco_path
        }[mode]

        yolo = YOLO(model_path)
        results = yolo(source=self.cam_image, device="0", stream=False, verbose=False, conf=0.5)

        label_set = {
            'stop': {'stop sign'},
            'tire': {'tire'},
            'person': {'person'}
        }[mode]

        count, biggest, _ = self.analyze_results(results, label_set)

        self.last_count = count
        self.last_size = biggest

        torch.cuda.empty_cache()
        gc.collect()

    def analyze_results(self, results, label_set):
        detected = 0
        biggest_bbox = 0
        image_size = self.cam_image.shape[0] * self.cam_image.shape[1]

        for result in results:
            boxes = result.boxes.cpu().numpy()
            labels = result.names
            for box in boxes:
                label = labels[int(box.cls)]
                if label in label_set:
                    detected += 1
                    area = 100 * ((box.xywh[0][2] * box.xywh[0][3]) / image_size)
                    if area > biggest_bbox:
                        biggest_bbox = area

        return detected, int(biggest_bbox * 100), results[0].plot()

    def letterbox_resize(self, img, size=(640, 640)):
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        if h == w:
            return cv2.resize(img, size, cv2.INTER_AREA)
        dif = max(h, w)
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        y, x = (dif - h) // 2, (dif - w) // 2
        mask[y:y+h, x:x+w] = img
        return cv2.resize(mask, size)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

