import rclpy
from rclpy.node import Node
from route_yolo_ros2.srv import DetectObject


class YoloDetectClient(Node):
    def __init__(self):
        super().__init__('yolo_detect_client')
        self.cli = self.create_client(DetectObject, 'detect_object')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

        self.req = DetectObject.Request()

    def send_request(self, target='person'):
        self.req.target = target
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    client = YoloDetectClient()

    target = 'person'  # Change to stop or tire  
    response = client.send_request(target)

    client.get_logger().info(
        f'Result for \"{target}\": count={response.count}, size={response.size}'
    )

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
