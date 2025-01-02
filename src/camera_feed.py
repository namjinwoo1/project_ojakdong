#!/usr/bin/env python3
import rospy
import copy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String  # String 타입을 import
from project_ojakdong.msg import DetectionResult, ClassifiedResult
from cv_bridge import CvBridge
import threading


class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.target_user_name = None  # 추적할 사용자 이름
        self.image = None

        # ROS 퍼블리셔
        self.robot_state_publisher = rospy.Publisher('/robot_state', String, queue_size=10)
        self.robot_action_publisher = rospy.Publisher('/robot_action', String, queue_size=10)

        # 메시지 구독
        self.detection_subscriber = rospy.Subscriber('/detection_result', DetectionResult, self.detection_callback)
        self.classified_subscriber = rospy.Subscriber('/classified_result', ClassifiedResult, self.classified_callback)
        self.lidar_subscriber = rospy.Subscriber('/scan', LaserScan, self.lidar_listener)

        # 상태 변수 초기화
        self.robot_state = "IDLE"
        self.lidar_distance = float('inf')
        self.target_center = None

        # 데이터 통합 처리용
        self.detection_centers = []
        self.detection_boxes = []
        self.classified_names = []

        # 데이터 처리 타이머
        rospy.Timer(rospy.Duration(0.1), lambda event: self.process_data())

        # OpenCV 설정
        # self.cv2_frame_size = (640, 480)
        # cv2.namedWindow("robot_view", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("robot_view", *self.cv2_frame_size)

    def lidar_listener(self, msg):
        self.lidar_distance = min(msg.ranges)

    def detection_callback(self, msg):
        try:
            if msg.centers and msg.image.data:
                self.detection_centers = msg.centers
                self.detection_boxes = msg.boxes
                self.image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error in detection_callback: {e}")

    def classified_callback(self, msg):
        try:
            if msg.custom_class_names:
                self.classified_names = msg.custom_class_names

                # 타겟 사용자 이름 자동 결정
                if self.target_user_name is None:
                    self.target_user_name = msg.target_user_name
                    rospy.loginfo(f"Target user set to: {self.target_user_name}")
                rospy.loginfo("Updated ClassifiedResult data.")
        except Exception as e:
            rospy.logerr(f"Error in classified_callback: {e}")

    def process_data(self):
        try:
            if self.image is None:
                rospy.logwarn("Image data is None, skipping processing.")
                return
            
            if not self.detection_centers or not self.classified_names:
                rospy.logwarn("Detection or Classification data is incomplete, skipping processing.")
                return

            for center, class_name in zip(self.detection_centers, self.classified_names):
                if class_name == self.target_user_name:
                    self.target_center = center
                    break
            else:
                self.target_center = None

            self.update_robot_state()

        except Exception as e:
            rospy.logerr(f"Error processing data: {e}")

    def update_robot_state(self):
        if self.target_center:
            x_center = self.target_center.x
            image_width = self.image.shape[1]
            center_offset = x_center - (image_width / 2)

            if self.lidar_distance <= 0.5:
                self.robot_state = "STOP"
                self.robot_action_publisher.publish("STOP")
            else:
                self.robot_state = "FOLLOW"
                if center_offset > 30:
                    self.robot_action_publisher.publish("TURN_RIGHT")
                elif center_offset < -30:
                    self.robot_action_publisher.publish("TURN_LEFT")
                else:
                    self.robot_action_publisher.publish("FORWARD")

            # rospy.loginfo(f"Robot State: {self.robot_state}, Offset: {center_offset}")
        else:
            self.robot_state = "SEARCH"
            self.robot_action_publisher.publish("SEARCH")

        self.robot_state_publisher.publish(self.robot_state)
        if self.image is not None:
            self.visualize_result()

    def visualize_result(self):
        if self.image is None:
            rospy.logwarn("No image to visualize.")
            return

        def display():
            frame = copy.deepcopy(self.image)

            cv2.putText(frame, f"State: {self.robot_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not self.target_center:
                cv2.putText(frame, "Searching for target...", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                x_center, y_center = int(self.target_center.x), int(self.target_center.y)
                cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Target Detected", (x_center - 50, y_center - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cv2.imshow("robot_view", frame)
            # cv2.waitKey(30)

        # 별도 스레드에서 GUI 실행
        display_thread = threading.Thread(target=display)
        display_thread.daemon = True
        display_thread.start()

if __name__ == "__main__":
    rospy.init_node('camera_feed', anonymous=True)

    processor = ImageProcessor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
