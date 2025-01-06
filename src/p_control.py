#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point32
from std_msgs.msg import String 
from project_ojakdong.msg import DetectionResult, ClassifiedResult

# 상수 정의
SAFE_DISTANCE = 0.3
FOLLOW_DISTANCE = 0.3
SEARCH_ROTATION_SPEED = 1.8
OBSTACLE_AVOIDANCE_SPEED = 0.8
OBSTACLE_AVOIDANCE_TURN_SPEED = 2.4
SEARCH_DURATION = 8
BACKUP_SPEED = -0.8
BACKUP_DURATION = 2
TURN_DURATION = 8
AVOIDANCE_TIMEOUT = 10
MAX_ANGULAR_SPEED = 1.8

class TargetFollower:
    def __init__(self):
        rospy.init_node('target_follower', anonymous=True)

        # 퍼블리셔 및 서브스크라이버 설정
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/robot_status', String, queue_size=10)

        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/detection_result', DetectionResult, self.detection_callback)
        rospy.Subscriber('/classified_result', ClassifiedResult, self.classified_callback)

        # 변수 초기화
        self.twist = Twist()
        self.target_user_name = None
        self.target_center = None
        self.closest_distance = float('inf')
        self.closest_angle = 0
        self.image_width = None
        self.image_height = None
        self.lidar_ranges = []
        self.target_detected = False
        self.state = "searching"
        self.state_start_time = rospy.Time.now()

        # 데이터 통합
        self.detection_centers = []
        self.detection_boxes = []
        self.detection_class_ids = []
        self.classified_names = []

        rospy.loginfo("TargetFollower initialized and running...")

    def scan_callback(self, data):
        """라이다 데이터를 통해 가장 가까운 거리 확인"""
        self.closest_distance = min(data.ranges)
        self.closest_angle = data.ranges.index(self.closest_distance)
        self.closest_angle = (self.closest_angle * data.angle_increment * 180 / np.pi) - 180
        self.lidar_ranges = data.ranges
        self.state_machine()

    def detection_callback(self, msg):
        """DetectionResult 메시지 처리"""
        self.detection_centers = msg.centers
        self.detection_boxes = msg.boxes
        self.detection_class_ids = msg.class_ids
        self.image_width = 640  # 고정된 이미지 폭 (필요시 메시지에서 가져오기)
        self.image_height = 480  # 고정된 이미지 높이

    def classified_callback(self, msg):
        """ClassifiedResult 메시지 처리"""
        self.target_user_name = msg.target_user_name
        self.classified_names = msg.custom_class_names
        rospy.loginfo(f"Tracking user: {self.target_user_name}")
        self.process_classification_data()

    def process_classification_data(self):
        """DetectionResult와 ClassifiedResult를 결합하여 사용자 추적"""
        if not self.detection_centers or not self.classified_names:
            self.target_detected = False
            self.target_center = None
            self.state = "searching"
            return

        # 선택된 사용자 필터링
        for idx, class_name in enumerate(self.classified_names):
            if class_name == self.target_user_name and idx < len(self.detection_centers):
                self.target_center = self.detection_centers[idx]
                self.target_detected = True
                self.state = "following"
                # rospy.loginfo(f"Target detected at center: {self.target_center}")
                break
        else:
            self.target_detected = False
            self.target_center = None
            self.state = "searching"

        self.state_machine()

    def state_machine(self):
        if self.state == "searching":
            self.search()
        elif self.state == "following":
            self.follow_target()
        elif self.state == "avoiding_obstacle":
            self.avoid_obstacle()
        elif self.state == "recovery":
            self.recovery()
        elif self.state == "STOP":
            self.stop_robot()

    def follow_target(self):
        if self.target_detected:
            self.status_pub.publish("Following target")

            # 중심점 계산
            image_center_x = self.image_width / 2
            error_x = self.target_center.x - image_center_x

            # 장애물 감지
            if self.closest_distance < SAFE_DISTANCE:
                rospy.logwarn("Obstacle detected! Switching to obstacle avoidance.")
                self.state = "avoiding_obstacle"
                self.state_start_time = rospy.Time.now()
                return  # 회피 상태로 즉시 전환

            # PID 기반 회전 속도 계산
            Kp = 0.008
            Kd = 0.002
            if not hasattr(self, 'prev_error'):
                self.prev_error = 0

            error_diff = error_x - self.prev_error
            angular_speed = -(Kp * error_x + Kd * error_diff)
            self.prev_error = error_x

            # 회전 속도 제한
            self.twist.angular.z = max(min(angular_speed, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

            # 선속도 동적 조절
            max_speed = 0.4
            scaling_factor = 1.0 - (abs(error_x) / (self.image_width / 2))
            linear_speed = scaling_factor * max_speed
            linear_speed = max(0.01, linear_speed)
            
            self.twist.linear.x = linear_speed if self.closest_distance > SAFE_DISTANCE else 0.0

            if self.target_center is None:
                rospy.loginfo("Target stopped, stopping robot.")
                self.state = "STOP"

            self.cmd_vel_pub.publish(self.twist)
            rospy.loginfo(f"Following target. Error X: {error_x}, Closest Distance: {self.closest_distance}")
        else:
            self.state = "searching"


    def search(self):
        """타겟을 찾기 위해 회전"""
        self.status_pub.publish("Searching for target")
        self.twist.linear.x = 0.0
        self.twist.angular.z = SEARCH_ROTATION_SPEED

        if self.closest_distance < SAFE_DISTANCE:
            self.state = "avoiding_obstacle"
            self.state_start_time = rospy.Time.now()

        self.cmd_vel_pub.publish(self.twist)

    def avoid_obstacle(self):
        """장애물 회피 - 후방 장애물 감지 및 동적 속도 조절"""
        self.status_pub.publish("Avoiding obstacle")

        # 장애물과의 최소 거리 및 각도 측정
        min_distance = min(self.lidar_ranges)
        min_index = self.lidar_ranges.index(min_distance)
        angle_to_obstacle = (min_index * 360 / len(self.lidar_ranges)) - 180  # -180 ~ 180도

        # 후방 장애물 감지 (170~190도)
        rear_range = self.lidar_ranges[170:190]  # 후방 20도 범위 데이터
        rear_min_distance = min(rear_range) if len(rear_range) > 0 else float('inf')

        # 동적 속도 설정
        dynamic_backup_speed = np.interp(min_distance, [0, SAFE_DISTANCE], [-0.8, 0])  # 장애물 가까울수록 후진 속도 증가
        dynamic_turn_speed = np.interp(abs(angle_to_obstacle), [0, 90], [0.8, 2.4])    # 장애물 방향에 따라 회전 속도 조절

        # 장애물 위치에 따라 후진 + 회전 (후방 장애물이 없을 경우에만 후진)
        if rear_min_distance > SAFE_DISTANCE:
            self.twist.linear.x = dynamic_backup_speed
        else:
            self.twist.linear.x = 0.0  # 후방 장애물이 있으면 후진 금지
            rospy.logwarn("Rear obstacle detected, stopping backward movement.")

        if angle_to_obstacle > 0:
            self.twist.angular.z = -dynamic_turn_speed  # 장애물이 오른쪽 -> 왼쪽으로 회전
        else:
            self.twist.angular.z = dynamic_turn_speed   # 장애물이 왼쪽 -> 오른쪽으로 회전

        # 장애물 회피 후 타겟 추적 상태로 복귀
        if min_distance > SAFE_DISTANCE and self.target_detected:
            rospy.loginfo("Obstacle cleared, returning to follow target.")
            self.state = "following"
        elif min_distance > SAFE_DISTANCE and not self.target_detected:
            rospy.loginfo("Obstacle cleared, returning to searching mode.")
            self.state = "searching"

        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo(f"Obstacle avoidance: Distance = {min_distance:.2f}, Angle = {angle_to_obstacle:.2f}, Rear Min Distance = {rear_min_distance:.2f}")


    def recovery(self):
        """회복 동작"""
        self.status_pub.publish("Recovery mode")
        free_angle = self.find_free_direction()

        if rospy.Time.now() - self.state_start_time < rospy.Duration(BACKUP_DURATION):
            self.twist.linear.x = BACKUP_SPEED
            self.twist.angular.z = free_angle * OBSTACLE_AVOIDANCE_TURN_SPEED / 180
        else:
            self.state = "searching"
            self.state_start_time = rospy.Time.now()

    def stop_robot(self):
        """로봇 정지"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    def find_free_direction(self):
        """가장 멀리 떨어진 방향 찾기"""
        max_distance = max(self.lidar_ranges)
        max_index = self.lidar_ranges.index(max_distance)
        free_angle = (max_index * len(self.lidar_ranges) * 180 / np.pi) - 180
        return free_angle

if __name__ == '__main__':
    try:
        follower = TargetFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down target follower.")
    finally:
        # 프로그램 종료 전에 정지 명령 보내기
        rospy.loginfo("Stopping the robot.")
        follower = TargetFollower()
        follower.stop_robot()