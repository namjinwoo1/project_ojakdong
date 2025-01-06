#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from project_ojakdong.msg import DetectionResult
from geometry_msgs.msg import Polygon, Point32
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
import cv2
import rospkg


class CameraProcessor:
    def __init__(self):
        rospy.init_node('camera_processor', anonymous=True)

        # YOLOv8 모델 로드 (PyTorch)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('project_ojakdong')
        model_path = f"{package_path}/model/yolov8n.pt"
        self.model = YOLO(model_path)

        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using device: {self.device}")
        self.model.to(self.device)

        # ROS 설정
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)     #usb_cam
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)  #gazebo
        self.result_pub = rospy.Publisher("/detection_result", DetectionResult, queue_size=10)
        self.image_pub = rospy.Publisher("/processed_image", Image, queue_size=10)

        self.bridge = CvBridge()
        rospy.loginfo("CameraProcessor Node Started with YOLOv8 nano (PyTorch)")

    def image_callback(self, msg):
        try:
            # ROS Image → OpenCV 이미지 변환
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # YOLOv8 추론 수행
            results = self.model(frame, verbose=False)

            # DetectionResult 메시지 생성
            detection_msg = DetectionResult()
            detection_msg.header = msg.header

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표 (x1, y1, x2, y2)
                class_ids = result.boxes.cls.cpu().numpy()  # 클래스 ID
                confs = result.boxes.conf.cpu().numpy()  # 신뢰도(Confidence)

                for box, cls_id, conf in zip(boxes, class_ids, confs):
                    if conf > 0.5 and int(cls_id) == 0:  # 'person' 클래스만 필터링
                        # DetectionResult 메시지 구성
                        detection_msg.class_ids.append(int(cls_id))
                        detection_msg.confidences.append(float(conf))

                        # Polygon 형태로 박스 좌표 변환
                        polygon = Polygon()
                        p1 = Point32(box[0], box[1], 0)
                        p2 = Point32(box[2], box[1], 0)
                        p3 = Point32(box[2], box[3], 0)
                        p4 = Point32(box[0], box[3], 0)
                        polygon.points.extend([p1, p2, p3, p4])

                        # 중심 좌표 계산
                        center = Point32((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0)

                        detection_msg.centers.append(center)
                        detection_msg.boxes.append(polygon)

                        # 박스를 그려서 이미지에 표시
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

                        # 디버깅 로그
                        # rospy.loginfo(f"Detected Person - Confidence: {conf}, Box: {box}")

            # 처리된 이미지를 메시지에 추가
            processed_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            detection_msg.image = processed_image_msg  # 이미지 포함

            # 결과 퍼블리시
            self.result_pub.publish(detection_msg)
            self.image_pub.publish(processed_image_msg)  # 처리된 이미지 퍼블리시

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")


if __name__ == '__main__':
    try:
        processor = CameraProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
