#!/usr/bin/env python3

import cv2
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import rospy
import rospkg
import os
import json
from cv_bridge import CvBridge
from project_ojakdong.msg import DetectionResult, ClassifiedResult
from geometry_msgs.msg import Polygon, Point32
import subprocess
import signal
import hashlib

def compute_model_hash(model_path):
    """모델 파일의 SHA256 해시를 계산"""
    hasher = hashlib.sha256()
    with open(model_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def save_model_hash(output_dir, model_hash):
    hash_file = os.path.join(output_dir, 'model_hash.txt')
    with open(hash_file, 'w') as f:
        f.write(model_hash)

def load_model_hash(output_dir):
    hash_file = os.path.join(output_dir, 'model_hash.txt')
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return f.read().strip()
    return None

class ClassifyNode:
    
    def __init__(self, mobilenet_model_path, class_indices_path, output_dir):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_TRT_MAX_ALLOWED_ENGINES'] = '2'  # 최대 엔진 수 제한
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENTS'] = '1'  # 세그먼트 기반 엔진 활성화
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 메모리 부족 방지

        self.bridge = CvBridge()
        self.trt_saved_model_dir = os.path.join(output_dir, 'trt_saved_model')

        # 모델 해시 계산 및 TensorRT 엔진 유효성 검증
        current_model_hash = compute_model_hash(mobilenet_model_path)
        saved_model_hash = load_model_hash(output_dir)

        if saved_model_hash != current_model_hash or not os.path.exists(self.trt_saved_model_dir):
            rospy.loginfo("모델이 변경되었거나 TensorRT 엔진이 없습니다. 변환을 시작합니다...")
            self.convert_model(mobilenet_model_path, output_dir)
            save_model_hash(output_dir, current_model_hash)
        else:
            rospy.loginfo("기존 TensorRT 모델을 사용합니다.")

        # TensorRT 모델 로드
        self.mobilenet_model = tf.saved_model.load(self.trt_saved_model_dir)
        self.infer = self.mobilenet_model.signatures['serving_default']

        # 엔진 강제 빌드 (Warm-up)
        dummy_input = tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32)
        rospy.loginfo("TensorRT 엔진을 사전 빌드합니다.")
        
        try:
            _ = self.infer(dummy_input)
            rospy.loginfo("TensorRT 엔진 빌드 완료.")
        except Exception as e:
            rospy.logerr(f"엔진 사전 빌드 중 오류 발생: {e}")
            rospy.logwarn("TensorRT 엔진을 재변환합니다.")
            self.convert_model(mobilenet_model_path, output_dir)
            self.mobilenet_model = tf.saved_model.load(self.trt_saved_model_dir)
            self.infer = self.mobilenet_model.signatures['serving_default']
            _ = self.infer(dummy_input)

        # 클래스 인덱스 로드
        self.class_indices = self.load_class_indices(class_indices_path)
        self.class_names = {v: k for k, v in self.class_indices.items()}
        self.target_user_name = self.get_user_name_from_ros_param()

        if not self.target_user_name:
            rospy.logerr("No user name provided. Classification cannot start.")
            raise ValueError("User name not set in ROS parameters.")

        rospy.Subscriber("/detection_result", DetectionResult, self.detection_callback)
        self.result_publisher = rospy.Publisher("/classified_image", DetectionResult, queue_size=5)
        self.classified_publisher = rospy.Publisher("/classified_result", ClassifiedResult, queue_size=5)

        self.camera_process = None
        self.start_camera_processor()

        rospy.loginfo("ClassifyNode started, subscribing to /detection_result")

    def load_class_indices(self, class_indices_path):
        """클래스 인덱스 로드"""
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            rospy.loginfo(f"Loaded class indices: {class_indices}")
            return class_indices
        except Exception as e:
            rospy.logerr(f"Failed to load class indices: {e}")
            return {}

    def get_user_name_from_ros_param(self):
        """ROS 파라미터에서 사용자 이름 가져오기"""
        if rospy.has_param('/current_user_name'):
            user_name = rospy.get_param('/current_user_name')
            if user_name in self.class_indices:
                rospy.loginfo(f"Selected user from ROS param: {user_name}")
                return user_name
            else:
                rospy.logerr(f"User name '{user_name}' not found in class indices.")
                return None
        else:
            rospy.logerr("ROS parameter '/current_user_name' is not set.")
            return None

    def start_camera_processor(self):
        """camera_processor_node 실행"""
        if self.camera_process is None:
            rospy.loginfo("Starting camera_processor_node...")
            self.camera_process = subprocess.Popen(["rosrun", "project_ojakdong", "camera_processor.py"])
        else:
            rospy.logwarn("camera_processor is already running.")

    def stop_camera_processor(self):
        """camera_processor_node 종료"""
        if self.camera_process is not None:
            rospy.loginfo("Stopping camera_processor...")
            self.camera_process.send_signal(signal.SIGINT)
            self.camera_process.wait()
            self.camera_process = None
        else:
            rospy.logwarn("camera_processor_node is not running.")

    def convert_model(self, h5_model_path, output_dir):
        model = tf.keras.models.load_model(h5_model_path, compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        model.save(saved_model_dir)
        
        conversion_params = trt.TrtConversionParams(
            precision_mode=trt.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=1 << 30
        )
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_dir,
            conversion_params=conversion_params
        )
        converter.convert()
        
        # TensorRT 엔진 빌드 및 저장
        def input_fn():
            yield tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32)
        
        rospy.loginfo("TensorRT 엔진을 사전 빌드 중입니다...")
        converter.build(input_fn=input_fn)
        converter.save(self.trt_saved_model_dir)
        rospy.loginfo(f"TensorRT 모델이 {self.trt_saved_model_dir}에 저장됨.")


    def detection_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")

            detection_result = DetectionResult()
            detection_result.image = msg.image
            classified_result = ClassifiedResult()

            for i, box in enumerate(msg.boxes):
                top_left = (int(box.points[0].x), int(box.points[0].y))
                bottom_right = (int(box.points[2].x), int(box.points[2].y))
                cropped_img = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                if cropped_img.size == 0:
                    rospy.logwarn(f"Skipped empty cropped image for box {i}")
                    continue

                # 이미지 전처리 및 예측
                resized_img = cv2.resize(cropped_img, (224, 224)) / 255.0
                resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32)  # float32로 변환
                predictions = self.infer(tf.constant(resized_img))

                # 예측 결과 추출
                prediction_scores = predictions['dense_1'].numpy()[0]
                class_name = max(self.class_indices, key=lambda k: prediction_scores[self.class_indices[k]])
                class_prob = prediction_scores[self.class_indices[class_name]]

                # 신뢰도 임계값 설정
                self.confidence_threshold = rospy.get_param('/confidence_threshold', 0.68)
                if class_prob < self.confidence_threshold:
                    class_name = "Other"

                # 클래스 ID와 이름 저장
                class_id = self.class_indices.get(class_name, -1)
                classified_result.custom_class_ids.append(class_id)
                classified_result.custom_class_names.append(class_name)

                # 바운딩 박스 색상 결정 (유저: 초록색, 비유저: 빨간색)
                if class_name == self.target_user_name:
                    color = (0, 255, 0)  # 초록색
                    classified_result.target_user_name = self.target_user_name
                else:
                    color = (0, 0, 255)  # 빨간색

                # 바운딩 박스 그리기
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
                label = f"{class_name}: {class_prob:.2f}"
                cv2.putText(frame, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # DetectionResult에 박스 정보 추가
                polygon = Polygon()
                polygon.points = [
                    Point32(x=top_left[0], y=top_left[1], z=0),
                    Point32(x=bottom_right[0], y=top_left[1], z=0),
                    Point32(x=bottom_right[0], y=bottom_right[1], z=0),
                    Point32(x=top_left[0], y=bottom_right[1], z=0)
                ]
                detection_result.boxes.append(polygon)
                detection_result.centers.append(Point32(
                    x=(top_left[0] + bottom_right[0]) / 2,
                    y=(top_left[1] + bottom_right[1]) / 2,
                    z=0
                ))
                detection_result.class_ids.append(msg.class_ids[i])
                detection_result.confidences.append(msg.confidences[i])

            # 결과 발행
            self.result_publisher.publish(detection_result)
            self.classified_publisher.publish(classified_result)
            
            # 화면에 결과 표시
            cv2.imshow("Classified Image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User terminated.")

        except Exception as e:
            rospy.logerr(f"Error during detection callback: {e}")

    def __del__(self):
        self.stop_camera_processor()

def main():
    rospy.init_node("classify_node")

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('project_ojakdong')
    mobilenet_model_path = os.path.join(package_path, 'model/mobilenetv2_finetuned.h5')
    class_indices_path = os.path.join(package_path, 'model/class_indices.json')
    output_dir = os.path.join(package_path, 'model')

    node = ClassifyNode(mobilenet_model_path, class_indices_path, output_dir)

    try:
        rospy.spin()
    finally:
        node.stop_camera_processor()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
