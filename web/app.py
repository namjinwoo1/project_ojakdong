import rospy
import rospkg
import os
import json
import signal
import threading
import time
import logging
from std_srvs.srv import Trigger
from flask import Flask, jsonify, request, render_template, Response
from project_ojakdong.msg import DetectionResult  # DetectionResult 메시지 임포트
from sensor_msgs.msg import Image  # 필터링 이미지 토픽용
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess

app = Flask(__name__)

# ROS 초기화
rospy.init_node('web_interface', anonymous=True)

# 서비스 클라이언트 초기화
rospy.wait_for_service('/start_camera')
rospy.wait_for_service('/stop_camera')
rospy.wait_for_service('/capture_image')
rospy.wait_for_service('/label_data')
rospy.wait_for_service('/start_finetuning')
rospy.wait_for_service('/start_classification')
rospy.wait_for_service('/start_filter_display')
rospy.wait_for_service('/stop_filter_display')


start_camera_srv = rospy.ServiceProxy('/start_camera', Trigger)
stop_camera_srv = rospy.ServiceProxy('/stop_camera', Trigger)
capture_image_srv = rospy.ServiceProxy('/capture_image', Trigger)
label_data_srv = rospy.ServiceProxy('/label_data', Trigger)
finetuning_srv = rospy.ServiceProxy('/start_finetuning', Trigger)
classification_srv = rospy.ServiceProxy('/start_classification', Trigger)
start_filter_display_srv = rospy.ServiceProxy('/start_filter_display', Trigger)
stop_filter_display_srv = rospy.ServiceProxy('/stop_filter_display', Trigger)

# OpenCV-ROS 변환용 Bridge
bridge = CvBridge()

rospack = rospkg.RosPack()
package_path = rospack.get_path('project_ojakdong')

# 글로벌 변수
latest_frame = None
camera_running = False  # 카메라 상태 추적
robot_control_process = None
filter_display_process = None
classification_running = False  # 분류 상태를 저장하는 전역 변수

def load_default_filter_config():
    default_config = {
        "brightness": 50,
        "contrast": 50,
        "blur": 0
    }
    config_dir = os.path.join(package_path, 'config')
    config_path = os.path.join(config_dir, 'filter_config.json')

    # 디렉터리 확인 및 생성
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(default_config, f)
    return default_config

# /detection_result 메시지 구독
def detection_result_callback(msg):
    global latest_frame, camera_running
    if not camera_running:
        return  # 카메라가 정지 상태면 업데이트 안 함
    try:
        # DetectionResult 메시지의 이미지를 OpenCV 형식으로 변환
        cv_image = bridge.imgmsg_to_cv2(msg.image, "bgr8")
        _, buffer = cv2.imencode('.jpg', cv_image)
        latest_frame = buffer.tobytes()
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")

rospy.Subscriber('/detection_result', DetectionResult, detection_result_callback)

# /filtered_image 토픽 구독
def filtered_image_callback(msg):
    global latest_frame
    try:
        # ROS 메시지에서 OpenCV 형식으로 변환
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        _, buffer = cv2.imencode('.jpg', cv_image)
        latest_frame = buffer.tobytes()
    except Exception as e:
        rospy.logerr(f"Error processing filtered image: {e}")

rospy.Subscriber('/filtered_image', Image, filtered_image_callback)

# 기본 경로 설정
@app.route("/")
def index():
    return render_template("index.html")  # templates 디렉토리의 index.html 반환

# 카메라 시작
@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_running
    try:
        response = start_camera_srv()
        camera_running = response.success
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

# 카메라 중지
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running, latest_frame, filter_display_process
    try:
        response = stop_camera_srv()
        camera_running = False  # 카메라 정지 시 항상 False로 설정
        latest_frame = None  # 카메라 정지 시 프레임 초기화

        # 필터링 화면 출력 중지
        if filter_display_process and filter_display_process.poll() is None:
            filter_display_process.terminate()
            filter_display_process.wait()
            filter_display_process = None

        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})


# 이미지 캡처
@app.route('/capture_image', methods=['POST'])
def capture_image():
    try:
        # 클라이언트로부터 사용자 이름 전달받음
        user_name = request.json.get('userName', '').strip()
        if not user_name:
            return jsonify({"success": False, "message": "User name is required."})

        # 사용자 이름을 ROS로 전달하는 방식으로 확장 가능
        rospy.set_param('/current_user_name', user_name)
        response = capture_image_srv()
        return jsonify({"success": response.success, "message": f"{response.message} for user: {user_name}"})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

# 데이터 라벨링
@app.route('/label_data', methods=['POST'])
def label_data():
    try:
        response = label_data_srv()
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

# 파인튜닝 시작
@app.route('/start_finetuning', methods=['POST'])
def start_finetuning():
    try:
        response = finetuning_srv()
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

# 분류 시작
@app.route('/start_classification', methods=['POST'])
def start_classification():
    try:
        # 클라이언트로부터 사용자 이름 전달받음
        user_name = request.json.get('userName', '').strip()
        if not user_name:
            return jsonify({"success": False, "message": "User name is required."})

        # 사용자 이름을 ROS 파라미터로 설정
        rospy.set_param('/current_user_name', user_name)

        # 분류 시작 서비스 호출
        response = classification_srv()
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

@app.route('/check_classification_status', methods=['GET'])
def check_classification_status():
    global classification_running
    return jsonify({"running": classification_running})

# 비디오 피드 스트리밍 엔드포인트
@app.route('/video_feed')
def video_feed():
    global camera_running

    def generate():
        global latest_frame
        while True:
            if not camera_running:
                # 검은 화면 반환
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', black_frame)
                frame = buffer.tobytes()
            elif latest_frame is not None:
                frame = latest_frame
            else:
                # 기본 검은 화면 유지
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', black_frame)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            rospy.sleep(0.1)  # 10fps 제한

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_filter_config', methods=['GET'])
def get_filter_config():
    try:
        config_path = os.path.join(package_path, 'config/filter_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                settings = json.load(f)
        else:
            settings = load_default_filter_config()  # 기본 설정 불러오기

        return jsonify({"success": True, "settings": settings})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to load settings: {e}"})

@app.route('/update_filter_config', methods=['POST'])
def update_filter_config():
    try:
        config_path = os.path.join(package_path, 'config/filter_config.json')
        settings = request.json

        # JSON 값이 문자열로 저장되지 않도록 정수로 변환
        settings['brightness'] = int(settings['brightness'])
        settings['contrast'] = int(settings['contrast'])
        settings['blur'] = int(settings['blur'])

        with open(config_path, 'w') as f:
            json.dump(settings, f)

        rospy.wait_for_service('/reload_filter_config')
        reload_config_srv = rospy.ServiceProxy('/reload_filter_config', Trigger)
        response = reload_config_srv()

        if response.success:
            return jsonify({"success": True, "message": "Filter settings updated and reloaded successfully."})
        else:
            return jsonify({"success": False, "message": "Failed to reload filter settings."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to update settings: {e}"})
    
@app.route('/start_robot_control', methods=['POST'])
def start_robot_control():
    try:
        response = rospy.ServiceProxy('/start_robot_control', Trigger)()
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

@app.route('/stop_robot_control', methods=['POST'])
def stop_robot_control():
    try:
        response = rospy.ServiceProxy('/stop_robot_control', Trigger)()
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

@app.route('/start_filter_display', methods=['POST'])
def start_filter_display():
    try:
        response = rospy.ServiceProxy('/start_filter_display', Trigger)()
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})

@app.route('/stop_filter_display', methods=['POST'])
def stop_filter_display():
    try:
        response = rospy.ServiceProxy('/stop_filter_display', Trigger)()
        return jsonify({"success": response.success, "message": response.message})
    except rospy.ServiceException as e:
        return jsonify({"success": False, "message": f"Service call failed: {e}"})
    
@app.route('/filtered_image_feed')
def filtered_image_feed():
    def generate():
        global latest_frame
        while True:
            # latest_frame이 있으면 필터링된 이미지를 반환
            if latest_frame is not None:
                frame = latest_frame
            else:
                # latest_frame이 없으면 검은 화면 반환
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', black_frame)
                frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            rospy.sleep(0.1)  # 10fps 제한

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
