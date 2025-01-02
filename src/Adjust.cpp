#include "Adjust.hpp"
#include <jsoncpp/json/json.h>
#include <fstream>
#include <ros/package.h>
#include "project_ojakdong/DetectionResult.h"
#include "project_ojakdong/ClassifiedResult.h"  // ClassifiedResult 메시지 추가

// 생성자: 필터 기본값 로드 및 ROS 구독 설정
void Adjust::setBrightness(int brightness) {
    brightness_ = brightness;
}

void Adjust::setContrast(int contrast) {
    contrast_ = contrast;
}

void Adjust::setBlur(int blur) {
    blur_ = blur;
}

Adjust::Adjust(ros::NodeHandle& nh) : nh_(nh), brightness_(50), contrast_(50), blur_(0) {
    // JSON 설정 파일 로드
    loadFilterConfig();
    loadUserIndices();
    loadSelectedUser();  // 선택된 사용자 로드

    // ROS 설정: 이미지 및 Inception 결과 구독
    image_subscriber_ = nh_.subscribe("/classified_image", 1, &Adjust::detectionCallback, this);
    classified_subscriber_ = nh_.subscribe("/classified_result", 1, &Adjust::classifiedCallback, this);

    ROS_INFO("Adjust node initialized. Subscribing to /classified_image and /classified_result.");
}

// ROS 파라미터에서 선택된 사용자 로드
void Adjust::loadSelectedUser() {
    if (nh_.getParam("/current_user_name", selected_user_)) {
        ROS_INFO("Selected user from ROS param: %s", selected_user_.c_str());
    } else {
        ROS_WARN("No user name set. Applying blur to all.");
        selected_user_ = "";  
    }
}

// JSON 파일에서 사용자 인덱스 로드
void Adjust::loadUserIndices() {
    std::string json_path = ros::package::getPath("project_ojakdong") + "/model/class_indices.json";
    std::ifstream file(json_path);
    if (!file.is_open()) {
        ROS_ERROR("Failed to open class_indices.json at path: %s", json_path.c_str());
        return;
    }

    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(file, root)) {
        ROS_ERROR("Failed to parse class_indices.json.");
        file.close();
        return;
    }

    class_indices_.clear();
    for (const auto& key : root.getMemberNames()) {
        class_indices_[key] = root[key].asInt();
    }

    file.close();
    ROS_INFO("Loaded user indices from JSON file.");
}

// JSON 파일에서 필터 설정 로드
void Adjust::loadFilterConfig() {
    std::string config_path = ros::package::getPath("project_ojakdong") + "/config/filter_config.json";

    std::ifstream config_file(config_path, std::ifstream::binary);
    if (!config_file.is_open()) {
        ROS_WARN("Could not open filter_config.json. Using default settings.");
        brightness_ = 50;
        contrast_ = 50;
        blur_ = 0;
        return;
    }

    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(config_file, root)) {
        ROS_WARN("Failed to parse filter_config.json. Using default settings.");
        config_file.close();
        brightness_ = 50;
        contrast_ = 50;
        blur_ = 0;
        return;
    }

    try {
        brightness_ = root["brightness"].isInt() ? root["brightness"].asInt() : 50;
        contrast_ = root["contrast"].isInt() ? root["contrast"].asInt() : 50;
        blur_ = root["blur"].isInt() ? root["blur"].asInt() : 0;
    } catch (const Json::LogicError& e) {
        ROS_ERROR("Error parsing filter config: %s", e.what());
        brightness_ = 50;
        contrast_ = 50;
        blur_ = 0;
    }

    ROS_INFO("Filter configuration loaded: Brightness=%d, Contrast=%d, Blur=%d", brightness_, contrast_, blur_);
    config_file.close();
}

// /classified_result 콜백: Inception 결과 수신
void Adjust::classifiedCallback(const project_ojakdong::ClassifiedResult::ConstPtr& msg) {
    target_user_name_ = msg->target_user_name;  // Inception으로 분류된 사용자 이름 저장
    custom_class_ids_ = msg->custom_class_ids;  // 분류된 클래스 ID 저장
    // ROS_INFO("Received classified result: Target user = %s", target_user_name_.c_str());
}

// 콜백 함수: 구독된 이미지에 필터링 적용
void Adjust::detectionCallback(const project_ojakdong::DetectionResult::ConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
        cv::Mat input_frame = cvPtr->image;

        if (input_frame.empty()) {
            ROS_WARN("Received empty image.");
            return;
        }

        std::vector<cv::Rect> detected_boxes;
        std::vector<int> class_ids;

        for (const auto& box : msg->boxes) {
            int x = box.points[0].x;
            int y = box.points[0].y;
            int width = box.points[2].x - x;
            int height = box.points[2].y - y;
            detected_boxes.emplace_back(x, y, width, height);
        }
        class_ids = msg->class_ids;

        if (detected_boxes.size() != custom_class_ids_.size()) {
            ROS_ERROR("Mismatch between detected boxes and class IDs. Boxes: %lu, Class IDs: %lu",
                      detected_boxes.size(), custom_class_ids_.size());
            return;
        }

        // 필터링 적용
        cv::Mat output_frame;
        adjustFrame(input_frame, output_frame, detected_boxes, custom_class_ids_);

    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

// 프레임 조정: 선택된 사용자만 필터링에서 제외
void Adjust::adjustFrame(const cv::Mat& input_frame, cv::Mat& output_frame,
                         const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& class_ids) {
    double alpha = contrast_ / 50.0;
    int beta = brightness_ - 50;
    input_frame.convertTo(output_frame, -1, alpha, beta);

    // selected_user_에 해당하는 사용자 ID 추출
    int selected_user_id = -1;
    auto it = class_indices_.find(selected_user_);
    if (it != class_indices_.end()) {
        selected_user_id = it->second;
    }

    for (size_t i = 0; i < detected_boxes.size(); ++i) {
        cv::Rect box = detected_boxes[i] & cv::Rect(0, 0, input_frame.cols, input_frame.rows);
        
        // Inception 결과에서 사용자 매칭 (custom_class_ids_ 사용)
        int detected_user_id = custom_class_ids_[i];  

        if (detected_user_id == selected_user_id) {
            cv::rectangle(output_frame, box, cv::Scalar(0, 0, 255), 2);  // 빨간 박스 (선택된 사용자)
        } else {
            if (blur_ > 1) {
                int blur_value = (blur_ % 2 == 0) ? blur_ + 1 : blur_;
                applyBlur(output_frame, box, blur_value);
            }
            cv::rectangle(output_frame, box, cv::Scalar(255, 0, 0), 2);  // 파란 박스 (기타 사용자)
        }
    }
}

// 흐림 효과 적용 (applyBlur 함수 구현 추가)
void Adjust::applyBlur(cv::Mat& frame, cv::Rect area, int kernelSize) {
    area &= cv::Rect(0, 0, frame.cols, frame.rows);  // 이미지 범위 내로 조정

    if (kernelSize % 2 == 0) {
        kernelSize += 1;  // 커널 크기를 홀수로 맞춤
    }

    cv::Mat roi = frame(area);  // 관심 영역 추출
    cv::GaussianBlur(roi, roi, cv::Size(kernelSize, kernelSize), 0);
}
