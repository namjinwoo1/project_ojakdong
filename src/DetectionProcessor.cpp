#include "DetectionProcessor.h"
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

void DetectionProcessor::processMessage(const project_ojakdong::DetectionResult::ConstPtr& msg) {
    try {
        // 메시지에서 이미지를 OpenCV Mat으로 변환
        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
        cv::Mat frame = cvPtr->image;

        if (frame.empty()) {
            ROS_WARN("DetectionProcessor: Received frame is empty.");
            return;
        }

        // 박스 정보 추출
        std::vector<cv::Rect> boxes;
        for (const auto& polygon : msg->boxes) {
            if (polygon.points.size() < 4) {
                ROS_WARN("DetectionProcessor: Polygon has less than 4 points: %zu", polygon.points.size());
                continue;
            }

            const auto& topLeft = polygon.points[0];
            const auto& bottomRight = polygon.points[2];

            int left = std::max(0, static_cast<int>(topLeft.x));
            int top = std::max(0, static_cast<int>(topLeft.y));
            int right = std::min(frame.cols - 1, static_cast<int>(bottomRight.x));
            int bottom = std::min(frame.rows - 1, static_cast<int>(bottomRight.y));

            int width = std::max(1, right - left);
            int height = std::max(1, bottom - top);

            boxes.emplace_back(left, top, width, height);
        }

        // 최신 데이터 업데이트
        {
            latestFrame = frame.clone();
            latestBoxes = boxes;
        }


    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("DetectionProcessor: cv_bridge Exception: %s", e.what());
    } catch (const std::exception& e) {
        ROS_ERROR("DetectionProcessor: std::exception occurred: %s", e.what());
    } catch (...) {
        ROS_ERROR("DetectionProcessor: Unknown exception occurred.");
    }
}

cv::Mat DetectionProcessor::getFrame() const {
    return latestFrame.empty() ? cv::Mat() : latestFrame.clone(); // 최신 프레임 복사 후 반환
}

std::vector<cv::Rect> DetectionProcessor::getBoxes() const {
    return latestBoxes; // 최신 경계 상자 반환
}
