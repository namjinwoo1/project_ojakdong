#include "checkuser_web.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ros/ros.h>

// UserDetectorWeb 클래스의 생성자
UserDetectorWeb::UserDetectorWeb() : displayFrame(cv::Mat()) {}

// 카메라 피드 업데이트 함수
void UserDetectorWeb::updateCameraFeed(const cv::Mat& frame) {
    if (frame.empty()) {
        ROS_WARN("updateCameraFeed(): Provided frame is empty.");
        return;
    }
    // 화면 업데이트용 프레임 복사
    displayFrame = frame.clone();
}

// 최신 화면 반환
cv::Mat UserDetectorWeb::getDisplayFrame() const {
    if (displayFrame.empty()) {
        ROS_WARN("getDisplayFrame(): Display frame is empty.");
        return cv::Mat();
    }
    return displayFrame.clone();
}
