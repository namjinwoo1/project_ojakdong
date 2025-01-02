#include "make_dataset.h"
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <filesystem>

namespace fs = std::filesystem;

MakeDataset::MakeDataset() : imageIndex(0) {}

bool MakeDataset::saveDetectedPerson(const DetectionProcessor& processor, const std::string& userName) {
    // 최신 프레임과 경계 상자를 가져옴
    cv::Mat frame = processor.getFrame();
    std::vector<cv::Rect> boxes = processor.getBoxes();

    if (frame.empty()) {
        ROS_WARN("saveDetectedPerson(): Frame is empty.");
        return false;
    }

    if (boxes.empty()) {
        ROS_WARN("saveDetectedPerson(): No detected boxes available.");
        return false;
    }

    if (userName.empty()) {
        ROS_WARN("saveDetectedPerson(): User name is not provided.");
        return false;
    }

    // 사용자 디렉토리 생성
    std::string baseDir = ros::package::getPath("project_ojakdong") + "/dataset/" + userName;
    if (!fs::exists(baseDir)) {
        fs::create_directories(baseDir);
    }

    const auto& box = boxes[0]; // 첫 번째 박스를 사용
    int left = std::max(0, box.x + 1);
    int top = std::max(0, box.y + 1);
    int right = std::min(frame.cols - 1, box.x + box.width - 1);
    int bottom = std::min(frame.rows - 1, box.y + box.height - 1);

    cv::Rect adjustedBox(left, top, right - left, bottom - top);

    if (adjustedBox.width <= 0 || adjustedBox.height <= 0) {
        ROS_WARN("saveDetectedPerson(): Adjusted bounding box is invalid. Saving aborted.");
        return false;
    }

    // 크롭된 이미지 저장
    cv::Mat cropped = frame(adjustedBox);
    std::string fileName = baseDir + "/image_" + std::to_string(imageIndex++) + ".jpg";

    if (cv::imwrite(fileName, cropped)) {
        ROS_INFO("Image successfully saved: %s", fileName.c_str());
        return true;
    } else {
        ROS_WARN("Failed to save image: %s", fileName.c_str());
        return false;
    }
}
