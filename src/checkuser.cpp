#include "checkuser.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/Point32.h>
#include <QString>
#include <QDir>
#include <QMessageBox>
#include <QInputDialog>
#include <cv_bridge/cv_bridge.h>
#include "project_ojakdong/DetectionResult.h"

using namespace cv;
using namespace std;

UserDetector::UserDetector() : isUserNameSet(false), imageIndex(0), personClassId(0) {}

void UserDetector::updateDetectionData(const project_ojakdong::DetectionResult::ConstPtr& msg) {
    try {
        // 메시지에서 이미지를 OpenCV Mat으로 변환
        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
        cv::Mat frame = cvPtr->image;

        if (frame.empty()) {
            ROS_WARN("수신된 이미지가 비어 있습니다.");
            return;
        }

        vector<int> classIds(msg->class_ids.begin(), msg->class_ids.end());
        vector<float> confidences(msg->confidences.begin(), msg->confidences.end());
        vector<cv::Rect> boxes;

        // 박스 정보 추출 및 유효성 검사
        for (const auto& polygon : msg->boxes) {
            if (polygon.points.size() < 4) {
                ROS_WARN("Polygon의 points 크기가 4보다 작습니다: %zu", polygon.points.size());
                continue;
            }

            const auto& topLeft = polygon.points[0];
            const auto& bottomRight = polygon.points[2];  // 정사각형을 보장하는 두 점 사용

            // 좌표 보정: 음수 값이나 프레임 경계를 벗어나는 좌표를 강제로 클리핑
            int left = std::max(0, static_cast<int>(topLeft.x));
            int top = std::max(0, static_cast<int>(topLeft.y));
            int right = std::min(frame.cols - 1, static_cast<int>(bottomRight.x));
            int bottom = std::min(frame.rows - 1, static_cast<int>(bottomRight.y));

            int width = std::max(1, right - left);   // 최소 너비 보장
            int height = std::max(1, bottom - top);  // 최소 높이 보장

            // 디버깅용
            ROS_INFO("조정된 박스: left[%d], top[%d], width[%d], height[%d]", left, top, width, height);

            boxes.emplace_back(left, top, width, height);
        }

        vector<cv::Rect> personBoxes;
        for (size_t i = 0; i < classIds.size(); ++i) {
            if (classIds[i] == personClassId && confidences[i] > 0.5) {
                personBoxes.push_back(boxes[i]);
            }
        }

        // 최신 데이터 업데이트
        latestFrame = frame.clone();
        latestBoxes = personBoxes;

        // 업데이트 신호 발행
        if (!latestFrame.empty()) {
            emit frameUpdated(latestFrame);
        } else {
            ROS_WARN("최신 프레임이 비어 있습니다. 신호를 발행하지 않습니다.");
        }
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge Exception: %s", e.what());
    } catch (const std::exception& e) {
        ROS_ERROR("std::exception 발생: %s", e.what());
    } catch (...) {
        ROS_ERROR("알 수 없는 예외 발생");
    }
}

bool UserDetector::captureAndSave() {
    if (latestFrame.empty() || latestBoxes.empty()) {
        ROS_WARN("captureAndSave(): 현재 프레임이 비어있거나 검출된 객체가 없습니다.");
        QMessageBox::warning(nullptr, "오류", "현재 프레임이 비어있거나 검출된 객체가 없습니다.");
        return false;
    }

    if (!isUserNameSet) {
        bool ok;
        QString userName = QInputDialog::getText(nullptr, "사용자 이름 입력", "사용자 이름:", QLineEdit::Normal, "", &ok);
        if (ok && !userName.isEmpty()) {
            currentUserName = userName;
            isUserNameSet = true;
            imageIndex = 0;
        } else {
            QMessageBox::warning(nullptr, "오류", "사용자 이름이 설정되지 않았습니다.");
            return false;
        }
    }

    if (latestBoxes.size() != 1) {
        QMessageBox::warning(nullptr, "오류", "검출된 사람이 한 명이 아닙니다. 저장을 중단합니다.");
        return false;
    }

    QString baseDir = QString::fromStdString(ros::package::getPath("project_ojakdong")) + "/dataset/" + currentUserName;
    QDir().mkpath(baseDir);

    const auto& box = latestBoxes[0];

    // 박스 경계를 상하좌우 1픽셀씩 축소
    int left = std::max(0, box.x + 1);
    int top = std::max(0, box.y + 1);
    int right = std::min(latestFrame.cols - 1, box.x + box.width - 1);
    int bottom = std::min(latestFrame.rows - 1, box.y + box.height - 1);

    // 축소된 박스를 기반으로 안전한 ROI 설정
    cv::Rect adjustedBox(left, top, right - left, bottom - top);

    if (adjustedBox.width <= 0 || adjustedBox.height <= 0) {
        ROS_WARN("조정된 박스가 유효하지 않습니다. 저장 중단.");
        QMessageBox::warning(nullptr, "오류", "조정된 박스가 유효하지 않습니다.");
        return false;
    }

    cv::Mat cropped = latestFrame(adjustedBox);
    QString fileName = baseDir + QString("/image_%1.jpg").arg(imageIndex++);
    if (cv::imwrite(fileName.toStdString(), cropped)) {
        ROS_INFO("이미지 저장 성공: %s", fileName.toStdString().c_str());
        QMessageBox::information(nullptr, "성공", "이미지가 저장되었습니다.");
        return true;
    } else {
        ROS_WARN("이미지 저장 실패: %s", fileName.toStdString().c_str());
        QMessageBox::warning(nullptr, "오류", "이미지 저장에 실패했습니다.");
        return false;
    }
}

void UserDetector::resetUserName() {
    isUserNameSet = false;
    currentUserName.clear();
}

cv::Mat UserDetector::getLatestFrame() const {
    if (latestFrame.empty()) {
        ROS_WARN("getLatestFrame(): 최신 프레임이 비어 있습니다.");
        return cv::Mat();  // 빈 Mat 반환
    }
    return latestFrame.clone();  // 복사본 반환
}
