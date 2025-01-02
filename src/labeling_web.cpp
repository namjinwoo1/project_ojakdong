#include "labeling_web.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

void LabelingToolWeb::organizeDataset(const std::string &baseDir, const std::string &userName) {
    fs::path datasetPath(baseDir);

    if (!fs::exists(datasetPath)) {
        ROS_WARN("Dataset directory does not exist: %s", baseDir.c_str());
        return;
    }

    fs::path userPath = datasetPath / userName;

    // 사용자 폴더 확인
    if (!fs::exists(userPath)) {
        ROS_WARN("User folder does not exist: %s", userPath.string().c_str());
        return;
    }

    // train, validation 디렉토리 생성
    fs::path trainPath = datasetPath / "train" / userName;
    fs::path valPath = datasetPath / "val" / userName;

    fs::create_directories(trainPath);
    fs::create_directories(valPath);

    std::vector<fs::path> imagePaths;

    // 이미지 파일 검색
    for (const auto &entry : fs::directory_iterator(userPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                imagePaths.push_back(entry.path());
            }
        }
    }

    if (imagePaths.empty()) {
        ROS_WARN("No images found in user folder: %s", userPath.string().c_str());
        return;
    }

    // 이미지 리스트를 무작위로 섞기
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(imagePaths.begin(), imagePaths.end(), g);

    // train과 val로 무작위 분류
    size_t totalImages = imagePaths.size();
    size_t trainCount = static_cast<size_t>(totalImages * 0.8); // 80%를 train
    size_t valCount = totalImages - trainCount;               // 나머지를 val

    for (size_t i = 0; i < totalImages; ++i) {
        fs::path oldPath = imagePaths[i];
        fs::path newPath = (i < trainCount) ? (trainPath / oldPath.filename()) : (valPath / oldPath.filename());

        try {
            fs::rename(oldPath, newPath);
            ROS_INFO("Moved file: %s -> %s", oldPath.string().c_str(), newPath.string().c_str());
        } catch (const fs::filesystem_error &e) {
            ROS_WARN("Failed to move file: %s", e.what());
        }
    }

    ROS_INFO("Dataset organized successfully for user: %s", userName.c_str());
}

void LabelingToolWeb::generateLabelFile(const std::string &labelPath, int classId, const std::string &imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        ROS_WARN("Failed to load image: %s", imagePath.c_str());
        return;
    }

    // YOLO 형식으로 라벨 생성
    std::ofstream outFile(labelPath);
    if (!outFile.is_open()) {
        ROS_WARN("Failed to create label file: %s", labelPath.c_str());
        return;
    }

    // 검출된 객체를 YOLO 형식으로 저장
    outFile << classId << " "
            << 0.5 << " "  // x_center (임시)
            << 0.5 << " "  // y_center (임시)
            << 0.1 << " "  // width (임시)
            << 0.1 << std::endl; // height (임시)

    outFile.close();
    ROS_INFO("Label file created successfully: %s", labelPath.c_str());
}
