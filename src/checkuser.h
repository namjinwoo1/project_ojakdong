#ifndef CHECKUSER_H
#define CHECKUSER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

class UserDetector {
private:
    cv::dnn::Net net;
    std::vector<std::string> classes;

    void loadClasses(const std::string& path);

public:
    UserDetector();
    std::vector<cv::Rect> detectObjects(const cv::Mat& frame, std::vector<int>& classIds, std::vector<float>& confidences);
    void startCamera(); // 실시간 카메라 화면 표시
    void captureAndSave(const cv::Mat& frame, const std::string& folderPath, const std::string& userName); // 사용자 이름 기반으로 이미지 저장
};

#endif
