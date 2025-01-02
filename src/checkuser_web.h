#ifndef CHECKUSER_WEB_H
#define CHECKUSER_WEB_H

#include <opencv2/opencv.hpp>

class UserDetectorWeb {
public:
    UserDetectorWeb();

    // 카메라 피드 업데이트 함수
    void updateCameraFeed(const cv::Mat& frame);

    // 최신 화면 반환
    cv::Mat getDisplayFrame() const;

private:
    cv::Mat displayFrame; // 화면 업데이트용 프레임
};

#endif // CHECKUSER_WEB_H
