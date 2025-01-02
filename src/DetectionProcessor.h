#ifndef DETECTION_PROCESSOR_H
#define DETECTION_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "project_ojakdong/DetectionResult.h"

// DetectionProcessor 클래스
// Detection 메시지를 처리하고 최신 데이터를 관리
class DetectionProcessor {
public:
    // Detection 메시지를 처리하고 데이터를 업데이트
    void processMessage(const project_ojakdong::DetectionResult::ConstPtr& msg);

    // 최신 프레임을 반환 (프레임이 없는 경우 빈 Mat 반환)
    cv::Mat getFrame() const;

    // 최신 경계 상자 목록을 반환
    std::vector<cv::Rect> getBoxes() const;

private:
    cv::Mat latestFrame;                      // 최신 프레임 (Detection에서 처리된 이미지)
    std::vector<cv::Rect> latestBoxes;        // 최신 경계 상자 (Detected Objects)
};

#endif // DETECTION_PROCESSOR_H
