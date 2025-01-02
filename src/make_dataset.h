#ifndef MAKE_DATASET_H
#define MAKE_DATASET_H

#include <opencv2/opencv.hpp>
#include <string>
#include "DetectionProcessor.h"

// MakeDataset 클래스 정의
class MakeDataset {
public:
    MakeDataset(); // 생성자

    // 검출된 사람 이미지를 저장
    bool saveDetectedPerson(const DetectionProcessor& processor, const std::string& userName);

private:
    int imageIndex; // 저장된 이미지의 인덱스
};

#endif // MAKE_DATASET_H
