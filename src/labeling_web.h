#ifndef LABELING_WEB_H
#define LABELING_WEB_H

#include <string>

class LabelingToolWeb {
public:
    // 데이터셋을 train/val 디렉토리로 분류
    void organizeDataset(const std::string &baseDir, const std::string &userName);

    // YOLO 형식의 라벨 파일 생성
    void generateLabelFile(const std::string &labelPath, int classId, const std::string &imagePath);
};

#endif // LABELING_WEB_H
