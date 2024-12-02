#ifndef LABELING_H
#define LABELING_H

#include <QString>
#include <QDir>
#include <QMessageBox>
#include <map> // 클래스 ID 관리를 위한 헤더 추가

class LabelingTool {
public:
    explicit LabelingTool() = default; // 기본 생성자
    ~LabelingTool() = default;         // 기본 소멸자

    // 데이터셋 정리 메소드
    void organizeDataset(const QString &baseDir, const QString &userName);

private:
    // 라벨 파일 생성 메소드 (YOLO 형식)
    void generateLabelFile(const QString &labelPath, int classId, const QString &imagePath);

    // 클래스 ID 동적 관리
    std::map<QString, int> classMap; // 사용자 이름과 클래스 ID 매핑
    int currentClassId = 1;          // 다음에 할당될 클래스 ID
};

#endif // LABELING_H
