#ifndef LABELING_H
#define LABELING_H

#include <QString>
#include <QDir>
#include <QMessageBox>

class LabelingTool {
public:
    explicit LabelingTool() = default; // 기본 생성자
    ~LabelingTool() = default;         // 기본 소멸자

    // 데이터셋 정리 메소드
    void organizeDataset(const QString &baseDir, const QString &userName); 
};

#endif // LABELING_H
