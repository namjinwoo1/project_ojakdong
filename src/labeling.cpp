#include "labeling.h"
#include <QDir>
#include <QMessageBox>
#include <iostream>

void LabelingTool::organizeDataset(const QString &baseDir, const QString &userName) {
    QDir datasetDir(baseDir);

    if (!datasetDir.exists()) {
        QMessageBox::warning(nullptr, "오류", "dataset 폴더가 존재하지 않습니다.");
        return;
    }

    QString userDir = datasetDir.absolutePath() + "/" + userName;

    // 사용자 폴더 확인
    QDir userFolder(userDir);
    if (!userFolder.exists()) {
        QMessageBox::warning(nullptr, "오류", userName + " 폴더가 존재하지 않습니다.");
        return;
    }

    // train, validation 디렉토리 생성
    QString trainDir = datasetDir.absolutePath() + "/train/" + userName;
    QString valDir = datasetDir.absolutePath() + "/val/" + userName;

    QDir().mkpath(trainDir);
    QDir().mkpath(valDir);

    QStringList images = userFolder.entryList(QStringList() << "*.jpg" << "*.png" << "*.bmp", QDir::Files);

    if (images.isEmpty()) {
        QMessageBox::warning(nullptr, "오류", userName + " 폴더에 이미지가 없습니다.");
        return;
    }

    // 이미지 파일을 train과 val로 분류
    int totalImages = images.size();
    int trainCount = static_cast<int>(totalImages * 0.8); // 80%를 train
    int valCount = totalImages - trainCount;             // 나머지를 val

    for (int i = 0; i < totalImages; ++i) {
        QString oldPath = userFolder.absoluteFilePath(images[i]);
        QString newPath = (i < trainCount)
                              ? trainDir + "/" + images[i]
                              : valDir + "/" + images[i];

        if (QFile::exists(newPath)) {
            std::cout << "이미 존재하는 파일: " << newPath.toStdString() << std::endl;
            continue;
        }

        if (QFile::rename(oldPath, newPath)) {
            std::cout << "파일 이동 성공: " << oldPath.toStdString() << " -> " << newPath.toStdString() << std::endl;
        } else {
            std::cerr << "파일 이동 실패: " << oldPath.toStdString() << std::endl;
        }
    }

    QMessageBox::information(nullptr, "완료", userName + " 데이터셋이 정리되었습니다.");
}
