#include "labeling.h"
#include <QDir>
#include <QMessageBox>
#include <QFile>
#include <iostream>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <random>

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

    QStringList images = userFolder.entryList(QStringList() << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp", QDir::Files);

    if (images.isEmpty()) {
        QMessageBox::warning(nullptr, "오류", userName + " 폴더에 이미지가 없습니다.");
        return;
    }

    // 이미지 리스트를 무작위로 섞기
    std::vector<QString> imageVector = images.toVector().toStdVector();
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(imageVector.begin(), imageVector.end(), g);

    // train과 val로 무작위 분류
    int totalImages = imageVector.size();
    int trainCount = static_cast<int>(totalImages * 0.8); // 80%를 train
    int valCount = totalImages - trainCount;             // 나머지를 val

    for (int i = 0; i < totalImages; ++i) {
        QString oldPath = userFolder.absoluteFilePath(imageVector[i]);
        QString newPath = (i < trainCount)
                              ? trainDir + "/" + imageVector[i]
                              : valDir + "/" + imageVector[i];

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

    QMessageBox::information(nullptr, "완료", userName + " 데이터셋이 무작위로 정리되었습니다.");
}

// 라벨 txt 파일 생성 함수
void LabelingTool::generateLabelFile(const QString &labelPath, int classId, const QString &imagePath) {
    cv::Mat image = cv::imread(imagePath.toStdString());
    if (image.empty()) {
        std::cerr << "이미지 로드 실패: " << imagePath.toStdString() << std::endl;
        return;
    }

    // YOLO 형식으로 라벨 생성
    std::ofstream outFile(labelPath.toStdString());
    if (!outFile.is_open()) {
        std::cerr << "라벨 파일 생성 실패: " << labelPath.toStdString() << std::endl;
        return;
    }

    // 검출된 객체를 YOLO 형식으로 저장
    outFile << classId << " "
            << 0.5 << " "  // x_center (임시)
            << 0.5 << " "  // y_center (임시)
            << 0.1 << " "  // width (임시)
            << 0.1 << std::endl; // height (임시)

    outFile.close();
    std::cout << "라벨 파일 생성 성공: " << labelPath.toStdString() << std::endl;
}
