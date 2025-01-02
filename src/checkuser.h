#ifndef CHECKUSER_H
#define CHECKUSER_H

#include <QObject> // QObject 포함
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <QString>
#include "project_ojakdong/DetectionResult.h" // DetectionResult 헤더 포함

class UserDetector : public QObject {  // QObject 상속
    Q_OBJECT  // Q_OBJECT 매크로 추가

private:
    int personClassId = 0;                 // "person" 클래스 ID (기본값 0으로 초기화)
    QString currentUserName;               // 현재 사용자 이름
    bool isUserNameSet = false;            // 사용자 이름 설정 여부
    int imageIndex = 0;                    // 저장된 이미지 인덱스

    cv::Mat latestFrame;                   // 최신 프레임
    std::vector<cv::Rect> latestBoxes;     // 최신 검출된 박스

public:
    // 생성자
    UserDetector();

    // DetectionResult 메시지 업데이트 함수
    void updateDetectionData(const project_ojakdong::DetectionResult::ConstPtr& msg);

    // 이미지 캡처 및 저장 함수 (성공 시 true 반환)
    bool captureAndSave();

    // 사용자 이름 초기화 함수
    void resetUserName();

    // 최신 프레임 반환 함수
    cv::Mat getLatestFrame() const;

signals:  // Qt 신호 선언
    void frameUpdated(const cv::Mat& frame); // QLabel에 업데이트할 프레임 신호
};

#endif // CHECKUSER_H
