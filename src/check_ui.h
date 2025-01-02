#ifndef CHECK_UI_H
#define CHECK_UI_H

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "checkuser.h"
#include "project_ojakdong/DetectionResult.h"

class UserDetectionUI : public QWidget {
    Q_OBJECT

public:
    explicit UserDetectionUI(QWidget *parent = nullptr); // 생성자
    ~UserDetectionUI(); // 소멸자

private slots:
    void startCamera();             // 카메라 시작
    void stopCamera();              // 카메라 중지
    void captureImage();            // 이미지 캡처
    void spinOnceCallback();        // ROS 메시지 처리용 spinOnce 콜백

private:
    // ROS 관련 변수
    ros::NodeHandle nh;                          // ROS 노드 핸들
    ros::Subscriber detectionSubscriber;         // ROS DetectionResult 메시지 구독자

    // UI 관련 변수
    QLabel *cameraLabel;                         // 카메라 화면 QLabel
    QPushButton *startButton;                    // 카메라 시작 버튼
    QPushButton *stopButton;                     // 카메라 중지 버튼
    QPushButton *captureButton;                  // 이미지 캡처 버튼
    QPushButton *labelButton;                    // 데이터 라벨링 버튼
    QPushButton *configButton;                   // 학습 데이터 설정 버튼
    QPushButton *finetuningButton;               // 파인튜닝 시작 버튼
    QPushButton *classifyButton;                 // 분류 시작 버튼
    QTimer *rosTimer;                            // ROS 메시지 처리를 위한 QTimer

    // 데이터 처리 관련 변수
    UserDetector *checkUser;                     // 사용자 객체 검출 및 데이터 관리 클래스 포인터
    bool isUserNameSet;                          // 사용자 이름 설정 여부

    // 내부 함수
    void processDetectionResult(const project_ojakdong::DetectionResult::ConstPtr &msg); // DetectionResult 처리 함수
    void updateFrame(const cv::Mat &processedFrame);                                     // QLabel 업데이트 함수
    void createDataset();                                                               // 데이터셋 생성
    void runMakeConfigLaunch();                                                        // makeconfig.launch 실행
    void runFinetuningLaunch();                                                        // finetuning.launch 실행
    void runClassifyLaunch();                                                          // classify.launch 실행
};

#endif // CHECK_UI_H
