#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QInputDialog>
#include <QLineEdit>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <QProcess>
#include "checkuser.h"
#include "labeling.h"
#include "check_ui.h"
#include "project_ojakdong/DetectionResult.h"
#include <cstdlib> 

using namespace cv;

UserDetectionUI::UserDetectionUI(QWidget *parent)
    : QWidget(parent), checkUser(new UserDetector()) { // UserDetector 포인터 초기화
    // UI 레이아웃 초기화
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    cameraLabel = new QLabel(this);
    cameraLabel->setFixedSize(640, 480);
    cameraLabel->setStyleSheet("background-color: black;");
    mainLayout->addWidget(cameraLabel);

    // 버튼 초기화
    startButton = new QPushButton("카메라 시작", this);
    stopButton = new QPushButton("카메라 중지", this);
    captureButton = new QPushButton("이미지 캡처", this);
    labelButton = new QPushButton("데이터 라벨링", this);
    configButton = new QPushButton("학습 데이터 설정", this);
    finetuningButton = new QPushButton("파인튜닝 시작", this);
    classifyButton = new QPushButton("분류 시작", this);

    captureButton->setEnabled(false);

    // UI 레이아웃 구성
    mainLayout->addWidget(cameraLabel);
    mainLayout->addWidget(startButton);
    mainLayout->addWidget(stopButton);
    mainLayout->addWidget(captureButton);
    mainLayout->addWidget(labelButton);
    mainLayout->addWidget(configButton);
    mainLayout->addWidget(finetuningButton);
    mainLayout->addWidget(classifyButton);

    // 버튼 클릭 이벤트 연결
    connect(startButton, &QPushButton::clicked, this, &UserDetectionUI::startCamera);
    connect(stopButton, &QPushButton::clicked, this, &UserDetectionUI::stopCamera);
    connect(captureButton, &QPushButton::clicked, this, &UserDetectionUI::captureImage);
    connect(labelButton, &QPushButton::clicked, this, &UserDetectionUI::createDataset);
    connect(configButton, &QPushButton::clicked, this, &UserDetectionUI::runMakeConfigLaunch);
    connect(finetuningButton, &QPushButton::clicked, this, &UserDetectionUI::runFinetuningLaunch);
    connect(classifyButton, &QPushButton::clicked, this, &UserDetectionUI::runClassifyLaunch);

    // ROS 메시지 처리를 위한 QTimer 초기화
    rosTimer = new QTimer(this);
    connect(rosTimer, &QTimer::timeout, this, &UserDetectionUI::spinOnceCallback);

    // UserDetector와 UI 연결
    connect(checkUser, &UserDetector::frameUpdated, this, &UserDetectionUI::updateFrame);
}

UserDetectionUI::~UserDetectionUI() {
    detectionSubscriber.shutdown(); // ROS 구독 종료
    rosTimer->stop();
    delete rosTimer;
    delete checkUser; // 메모리 해제
}

void UserDetectionUI::startCamera() {
    system("rosrun project_ojakdong camera_processor &");
    detectionSubscriber = nh.subscribe("/detection_result", 1, &UserDetectionUI::processDetectionResult, this);
    ROS_INFO("구독 시작: /detection_result");
    rosTimer->start(30); // 30ms 간격으로 spinOnce 호출
    captureButton->setEnabled(true);
}

void UserDetectionUI::stopCamera() {
    detectionSubscriber.shutdown();
    system("pkill -f camera_processor");
    rosTimer->stop();
    cameraLabel->clear();
    captureButton->setEnabled(false);
    QMessageBox::information(this, "알림", "카메라를 중지했습니다.");
}

void UserDetectionUI::spinOnceCallback() {
    ros::spinOnce(); // ROS 메시지 처리
}

void UserDetectionUI::processDetectionResult(const project_ojakdong::DetectionResult::ConstPtr &msg) {
    try {
        // ROS DetectionResult 메시지 처리
        checkUser->updateDetectionData(msg);

        // QLabel 업데이트
        if (!checkUser->getLatestFrame().empty()) {
            updateFrame(checkUser->getLatestFrame());
        }
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge 변환 실패: %s", e.what());
    }
}

void UserDetectionUI::updateFrame(const cv::Mat &processedFrame) {
    // OpenCV Mat을 QImage로 변환 후 QLabel에 표시
    cv::Mat rgbFrame;
    cvtColor(processedFrame, rgbFrame, COLOR_BGR2RGB);
    QImage qFrame(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);
    cameraLabel->setPixmap(QPixmap::fromImage(qFrame).scaled(cameraLabel->size(), Qt::KeepAspectRatio));
}

void UserDetectionUI::captureImage() {
    if (!checkUser->captureAndSave()) {
        QMessageBox::warning(this, "오류", "이미지를 캡처하거나 저장하는 데 실패했습니다.");
    }
}

void UserDetectionUI::createDataset() {
    std::string packagePath = ros::package::getPath("project_ojakdong");
    QString baseDir = QString::fromStdString(packagePath) + "/dataset";

    // 사용자 이름 입력받기
    bool ok;
    QString userName = QInputDialog::getText(this, "사용자 이름 입력", "데이터셋 구조를 생성할 사용자 이름:", QLineEdit::Normal, "", &ok);

    if (!ok || userName.isEmpty()) {
        QMessageBox::warning(this, "오류", "사용자 이름을 입력하지 않았습니다.");
        return;
    }

    // LabelingTool의 organizeDataset 메소드 호출
    LabelingTool labelingTool;
    labelingTool.organizeDataset(baseDir, userName);
}

void UserDetectionUI::runMakeConfigLaunch() {
    QString command = "roslaunch project_ojakdong makeconfig.launch";
    QProcess::startDetached(command);
    QMessageBox::information(this, "실행", "makeconfig.launch 파일이 실행되었습니다.");
}

void UserDetectionUI::runFinetuningLaunch() {
    QString command = "roslaunch project_ojakdong finetuning.launch";
    QProcess::startDetached(command);
    QMessageBox::information(this, "실행", "finetuning.launch 파일이 실행되었습니다.");
}

void UserDetectionUI::runClassifyLaunch() {
    QString command = "roslaunch project_ojakdong classify.launch";
    QProcess::startDetached(command);
    QMessageBox::information(this, "실행", "classify.launch 파일이 실행되었습니다.");
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "check_ui");
    QApplication app(argc, argv);

    UserDetectionUI window;
    window.setWindowTitle("YOLOv4 사용자 검출");
    window.show();

    return app.exec();
}
