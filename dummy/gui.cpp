#include "gui.h"
#include <QVBoxLayout>
#include <QApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ros/package.h>
#include <fstream>

using namespace cv;

// YOLOv4 설정
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;

// 클래스 이름 로드 함수
std::vector<std::string> loadClassNames(const std::string &namesPath) {
    std::vector<std::string> classes;
    std::ifstream ifs(namesPath);
    std::string line;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
    return classes;
}

QImage MatToQImage(const Mat &mat) {
    if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped();
    } else if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    } else {
        return QImage();
    }
}

GUI::GUI(QWidget *parent) : QWidget(parent), timer(new QTimer(this)) {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // QLabel로 기본 화면 표시
    cameraLabel = new QLabel(this);
    cameraLabel->setFixedSize(640, 480);
    cameraLabel->setStyleSheet("background-color: black;");
    mainLayout->addWidget(cameraLabel);

    // 촬영 버튼
    captureButton = new QPushButton("촬영", this);
    mainLayout->addWidget(captureButton);

    // 종료 버튼
    exitButton = new QPushButton("종료", this);
    mainLayout->addWidget(exitButton);

    // 종료 버튼 클릭 시 종료
    connect(exitButton, &QPushButton::clicked, qApp, &QApplication::quit);

    // 촬영 버튼 클릭 시 검출 시작
    connect(captureButton, &QPushButton::clicked, this, &GUI::startDetection);
}

void GUI::startDetection() {
    // 카메라 초기화
    if (!cap.open(0)) {
        qWarning("카메라를 열 수 없습니다.");
        return;
    }

    // QTimer 시작
    connect(timer, &QTimer::timeout, this, &GUI::updateCameraFrame);
    timer->start(33); // 약 30 FPS
}

// YOLO 추론 및 결과 처리
void GUI::updateCameraFrame() {
    cap >> frame;
    if (frame.empty()) {
        return;
    }

    // YOLOv4 모델 경로 설정
    std::string package_path = ros::package::getPath("project_ojakdong");
    const std::string CFG_PATH = package_path + "/model/yolov4.cfg";
    const std::string WEIGHTS_PATH = package_path + "/model/yolov4.weights";
    const std::string NAMES_PATH = package_path + "/model/coco.names";

    // 클래스 이름 로드
    static std::vector<std::string> classes = loadClassNames(NAMES_PATH);

    // "person" 클래스의 인덱스 가져오기
    auto it = std::find(classes.begin(), classes.end(), "person");
    if (it == classes.end()) {
        qWarning("클래스 파일에 'person' 클래스가 존재하지 않습니다.");
        return;
    }
    int personClassId = std::distance(classes.begin(), it);

    // YOLOv4 네트워크 초기화
    static dnn::Net net = dnn::readNetFromDarknet(CFG_PATH, WEIGHTS_PATH);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // YOLOv4 입력 처리
    Mat blob = dnn::blobFromImage(frame, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // 추론 실행
    std::vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // 결과 처리
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    for (const auto &output : outs) {
        for (int i = 0; i < output.rows; i++) {
            Mat scores = output.row(i).colRange(5, output.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            // confidence가 임계값보다 높고 "person" 클래스일 경우만 처리
            if (confidence > CONFIDENCE_THRESHOLD && classIdPoint.x == personClassId) {
                int centerX = (int)(output.at<float>(i, 0) * frame.cols);
                int centerY = (int)(output.at<float>(i, 1) * frame.rows);
                int width = (int)(output.at<float>(i, 2) * frame.cols);
                int height = (int)(output.at<float>(i, 3) * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // NMS 적용
    std::vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int i : indices) {
        Rect box = boxes[i];
        rectangle(frame, box, Scalar(0, 255, 0), 2); // 검출된 객체를 초록색 사각형으로 표시
    }

    // 결과를 QLabel에 표시
    QImage qFrame = MatToQImage(frame);
    cameraLabel->setPixmap(QPixmap::fromImage(qFrame).scaled(cameraLabel->size(), Qt::KeepAspectRatio));
}
