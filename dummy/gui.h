#ifndef GUI_H
#define GUI_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <QImage>

class GUI : public QWidget {
    Q_OBJECT

public:
    explicit GUI(QWidget *parent = nullptr);

private slots:
    void startDetection();       // 촬영 버튼 눌렀을 때 실행
    void updateCameraFrame();    // 카메라에서 프레임 가져오기 및 YOLO 실행

private:
    QLabel *cameraLabel;
    QPushButton *captureButton;
    QPushButton *exitButton;

    QTimer *timer;              // 주기적으로 카메라 업데이트
    cv::VideoCapture cap;       // OpenCV 카메라 캡처
    cv::Mat frame;              // 현재 카메라 프레임
};

#endif // GUI_H
