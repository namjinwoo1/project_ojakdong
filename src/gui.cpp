#include <QApplication>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>
#include <QTimer>
#include "facetracker.h"

class CameraControl : public QWidget {
    Q_OBJECT
private:
    cv::VideoCapture cap;
    QTimer* timer;
    QLabel* videoLabel;
    FaceTracker* faceTracker;

public:
    CameraControl(QWidget* parent = nullptr)
        : QWidget(parent), faceTracker(nullptr) {
        // 모델 경로 초기화
        std::string onnxModelPath = "/home/jinwoo/catkin_ws/src/project_ojakdong/src/onnx_model.onnx";
        std::string deployPath = "/home/jinwoo/catkin_ws/src/project_ojakdong/src/deploy.prototxt";
        std::string caffeModelPath = "/home/jinwoo/catkin_ws/src/project_ojakdong/src/res10_300x300_ssd_iter_140000_fp16.caffemodel";

        faceTracker = new FaceTracker(onnxModelPath, deployPath, caffeModelPath);

        // GUI 구성
        QVBoxLayout* layout = new QVBoxLayout(this);
        videoLabel = new QLabel(this);
        videoLabel->setFixedSize(640, 480);
        layout->addWidget(videoLabel);

        QPushButton* startButton = new QPushButton("카메라 시작", this);
        QPushButton* stopButton = new QPushButton("카메라 종료", this);
        layout->addWidget(startButton);
        layout->addWidget(stopButton);

        timer = new QTimer(this);
        connect(startButton, &QPushButton::clicked, this, &CameraControl::startCamera);
        connect(stopButton, &QPushButton::clicked, this, &CameraControl::stopCamera);
        connect(timer, &QTimer::timeout, this, &CameraControl::updateFrame);
    }

    ~CameraControl() {
        delete faceTracker;
        cap.release();
    }

private slots:
    void startCamera() {
        if (!cap.isOpened()) {
            cap.open(0); // 기본 카메라 열기
            if (!cap.isOpened()) {
                std::cerr << "카메라를 열 수 없습니다!" << std::endl;
                return;
            }
        }
        timer->start(30);
    }

    void stopCamera() {
        if (cap.isOpened()) {
            cap.release();
            videoLabel->clear();
        }
        timer->stop();
    }

    void displayFrame(const cv::Mat& frame) {
        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        QImage image(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);
        videoLabel->setPixmap(QPixmap::fromImage(image).scaled(videoLabel->size(),
                                                            Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    void updateFrame() {
        if (!cap.isOpened()) {
            return; // 카메라가 열려 있지 않으면 종료
        }

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            return; // 프레임이 비어 있으면 종료
        }

        auto faces = faceTracker->detectFaces(frame);
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2); // 얼굴 검출 사각형 표시
        }

        // 화면에 표시
        displayFrame(frame);
    }
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    CameraControl cameraControl;
    cameraControl.setWindowTitle("얼굴 검출");
    cameraControl.show();

    return app.exec();
}
#include "gui.moc"
