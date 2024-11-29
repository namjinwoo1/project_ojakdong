#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include "checkuser.h"
#include <QInputDialog>
#include <QLineEdit>
#include <vector> // vector를 사용하기 위해 추가

using namespace cv;

class UserDetectionUI : public QWidget {
    Q_OBJECT

public:
    explicit UserDetectionUI(QWidget *parent = nullptr);

private slots:
    void startCamera();        // 카메라 시작
    void stopCamera();         // 카메라 중지
    void captureImage();       // 이미지 캡처
    void updateFrame();        // 카메라 프레임 업데이트

private:
    QString currentUserName; // 현재 사용자 이름
    bool isUserNameSet;      // 사용자 이름 설정 여부
    QLabel *cameraLabel;       // 카메라 화면 표시용 QLabel
    QPushButton *startButton;  // 카메라 시작 버튼
    QPushButton *stopButton;   // 카메라 중지 버튼
    QPushButton *captureButton;// 캡처 버튼
    QTimer *timer;             // 카메라 프레임 업데이트용 타이머
    VideoCapture cap;          // OpenCV VideoCapture 객체
    Mat frame;                 // 현재 프레임 저장
    UserDetector detector;     // 객체 검출 클래스
};

UserDetectionUI::UserDetectionUI(QWidget *parent)
    : QWidget(parent), timer(new QTimer(this)), isUserNameSet(false) {
    // UI 레이아웃 설정
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // 카메라 화면 표시용 QLabel
    cameraLabel = new QLabel(this);
    cameraLabel->setFixedSize(640, 480);
    cameraLabel->setStyleSheet("background-color: black;");
    mainLayout->addWidget(cameraLabel);

    // 버튼들
    startButton = new QPushButton("카메라 시작", this);
    stopButton = new QPushButton("카메라 중지", this);
    captureButton = new QPushButton("이미지 캡처", this);
    captureButton->setEnabled(false); // 초기에는 캡처 버튼 비활성화

    mainLayout->addWidget(startButton);
    mainLayout->addWidget(stopButton);
    mainLayout->addWidget(captureButton);

    // 버튼 클릭 이벤트 연결
    connect(startButton, &QPushButton::clicked, this, &UserDetectionUI::startCamera);
    connect(stopButton, &QPushButton::clicked, this, &UserDetectionUI::stopCamera);
    connect(captureButton, &QPushButton::clicked, this, &UserDetectionUI::captureImage);

    // 타이머와 프레임 업데이트 함수 연결
    connect(timer, &QTimer::timeout, this, &UserDetectionUI::updateFrame);
}

void UserDetectionUI::startCamera() {
    if (!cap.open(0)) {
        QMessageBox::warning(this, "오류", "카메라를 열 수 없습니다.");
        return;
    }
    captureButton->setEnabled(true); // 캡처 버튼 활성화
    timer->start(33); // 약 30 FPS
}

void UserDetectionUI::stopCamera() {
    timer->stop();
    cap.release();
    cameraLabel->clear();
    captureButton->setEnabled(false);

    // 사용자 이름 초기화
    currentUserName.clear();
    isUserNameSet = false;
}


void UserDetectionUI::captureImage() {
    if (frame.empty()) {
        QMessageBox::warning(this, "오류", "프레임이 비어 있습니다.");
        return;
    }

    // 사용자 이름이 설정되지 않은 경우 입력받기
    if (!isUserNameSet) {
        bool ok;
        QString userName = QInputDialog::getText(this, "사용자 이름 입력", "사용자 이름:", QLineEdit::Normal, "", &ok);

        if (!ok || userName.isEmpty()) {
            QMessageBox::warning(this, "오류", "사용자 이름을 입력하지 않았습니다.");
            return;
        }

        currentUserName = userName; // 사용자 이름 저장
        isUserNameSet = true;      // 이름 설정 완료
    }

    // 저장 폴더 경로 생성
    QString baseDir = QDir::currentPath() + "/dataset";
    QString userDir = baseDir + "/" + currentUserName;

    // 디렉토리 생성
    QDir().mkpath(userDir);

    // 검출된 객체 확인
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes = detector.detectObjects(frame, classIds, confidences);

    // 검출된 사람이 2명 이상인 경우 예외 처리
    if (boxes.size() > 1) {
        QMessageBox::warning(this, "오류", "검출된 사람이 2명 이상입니다. 캡처할 수 없습니다.");
        return;
    }

    if (boxes.empty()) {
        QMessageBox::warning(this, "오류", "검출된 사람이 없습니다.");
        return;
    }

    // 저장 경로를 std::string으로 변환
    std::string folderPath = userDir.toStdString();

    // 객체 검출 및 저장로
    detector.captureAndSave(frame, folderPath, currentUserName.toStdString());

}


void UserDetectionUI::updateFrame() {
    cap >> frame;
    if (frame.empty()) return;

    // 검출된 객체 표시를 위한 사각형 그리기
    Mat displayFrame = frame.clone(); // 화면 표시용 복제
    std::vector<int> classIds; // std::vector로 명시
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes = detector.detectObjects(displayFrame, classIds, confidences);

    for (const auto& box : boxes) {
        rectangle(displayFrame, box, Scalar(0, 255, 0), 2); // 초록색 사각형
        putText(displayFrame, "Person", Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    }

    // QLabel에 표시
    QImage qFrame = QImage(displayFrame.data, displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_RGB888).rgbSwapped();
    cameraLabel->setPixmap(QPixmap::fromImage(qFrame).scaled(cameraLabel->size(), Qt::KeepAspectRatio));
}

// main 함수
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    UserDetectionUI window;
    window.setWindowTitle("YOLOv4 사용자 검출");
    window.show();

    return app.exec();
}

#include "check_ui.moc"
