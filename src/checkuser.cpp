#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "checkuser.h"
#include <ros/package.h>

using namespace cv;
using namespace dnn;
using namespace std;

UserDetector::UserDetector() {
    std::string package_path = ros::package::getPath("project_ojakdong");
    const string CONFIG_PATH = package_path + "/model/yolov4.cfg";
    const string WEIGHTS_PATH = package_path + "/model/yolov4.weights";
    const string CLASSES_PATH = package_path + "/model/coco.names";

    net = readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    loadClasses(CLASSES_PATH);
}

void UserDetector::loadClasses(const string& path) {
    ifstream ifs(path);
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }
}

vector<Rect> UserDetector::detectObjects(const Mat& frame, vector<int>& classIds, vector<float>& confidences) {
    Mat blob = blobFromImage(frame, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    vector<Rect> boxes;
    for (auto& output : outs) {
        for (int i = 0; i < output.rows; ++i) {
            Mat scores = output.row(i).colRange(5, output.cols);
            Point classIdPoint;
            double confidence;

            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > 0.5) { // CONFIDENCE_THRESHOLD
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

    vector<int> indices;
    NMSBoxes(boxes, confidences, 0.5, 0.4, indices); // NMS_THRESHOLD

    vector<Rect> filteredBoxes;
    for (int idx : indices) {
        if (classes[classIds[idx]] == "person") {
            filteredBoxes.push_back(boxes[idx]);
        }
    }

    return filteredBoxes;
}

void UserDetector::startCamera() {
    VideoCapture cap(0); // 카메라 연결
    if (!cap.isOpened()) {
        cerr << "카메라를 열 수 없습니다!" << endl;
        return;
    }

    namedWindow("Detection", WINDOW_NORMAL);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "카메라 프레임을 읽을 수 없습니다!" << endl;
            break;
        }

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes = detectObjects(frame, classIds, confidences);

        // 객체에 사각형 표시
        for (const auto& box : boxes) {
            rectangle(frame, box, Scalar(0, 255, 0), 2); // 초록색 사각형
            putText(frame, "Person", Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        // 화면에 표시
        imshow("Detection", frame);

        int key = waitKey(1);
        if (key == 27) { // ESC 키로 종료
            break;
        }
    }

    cap.release();
    destroyAllWindows();
}


void UserDetector::captureAndSave(const cv::Mat& frame, const std::string& folderPath, const std::string& userName) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes = detectObjects(frame, classIds, confidences);

    if (boxes.empty()) {
        cerr << "검출된 사람이 없습니다!" << endl;
        return;
    }

    // 이미지 저장
    static int count = 0; // 이미지 번호
    for (const auto& box : boxes) {
        Rect safeBox = box & Rect(0, 0, frame.cols, frame.rows); // ROI가 프레임 내부에 있도록 보장
        Mat cropped = frame(safeBox).clone();

        // 파일 경로 생성
        std::string filename = folderPath + "/" + userName + "_" + std::to_string(count++) + ".jpg";

        // 이미지 저장
        imwrite(filename, cropped);
        cout << "저장 완료: " << filename << endl;
    }
}




