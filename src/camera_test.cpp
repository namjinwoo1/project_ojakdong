#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>

// 네임스페이스 설정
using namespace cv;
using namespace cv::face;
using namespace std;

// 전역 변수
Ptr<LBPHFaceRecognizer> faceRecognizer = LBPHFaceRecognizer::create();
CascadeClassifier faceCascade;
map<int, string> userDatabase;

// 얼굴 데이터 저장 파일
const string DATA_FILE = "user_data.yml";

// 사용자 이름과 ID를 저장하는 파일
const string USER_DB_FILE = "user_database.txt";

// 함수 선언
void loadUserDatabase();
void saveUserDatabase();
void registerUser(VideoCapture& camera);
void startRecording(VideoCapture& camera);

int main() {
    // 카메라와 Haar Cascade 초기화
    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cerr << "카메라를 열 수 없습니다!" << endl;
        return -1;
    }
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Haar Cascade 파일을 로드할 수 없습니다!" << endl;
        return -1;
    }

    // 사용자 데이터 로드
    loadUserDatabase();
    faceRecognizer->read(DATA_FILE);

    while (true) {
        cout << "1: 사용자 등록\n2: 녹화 시작\n0: 종료\n선택: ";
        int choice;
        cin >> choice;

        if (choice == 1) {
            registerUser(camera);
        } else if (choice == 2) {
            startRecording(camera);
        } else if (choice == 0) {
            break;
        } else {
            cout << "잘못된 입력입니다!" << endl;
        }
    }

    return 0;
}

// 사용자 데이터 로드
void loadUserDatabase() {
    ifstream file(USER_DB_FILE);
    if (file.is_open()) {
        int id;
        string name;
        while (file >> id >> name) {
            userDatabase[id] = name;
        }
        file.close();
    }
}

// 사용자 데이터 저장
void saveUserDatabase() {
    ofstream file(USER_DB_FILE);
    if (file.is_open()) {
        for (const auto& pair : userDatabase) {
            file << pair.first << " " << pair.second << endl;
        }
        file.close();
    }
}

// 사용자 등록
void registerUser(VideoCapture& camera) {
    cout << "사용자 이름 입력: ";
    string name;
    cin >> name;

    Mat frame, gray;
    vector<Mat> faces;
    vector<int> labels;

    cout << "얼굴 학습 중... (ESC 키를 눌러 종료)" << endl;
    while (true) {
        camera >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> detectedFaces;
        faceCascade.detectMultiScale(gray, detectedFaces);

        for (const auto& face : detectedFaces) {
            rectangle(frame, face, Scalar(255, 0, 0), 2);
            Mat faceROI = gray(face);
            resize(faceROI, faceROI, Size(200, 200));
            faces.push_back(faceROI);
            labels.push_back(userDatabase.size() + 1); // 새로운 사용자 ID
        }

        imshow("사용자 등록", frame);
        if (waitKey(30) == 27) break; // ESC 키
    }

    faceRecognizer->update(faces, labels);
    faceRecognizer->write(DATA_FILE);

    userDatabase[userDatabase.size() + 1] = name;
    saveUserDatabase();

    cout << "사용자 등록 완료!" << endl;
}

// 녹화 시작
void startRecording(VideoCapture& camera) {
    Mat frame, gray;
    cout << "녹화 중... (ESC 키를 눌러 종료)" << endl;

    while (true) {
        camera >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> detectedFaces;
        faceCascade.detectMultiScale(gray, detectedFaces);

        for (const auto& face : detectedFaces) {
            Mat faceROI = gray(face);
            resize(faceROI, faceROI, Size(200, 200));

            int label;
            double confidence;
            faceRecognizer->predict(faceROI, label, confidence);

            string text = (confidence < 50) ? userDatabase[label] : "Unknown";
            rectangle(frame, face, Scalar(0, 255, 0), 2);
            putText(frame, text, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
        }

        imshow("녹화 화면", frame);
        if (waitKey(30) == 27) break; // ESC 키
    }

    cout << "녹화 종료!" << endl;
}
