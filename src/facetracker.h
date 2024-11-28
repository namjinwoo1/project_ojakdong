#ifndef FACETRACKER_H
#define FACETRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <iostream>

class FaceTracker {
private:
    cv::dnn::Net net;               // DNN 모델
    double confidence_threshold;    // 신뢰도 임계값

public:
    FaceTracker(const std::string& onnx_model_path, const std::string& deploy_prototxt_path,
                const std::string& caffe_model_path, double threshold = 0.5)
        : confidence_threshold(threshold) {
        try {
            // ONNX 모델 로드 시도
            net = cv::dnn::readNetFromONNX(onnx_model_path);
            if (net.empty()) {
                throw std::runtime_error("ONNX model is empty.");
            }
            std::cout << "ONNX model loaded successfully from: " << onnx_model_path << std::endl;
        } catch (const cv::Exception& e) {
            // ONNX 모델 로드 실패 시 ResNet-SSD 로드
            std::cerr << "Warning: ONNX model loading failed: " << e.what() << std::endl;
            std::cerr << "Switching to ResNet-SSD model." << std::endl;

            try {
                net = cv::dnn::readNetFromCaffe(deploy_prototxt_path, caffe_model_path);
                if (net.empty()) {
                    throw std::runtime_error("ResNet-SSD model is empty.");
                }
                std::cout << "ResNet-SSD model loaded successfully from: "
                          << deploy_prototxt_path << " and " << caffe_model_path << std::endl;
            } catch (const cv::Exception& ex) {
                std::cerr << "Failed to load ResNet-SSD model: " << ex.what() << std::endl;
                net = cv::dnn::Net(); // 빈 네트워크로 초기화
            }
        }
    }

    std::vector<cv::Rect> detectFaces(const cv::Mat& frame) {
        std::vector<cv::Rect> faces;

        if (net.empty()) {
            std::cerr << "No valid model loaded. Skipping face detection." << std::endl;
            return faces; // 모델이 없으면 빈 리스트 반환
        }

        // 입력 블롭 생성
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300),
                                              cv::Scalar(104.0, 177.0, 123.0), false, false);
        net.setInput(blob);

        // 네트워크 추론
        cv::Mat detections = net.forward();

        // 결과 해석
        cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > confidence_threshold) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                faces.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
            }
        }

        return faces;
    }
};

#endif // FACETRACKER_H
