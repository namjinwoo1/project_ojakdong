#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

class FaceDetection {
private:
    cv::dnn::Net net;
    double confidence_threshold;

public:
    FaceDetection(const std::string& model_path, double threshold = 0.5)
        : confidence_threshold(threshold) {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            throw std::runtime_error("Failed to load ONNX model.");
        }
    }

    std::vector<cv::Rect> detectFaces(const cv::Mat& frame, const std::string& cascade_path) {
        std::vector<cv::Rect> faces;
        cv::CascadeClassifier faceCascade(cascade_path);

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(100, 100));
        return faces;
    }

    std::vector<float> runInference(const cv::Mat& face) {
        cv::Mat blob = cv::dnn::blobFromImage(face, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();

        std::vector<float> result(output.begin<float>(), output.end<float>());
        return result;
    }
};

#endif // FACEDETECTION_H
