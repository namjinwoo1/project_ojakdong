#ifndef DLIB_FACE_TRACKER_H
#define DLIB_FACE_TRACKER_H

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/dnn.h>
#include <opencv2/opencv.hpp>
#include <vector>

// 신경망 정의
template <typename SUBNET>
using fc_no_bias_128 = dlib::fc_no_bias<128, SUBNET>;

template <typename SUBNET>
using relu_affine = dlib::relu<dlib::affine<SUBNET>>;

using anet_type = dlib::loss_metric<relu_affine<fc_no_bias_128<dlib::avg_pool_everything<dlib::input_rgb_image>>>>;

class DlibFaceTracker {
private:
    dlib::frontal_face_detector detector;              // 얼굴 검출기
    dlib::shape_predictor shapePredictor;              // 랜드마크 예측기
    anet_type faceRecognizer;                    // 얼굴 임베딩 생성기

public:
    DlibFaceTracker(const std::string& shapeModelPath, const std::string& faceModelPath) {
        // 모델 로드
        detector = dlib::get_frontal_face_detector();
        dlib::deserialize(shapeModelPath) >> shapePredictor;
        std::cout << "Loading face recognition model from: " << faceModelPath << std::endl;
        dlib::deserialize(faceModelPath) >> faceRecognizer;
    }

    std::vector<cv::Rect> detectFaces(const cv::Mat& frame) {
        std::vector<cv::Rect> faces;

        // OpenCV -> dlib 변환
        dlib::cv_image<dlib::bgr_pixel> dlibFrame(frame);
        std::vector<dlib::rectangle> detectedFaces = detector(dlibFrame);

        // dlib -> OpenCV 변환
        for (const auto& face : detectedFaces) {
            faces.emplace_back(cv::Rect(cv::Point(face.left(), face.top()),
                                        cv::Point(face.right(), face.bottom())));
        }

        return faces;
    }

    dlib::matrix<float, 0, 1> extractFeatures(const cv::Mat& face) {
        // OpenCV -> dlib 변환
        dlib::cv_image<dlib::bgr_pixel> dlibFace(face);
        dlib::matrix<dlib::rgb_pixel> faceChip;
        dlib::extract_image_chip(dlibFace, dlib::get_face_chip_details(shapePredictor(dlibFace, detector(dlibFace)[0])), faceChip);

        // 얼굴 임베딩 생성
        return faceRecognizer(faceChip);
    }

    double computeSimilarity(const dlib::matrix<float, 0, 1>& features1, const dlib::matrix<float, 0, 1>& features2) {
        return dlib::length(features1 - features2); // 두 특징 벡터 간 유클리디안 거리 계산
    }
};

#endif // DLIB_FACE_TRACKER_H
