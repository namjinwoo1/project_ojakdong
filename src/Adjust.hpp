#ifndef ADJUST_HPP
#define ADJUST_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include "project_ojakdong/DetectionResult.h"
#include "project_ojakdong/ClassifiedResult.h"

class Adjust {
public:
    void setBrightness(int brightness);
    void setContrast(int contrast);
    void setBlur(int blur);

    Adjust(ros::NodeHandle& nh);

    void adjustFrame(const cv::Mat& input_frame, cv::Mat& output_frame,
                     const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& class_ids);

    void loadFilterConfig();
    void loadUserIndices();  // 누락된 부분 추가
    void loadSelectedUser(); // UI에서 선택된 사용자 불러오기

    void applyBlur(cv::Mat& frame, cv::Rect area, int kernelSize);
    void classifiedCallback(const project_ojakdong::ClassifiedResult::ConstPtr& msg);

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_subscriber_;
    ros::Publisher image_publisher_;
    ros::Subscriber classified_subscriber_;  // Inception 결과 구독자 추가


    int brightness_;
    int contrast_;
    int blur_;

    std::map<std::string, int> class_indices_; // 사용자 이름과 인덱스를 매핑
    std::string selected_user_;                // 선택된 사용자 이름
    std::string target_user_name_;  // Inception으로부터 분류된 사용자 이름 저장
    std::vector<int> custom_class_ids_;  // Inception 분류 결과 클래스 ID 저장
    

    void detectionCallback(const project_ojakdong::DetectionResult::ConstPtr& msg);

};

#endif // ADJUST_HPP
