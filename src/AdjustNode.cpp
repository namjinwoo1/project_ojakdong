#include "Adjust.hpp"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <project_ojakdong/DetectionResult.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <ros/package.h>
#include <std_srvs/Trigger.h>

class FilterNode {
public:
    FilterNode() : nh_("~"), adjust_(nh_) {  // NodeHandle 전달
        // ROS 노드 초기화
        image_publisher_ = nh_.advertise<sensor_msgs::Image>("/filtered_image", 1);
        detection_subscriber_ = nh_.subscribe("/detection_result", 1, &FilterNode::detectionCallback, this);
        filter_reload_service_ = nh_.advertiseService("/reload_filter_config", &FilterNode::reloadFilterConfig, this);

        // 초기 필터 설정 로드
        loadFilterConfig();

        ROS_INFO("FilterNode started, subscribing to /detection_result and ready to reload filter settings.");

        cv::Mat blank_frame = cv::Mat::zeros(480, 640, CV_8UC3);  // 640x480 해상도, 검은 배경
        sensor_msgs::ImagePtr blank_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", blank_frame).toImageMsg();
        image_publisher_.publish(blank_msg);

        ROS_INFO("FilterNode initialized and blank image published.");
    }

    // /detection_result 메시지 콜백
    void detectionCallback(const project_ojakdong::DetectionResult::ConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->image, "bgr8");
            cv::Mat input_frame = cv_ptr->image;

            // 객체의 박스와 클래스 ID 추출
            std::vector<cv::Rect> detected_boxes;
            std::vector<int> class_ids;
            for (const auto& box : msg->boxes) {
                cv::Rect rect(
                    cv::Point(box.points[0].x, box.points[0].y),
                    cv::Point(box.points[2].x, box.points[2].y)
                );
                detected_boxes.push_back(rect);
            }
            class_ids.assign(msg->class_ids.begin(), msg->class_ids.end());

            // 필터링 적용
            cv::Mat filtered_frame;
            adjust_.adjustFrame(input_frame, filtered_frame, detected_boxes, class_ids);

            // 필터링된 이미지 발행
            sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(msg->image.header, "bgr8", filtered_frame).toImageMsg();
            image_publisher_.publish(output_msg);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge Exception: %s", e.what());
        }
    }

    // 필터 설정 재로드 서비스
    bool reloadFilterConfig(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
        if (loadFilterConfig()) {
            res.success = true;
            res.message = "Filter settings reloaded successfully.";
            ROS_INFO("Filter settings reloaded successfully.");
        } else {
            res.success = false;
            res.message = "Failed to reload filter settings.";
            ROS_ERROR("Failed to reload filter settings.");
        }
        return true;
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher image_publisher_;
    ros::Subscriber detection_subscriber_;
    ros::ServiceServer filter_reload_service_;
    Adjust adjust_;

    // JSON 파일에서 필터 설정 로드
    bool loadFilterConfig() {
        std::string config_path = ros::package::getPath("project_ojakdong") + "/config/filter_config.json";
        std::ifstream file(config_path);
        if (!file.is_open()) {
            ROS_ERROR("Failed to open filter_config.json at path: %s", config_path.c_str());
            return false;
        }

        Json::Value root;
        Json::Reader reader;
        if (!reader.parse(file, root)) {
            ROS_ERROR("Failed to parse filter_config.json.");
            file.close();
            return false;
        }

        try {
            adjust_.setBrightness(root["brightness"].asInt());
            adjust_.setContrast(root["contrast"].asInt());
            adjust_.setBlur(root["blur"].asInt());
            ROS_INFO("Filter settings loaded: Brightness=%d, Contrast=%d, Blur=%d",
                     root["brightness"].asInt(), root["contrast"].asInt(), root["blur"].asInt());
        } catch (const std::exception& e) {
            ROS_ERROR("Error applying filter settings: %s", e.what());
            return false;
        }

        file.close();
        return true;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "filter_node");

    FilterNode filter_node;

    ros::spin();
    return 0;
}
