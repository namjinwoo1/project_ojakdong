#include "Adjust.hpp"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <project_ojakdong/DetectionResult.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <ros/package.h>
#include <std_srvs/Trigger.h>
#include <sys/stat.h>  // 파일 상태 확인 및 디렉터리 생성
#include <unistd.h>    // POSIX 시스템 콜 (mkdir)

class FilterNode {
public:
    FilterNode() : nh_("~"), adjust_(nh_) {
        image_publisher_ = nh_.advertise<sensor_msgs::Image>("/filtered_image", 1);
        detection_subscriber_ = nh_.subscribe("/detection_result", 1, &FilterNode::detectionCallback, this);
        filter_reload_service_ = nh_.advertiseService("/reload_filter_config", &FilterNode::reloadFilterConfig, this);
        recording_service_ = nh_.advertiseService("/start_recording", &FilterNode::startRecording, this);
        stop_recording_service_ = nh_.advertiseService("/stop_recording", &FilterNode::stopRecording, this);
        record_timer_ = nh_.createTimer(ros::Duration(1.0), &FilterNode::checkRecordingStatus, this);

        loadFilterConfig();

        ROS_INFO("FilterNode started, subscribing to /detection_result.");

        cv::Mat blank_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        sensor_msgs::ImagePtr blank_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", blank_frame).toImageMsg();
        image_publisher_.publish(blank_msg);
    }

    bool startRecording(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
        if (!recording_) {
            std::string video_dir = ros::package::getPath("project_ojakdong") + "/videos";
            struct stat info;
            if (stat(video_dir.c_str(), &info) != 0) {
                ROS_WARN("Video directory does not exist. Creating: %s", video_dir.c_str());
                if (mkdir(video_dir.c_str(), 0777) == -1) {
                    ROS_ERROR("Failed to create video directory.");
                    res.success = false;
                    res.message = "Failed to create video directory.";
                    return true;
                }
            }

            std::string video_path = video_dir + "/filtered_output_" + std::to_string(ros::Time::now().toSec()) + ".avi";
            int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            double fps = 10.0;
            video_writer_.open(video_path, fourcc, fps, cv::Size(640, 480));

            if (video_writer_.isOpened()) {
                recording_ = true;
                res.success = true;
                res.message = "Recording started.";
                ROS_INFO("Recording started, saving to %s", video_path.c_str());
            } else {
                ROS_ERROR("Failed to open video writer. Check codec and file path.");
                res.success = false;
                res.message = "Failed to open video writer.";
            }
        } else {
            res.success = false;
            res.message = "Already recording.";
        }
        return true;
    }

    bool stopRecording(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
        if (recording_) {
            video_writer_.release();
            recording_ = false;
            res.success = true;
            res.message = "Recording stopped.";
            ROS_INFO("Recording stopped.");
        } else {
            res.success = false;
            res.message = "No active recording to stop.";
            ROS_WARN("No active recording to stop.");
        }
        return true;
    }

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

    void detectionCallback(const project_ojakdong::DetectionResult::ConstPtr& msg) {
        try {
            // cv_bridge로 이미지 변환
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->image, "bgr8");
            if (!cv_ptr) {
                ROS_ERROR("cv_bridge conversion failed.");
                return;
            }

            cv::Mat input_frame = cv_ptr->image;
            if (input_frame.empty()) {
                ROS_ERROR("Received empty image frame.");
                return;
            }

            // 바운딩 박스 처리
            std::vector<cv::Rect> detected_boxes;
            std::vector<int> class_ids;
            for (const auto& box : msg->boxes) {
                cv::Rect rect(
                    cv::Point(box.points[0].x, box.points[0].y),
                    cv::Point(box.points[2].x, box.points[2].y)
                );
                if (rect.area() <= 0) {
                    ROS_WARN("Invalid bounding box detected. Skipping...");
                    continue;
                }
                detected_boxes.push_back(rect);
            }
            class_ids.assign(msg->class_ids.begin(), msg->class_ids.end());

            // 필터 적용
            cv::Mat filtered_frame;
            adjust_.adjustFrame(input_frame, filtered_frame, detected_boxes, class_ids);

            if (filtered_frame.empty()) {
                ROS_WARN("Filtered frame is empty. Skipping publishing.");
                return;
            }

            // 필터링된 이미지 퍼블리시
            sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(msg->image.header, "bgr8", filtered_frame).toImageMsg();
            image_publisher_.publish(output_msg);

            // 녹화 시작 (자동)
            last_filtered_image_time_ = ros::Time::now();
            if (!recording_) {
                std_srvs::Trigger srv;
                startRecording(srv.request, srv.response);
                ROS_INFO("Recording started automatically.");
            }

            // 녹화 중인 경우 비디오에 프레임 저장
            if (recording_) {
                if (!filtered_frame.empty()) {
                    video_writer_.write(filtered_frame);
                } else {
                    ROS_WARN("Filtered frame is empty. Skipping frame write.");
                }
            }
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge Exception: %s", e.what());
        } catch (std::exception& e) {
            ROS_ERROR("Standard Exception: %s", e.what());
        } catch (...) {
            ROS_ERROR("Unknown exception in detectionCallback.");
        }
    }


    void checkRecordingStatus(const ros::TimerEvent&) {
        if (recording_) {
            ros::Duration time_since_last_image = ros::Time::now() - last_filtered_image_time_;
            if (time_since_last_image.toSec() > 5.0) {
                std_srvs::Trigger srv;
                stopRecording(srv.request, srv.response);
            }
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher image_publisher_;
    ros::Subscriber detection_subscriber_;
    ros::ServiceServer filter_reload_service_;
    ros::ServiceServer recording_service_;
    ros::ServiceServer stop_recording_service_;
    // 마지막으로 필터링된 이미지의 시간을 기록하는 변수 추가
    ros::Time last_filtered_image_time_;
    ros::Timer record_timer_;
    Adjust adjust_;
    bool recording_ = false;
    cv::VideoWriter video_writer_;

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
