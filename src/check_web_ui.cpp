#include <ros/ros.h>
#include <ros/package.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>
#include "DetectionProcessor.h"
#include "checkuser_web.h"
#include "make_dataset.h"
#include "labeling_web.h"
#include <geometry_msgs/Twist.h>


class WebInterface {
public:
    WebInterface() {
        ros::NodeHandle nh;

        cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

        // ROS 서비스 서버 등록
        startCameraService = nh.advertiseService("/start_camera", &WebInterface::startCameraCallback, this);
        stopCameraService = nh.advertiseService("/stop_camera", &WebInterface::stopCameraCallback, this);
        captureImageService = nh.advertiseService("/capture_image", &WebInterface::captureImageCallback, this);
        labelDataService = nh.advertiseService("/label_data", &WebInterface::labelDataCallback, this);
        configureDataService = nh.advertiseService("/configure_data", &WebInterface::configureDataCallback, this);
        finetuningService = nh.advertiseService("/start_finetuning", &WebInterface::startFinetuningCallback, this);
        classifyService = nh.advertiseService("/start_classification", &WebInterface::startClassificationCallback, this);
        startRobotControlService = nh.advertiseService("/start_robot_control", &WebInterface::startRobotControlCallback, this);
        stopRobotControlService = nh.advertiseService("/stop_robot_control", &WebInterface::stopRobotControlCallback, this);
        startFilterService = nh.advertiseService("/start_filter_display", &WebInterface::startFilterDisplayCallback, this);
        stopFilterService = nh.advertiseService("/stop_filter_display", &WebInterface::stopFilterDisplayCallback, this);

        detectionSubscriber = nh.subscribe("/detection_result", 1, &WebInterface::processDetectionResult, this);

        ROS_INFO("Web Interface Node Started");
    }

    ~WebInterface() {
        system("pkill -f camera_processor");
        ROS_INFO("Camera Processor Stopped");
    }

private:
    ros::ServiceServer startCameraService;
    ros::ServiceServer stopCameraService;
    ros::ServiceServer captureImageService;
    ros::ServiceServer labelDataService;
    ros::ServiceServer configureDataService;
    ros::ServiceServer finetuningService;
    ros::ServiceServer classifyService;
    ros::ServiceServer startRobotControlService; // 추가: 로봇 제어 서비스
    ros::ServiceServer stopRobotControlService;        // 추가: 로봇 정지 서비스
    ros::ServiceServer startFilterService;
    ros::ServiceServer stopFilterService;
    ros::Subscriber detectionSubscriber;
    ros::Subscriber filteredImageSubscriber;

    ros::Publisher cmd_vel_pub;
    
    
    DetectionProcessor detectionProcessor; 
    UserDetectorWeb userDetector;
    MakeDataset makeDataset;
    LabelingToolWeb labelingTool;

    // ROS 파라미터에서 사용자 이름 가져오기
    std::string getUserName() {
        if (!ros::param::has("/current_user_name")) {
            ROS_WARN("User name is not set in /current_user_name parameter.");
            return "";
        }
        std::string userName;
        ros::param::get("/current_user_name", userName);
        return userName;
    }

    bool startCameraCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        system("rosrun project_ojakdong camera_processor.py &");
        res.success = true;
        res.message = "Camera started.";
        ROS_INFO("Camera started.");
        return true;
    }

    bool stopCameraCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        system("pkill -f camera_processor");
        system("pkill -f classify");
        res.success = true;
        res.message = "Camera stopped.";
        ROS_INFO("Camera stopped.");
        return true;
    }

    bool startRobotControlCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Starting robot control...");
        int ret = system("roslaunch project_ojakdong p_control.launch &");
        if (ret == 0) {
            res.success = true;
            res.message = "Robot control started successfully.";
        } else {
            res.success = false;
            res.message = "Failed to start robot control.";
        }
        return true;
    }

    bool stopRobotControlCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Stopping robot control...");

        // 로봇 정지 명령 발행
        geometry_msgs::Twist stop_twist;
        stop_twist.linear.x = 0.0;
        stop_twist.angular.z = 0.0;

        ros::Rate rate(10);
        for (int i = 0; i < 10; ++i) {
            cmd_vel_pub.publish(stop_twist);
            rate.sleep();
        }
        // p_control 프로세스 종료
        int ret = system("pkill -f p_control.launch");
        if (ret == 0) {
            res.success = true;
            res.message = "Robot control stopped successfully.";
        } else {
            res.success = false;
            res.message = "Failed to stop robot control.";
        }
        ROS_INFO("Stop command sent to /cmd_vel.");

        return true;
    }


    bool captureImageCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Capture image service triggered.");

        std::string userName = getUserName();
        if (userName.empty()) {
            ROS_WARN("User name is not provided.");
            res.success = false;
            res.message = "User name is required for capturing an image.";
            return false;
        }

        if (makeDataset.saveDetectedPerson(detectionProcessor, userName)) {
            res.success = true;
            res.message = "Image captured and saved successfully.";
            ROS_INFO("Image captured and saved successfully for user: %s", userName.c_str());
        } else {
            res.success = false;
            res.message = "Failed to capture and save image.";
            ROS_WARN("Failed to capture and save image.");
        }

        return true;
    }

    bool labelDataCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Label data service triggered.");
        std::string userName = getUserName();
        if (userName.empty()) {
            ROS_WARN("User name is not provided.");
            res.success = false;
            res.message = "User name is required for labeling data.";
            return false;
        }

        std::string baseDir = ros::package::getPath("project_ojakdong") + "/dataset";

        try {
            labelingTool.organizeDataset(baseDir, userName);
            res.success = true;
            res.message = "Data labeling completed.";
            ROS_INFO("Data labeling for user '%s' completed.", userName.c_str());
        } catch (const std::exception &e) {
            ROS_ERROR("Data labeling failed: %s", e.what());
            res.success = false;
            res.message = "Data labeling failed.";
        }

        return true;
    }

    bool configureDataCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Configure data service triggered.");
        system("roslaunch project_ojakdong makeconfig.launch");
        res.success = true;
        res.message = "Training data configuration End.";
        return true;
    }

    bool startFinetuningCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Start finetuning service triggered.");
        system("roslaunch project_ojakdong finetuning.launch");
        res.success = true;
        res.message = "Fine-tuning End.";
        return true;
    }

    bool startClassificationCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Start classification service triggered.");
        system("roslaunch project_ojakdong classify.launch &");
        res.success = true;
        res.message = "Classification End.";
        return true;
    }

    void processDetectionResult(const project_ojakdong::DetectionResult::ConstPtr &msg) {
        detectionProcessor.processMessage(msg); 
        userDetector.updateCameraFeed(detectionProcessor.getFrame());
    }

        // filter_node 실행 콜백
    bool startFilterDisplayCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Starting filter display...");
        int ret = system("rosrun project_ojakdong filter_node &");
        if (ret == 0) {
            res.success = true;
            res.message = "Filter display started successfully.";
        } else {
            res.success = false;
            res.message = "Failed to start filter display.";
        }
        return true;
    }

    // filter_node 종료 콜백
    bool stopFilterDisplayCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
        ROS_INFO("Stopping filter display...");
        int ret = system("pkill -f filter_node");
        if (ret == 0) {
            res.success = true;
            res.message = "Filter display stopped successfully.";
        } else {
            res.success = false;
            res.message = "Failed to stop filter display.";
        }
        return true;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "web_interface_node");

    WebInterface webInterface;
    ros::spin();
    return 0;
}

