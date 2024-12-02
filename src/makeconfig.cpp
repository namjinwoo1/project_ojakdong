#include <ros/ros.h>
#include <ros/package.h>
#include <QString>
#include <QDir>
#include <iostream>
#include <fstream>
#include <json/json.h> // JSON 라이브러리 필요

void generateJSONConfig(const QString& trainDir, const QString& jsonPath) {
    ROS_INFO_STREAM("Raw trainDir: " << trainDir.toStdString());
    ROS_INFO_STREAM("Raw jsonPath: " << jsonPath.toStdString());

    QString cleanTrainDir = QDir::cleanPath(trainDir);
    ROS_INFO_STREAM("Cleaned trainDir: " << cleanTrainDir.toStdString());

    QDir baseDir(cleanTrainDir);
    ROS_INFO_STREAM("BaseDir absolute path: " << baseDir.absolutePath().toStdString());

    if (!baseDir.exists()) {
        ROS_ERROR_STREAM("학습 폴더가 존재하지 않습니다: " << baseDir.absolutePath().toStdString());
        return;
    }

    QStringList classDirs = baseDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);

    if (classDirs.isEmpty()) {
        ROS_ERROR_STREAM("학습 폴더에 클래스 디렉토리가 없습니다.");
        return;
    }

    Json::Value root;
    Json::Value classArray(Json::arrayValue);

    for (const QString& className : classDirs) {
        classArray.append(className.toStdString());
    }

    int numClasses = classDirs.size();
    int filters = 3 * (numClasses + 5);

    // 명시적으로 val_dir 생성
    QString valDir = QDir::cleanPath(baseDir.absolutePath() + "/../val");

    root["classes"] = classArray;
    root["num_classes"] = numClasses;
    root["filters_per_yolo_layer"] = filters;
    root["train_dir"] = baseDir.absolutePath().toStdString();
    root["val_dir"] = valDir.toStdString();
    root["backup_dir"] = QDir::currentPath().toStdString() + "/backup";

    std::ofstream outFile(jsonPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Don't save JSON " << jsonPath.toStdString());
        return;
    }

    outFile << root.toStyledString();
    outFile.close();

    ROS_INFO_STREAM("Create JSON " << jsonPath.toStdString());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "make_config_node");
    ros::NodeHandle nh;

    // ROS 패키지 경로 동적 검색
    std::string packagePath = ros::package::getPath("project_ojakdong");
    if (packagePath.empty()) {
        ROS_ERROR("패키지 경로를 찾을 수 없습니다: project_ojakdong");
        return 1;
    }
    std::string trainDir = packagePath + "/dataset/train";
    std::string jsonPath = packagePath + "/config.json";

    ROS_INFO_STREAM("Using dynamically resolved TrainDir: " << trainDir);
    ROS_INFO_STREAM("Using dynamically resolved JsonPath: " << jsonPath);

    // JSON 파일 생성
    generateJSONConfig(QString::fromStdString(trainDir), QString::fromStdString(jsonPath));

    ros::spinOnce(); // 단일 실행이므로 spinOnce 사용
    return 0;
}

