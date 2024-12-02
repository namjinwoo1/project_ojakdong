#include <ros/ros.h>
#include <ros/package.h>
#include <QString>
#include <QDir>
#include <iostream>
#include <fstream>
#include <QDirIterator> // QDirIterator를 사용하기 위한 헤더 추가
#include <json/json.h>

// YOLO 설정 파일 수정 함수
void updateYOLOConfig(const QString& configTemplatePath, const QString& outputConfigPath, int numClasses, int filters) {
    std::ifstream inFile(configTemplatePath.toStdString());
    if (!inFile.is_open()) {
        ROS_ERROR_STREAM("Unable to read YOLO config file: " << configTemplatePath.toStdString());
        return;
    }

    std::ofstream outFile(outputConfigPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Unable to generate YOLO config file: " << outputConfigPath.toStdString());
        return;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        if (line.find("filters=") != std::string::npos) {
            outFile << "filters=" << filters << "\n";
        } else if (line.find("classes=") != std::string::npos) {
            outFile << "classes=" << numClasses << "\n";
        } else {
            outFile << line << "\n";
        }
    }

    inFile.close();
    outFile.close();
    ROS_INFO_STREAM("YOLO configuration file creation complete: " << outputConfigPath.toStdString());
}

// 데이터셋 경로 파일 생성 함수
void generateDatasetPaths(const QString& datasetDir, const QString& outputPath) {
    QDir baseDir(datasetDir);

    // 디렉토리 확인
    if (!baseDir.exists()) {
        ROS_ERROR_STREAM("Dataset folder does not exist: " << datasetDir.toStdString());
        return;
    }

    // 하위 디렉토리의 모든 이미지 파일 검색
    QStringList filters = {"*.jpg", "*.png", "*.jpeg", "*.bmp"};
    QStringList imageFiles;
    QDirIterator it(datasetDir, filters, QDir::Files, QDirIterator::Subdirectories);

    while (it.hasNext()) {
        imageFiles << it.next();
    }

    // 디버깅: 감지된 파일 수 확인
    ROS_INFO_STREAM("Found " << imageFiles.size() << " image files in: " << datasetDir.toStdString());

    if (imageFiles.isEmpty()) {
        ROS_WARN_STREAM("No image files found in dataset directory: " << datasetDir.toStdString());
        return;
    }

    // 파일 경로 출력
    for (const QString& file : imageFiles) {
        ROS_INFO_STREAM("Image file: " << file.toStdString());
    }

    // 파일 저장
    QString resolvedOutputPath = QDir::cleanPath(outputPath);
    std::ofstream outFile(resolvedOutputPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Unable to create dataset path file: " << resolvedOutputPath.toStdString());
        return;
    }

    for (const QString& file : imageFiles) {
        outFile << file.toStdString() << "\n";
    }

    outFile.close();
    ROS_INFO_STREAM("Dataset path file creation complete: " << resolvedOutputPath.toStdString());
}

void generateJSONConfig(const QString& trainDir, const QString& jsonPath, const QString& configTemplatePath, const QString& outputConfigPath, const QString& packagePath) {
    QDir baseDir(trainDir);
    if (!baseDir.exists()) {
        ROS_ERROR_STREAM("The learning folder does not exist: " << trainDir.toStdString());
        return;
    }

    QStringList classDirs = baseDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    if (classDirs.isEmpty()) {
        ROS_ERROR_STREAM("There is no class directory in the learning folder.");
        return;
    }

    Json::Value root;
    Json::Value classArray(Json::arrayValue);

    for (const QString& className : classDirs) {
        classArray.append(className.toStdString());
    }

    int numClasses = classDirs.size();
    int filters = 3 * (numClasses + 5);

    // JSON 정보 설정
    root["classes"] = classArray;
    root["num_classes"] = numClasses;
    root["filters_per_yolo_layer"] = filters;
    root["train_dir"] = trainDir.toStdString();
    root["val_dir"] = QDir(trainDir).absolutePath().replace("train", "val").toStdString();
    root["backup_dir"] = QDir::currentPath().toStdString() + "/backup";

    std::ofstream outFile(jsonPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Unable to save JSON file:" << jsonPath.toStdString());
        return;
    }

    outFile << root.toStyledString();
    outFile.close();
    ROS_INFO_STREAM("JSON configuration file creation complete: " << jsonPath.toStdString());

    // YOLO 설정 파일 업데이트
    updateYOLOConfig(configTemplatePath, outputConfigPath, numClasses, filters);

    // 데이터셋 경로 파일 생성
    QString trainPathFile = packagePath + "/dataset/train/train.txt";
    QString valPathFile = packagePath + "/dataset/val/val.txt";

    generateDatasetPaths(trainDir, trainPathFile);
    generateDatasetPaths(QDir(trainDir).absolutePath().replace("train", "val"), valPathFile);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "make_config_node");
    ros::NodeHandle nh;

    // ROS 패키지 경로 가져오기
    std::string packagePath = ros::package::getPath("project_ojakdong");
    if (packagePath.empty()) {
        ROS_ERROR("Package path not found: project_ojakdong");
        return -1;
    }

    // 패키지 경로 기반으로 경로 설정
    std::string trainDir = packagePath + "/dataset/train";
    std::string jsonPath = packagePath + "/model/config.json";
    std::string configTemplatePath = packagePath + "/model/yolov4.cfg";
    std::string outputConfigPath = packagePath + "/model/yolov4_custom.cfg";

    ROS_INFO_STREAM("TrainDir: " << trainDir);
    ROS_INFO_STREAM("JsonPath: " << jsonPath);
    ROS_INFO_STREAM("ConfigTemplatePath: " << configTemplatePath);
    ROS_INFO_STREAM("OutputConfigPath: " << outputConfigPath);

    // JSON 파일 생성 및 YOLO 설정 파일 수정
    generateJSONConfig(QString::fromStdString(trainDir),
                    QString::fromStdString(jsonPath),
                    QString::fromStdString(configTemplatePath),
                    QString::fromStdString(outputConfigPath),
                    QString::fromStdString(packagePath));

    ros::spinOnce();
    return 0;
}


