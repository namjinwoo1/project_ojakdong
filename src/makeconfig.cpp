#include <ros/ros.h>
#include <ros/package.h>
#include <QString>
#include <QDir>
#include <iostream>
#include <fstream>
#include <QDirIterator>
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
    if (!baseDir.exists()) {
        ROS_ERROR_STREAM("Dataset folder does not exist: " << datasetDir.toStdString());
        return;
    }

    QStringList filters = {"*.jpg", "*.png", "*.jpeg", "*.bmp"};
    QStringList imageFiles;
    QDirIterator it(datasetDir, filters, QDir::Files, QDirIterator::Subdirectories);

    while (it.hasNext()) {
        imageFiles << it.next();
    }

    ROS_INFO_STREAM("Found " << imageFiles.size() << " image files in: " << datasetDir.toStdString());

    if (imageFiles.isEmpty()) {
        ROS_WARN_STREAM("No image files found in dataset directory: " << datasetDir.toStdString());
        return;
    }

    std::ofstream outFile(outputPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Unable to create dataset path file: " << outputPath.toStdString());
        return;
    }

    for (const QString& file : imageFiles) {
        outFile << file.toStdString() << "\n";
    }

    outFile.close();
    ROS_INFO_STREAM("Dataset path file creation complete: " << outputPath.toStdString());
}

// obj.names 파일 생성
void generateObjNames(const QString& objNamesPath, const QStringList& userClasses) {
    std::ofstream outFile(objNamesPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Unable to create obj.names file: " << objNamesPath.toStdString());
        return;
    }

    outFile << "person\n";  // 기본 클래스
    for (const QString& userClass : userClasses) {
        outFile << userClass.toStdString() << "\n";
    }

    outFile.close();
    ROS_INFO_STREAM("obj.names file creation complete: " << objNamesPath.toStdString());
}

// obj.data 파일 생성
void generateObjData(const QString& objDataPath, const QString& objNamesPath, const QString& trainPath, const QString& valPath, const QString& backupPath) {
    std::ofstream outFile(objDataPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Unable to create obj.data file: " << objDataPath.toStdString());
        return;
    }

    std::ifstream objNamesFile(objNamesPath.toStdString());
    int numClasses = 0;
    std::string line;
    while (std::getline(objNamesFile, line)) {
        if (!line.empty()) ++numClasses;
    }
    objNamesFile.close();

    outFile << "classes=" << numClasses << "\n";
    outFile << "train=" << trainPath.toStdString() << "\n";
    outFile << "valid=" << valPath.toStdString() << "\n";
    outFile << "names=" << objNamesPath.toStdString() << "\n";
    outFile << "backup=" << backupPath.toStdString() << "\n";

    outFile.close();
    ROS_INFO_STREAM("obj.data file creation complete: " << objDataPath.toStdString());
}

// JSON 및 경로 파일 생성
void generateJSONConfig(const QString& trainDir, const QString& jsonPath, const QString& configTemplatePath, const QString& outputConfigPath, const QString& packagePath) {
    QDir baseDir(trainDir);
    if (!baseDir.exists()) {
        ROS_ERROR_STREAM("The learning folder does not exist: " << trainDir.toStdString());
        return;
    }

    QStringList userClasses = baseDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);

    if (userClasses.isEmpty()) {
        ROS_ERROR_STREAM("There is no user data in the train folder.");
        return;
    }

    Json::Value root;
    Json::Value classArray(Json::arrayValue);

    classArray.append("person");  // 기본 클래스 추가
    for (const QString& userClass : userClasses) {
        classArray.append(userClass.toStdString());
    }

    int numClasses = userClasses.size() + 1;  // 기본 클래스 포함
    int filters = (numClasses + 5) * 3;

    root["classes"] = classArray;
    root["num_classes"] = numClasses;
    root["filters_per_yolo_layer"] = filters;

    std::ofstream outFile(jsonPath.toStdString());
    if (!outFile.is_open()) {
        ROS_ERROR_STREAM("Unable to save JSON file: " << jsonPath.toStdString());
        return;
    }

    outFile << root.toStyledString();
    outFile.close();

    updateYOLOConfig(configTemplatePath, outputConfigPath, numClasses, filters);

    QString trainPathFile = packagePath + "/dataset/train/train.txt";
    QString valPathFile = packagePath + "/dataset/val/val.txt";
    QString objNamesPath = packagePath + "/dataset/obj.names";
    QString objDataPath = packagePath + "/dataset/obj.data";
    QString backupDir = packagePath + "/backup";

    generateDatasetPaths(trainDir, trainPathFile);
    generateDatasetPaths(QDir(trainDir).absolutePath().replace("train", "val"), valPathFile);
    generateObjNames(objNamesPath, userClasses);
    generateObjData(objDataPath, objNamesPath, trainPathFile, valPathFile, backupDir);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "make_config_node");
    ros::NodeHandle nh;

    std::string packagePath = ros::package::getPath("project_ojakdong");
    if (packagePath.empty()) {
        ROS_ERROR("Package path not found: project_ojakdong");
        return -1;
    }

    std::string trainDir = packagePath + "/dataset/train";
    std::string jsonPath = packagePath + "/model/config.json";
    std::string configTemplatePath = packagePath + "/model/yolov4-tiny.cfg";
    std::string outputConfigPath = packagePath + "/model/yolov4-tiny_custom.cfg";

    generateJSONConfig(QString::fromStdString(trainDir),
                       QString::fromStdString(jsonPath),
                       QString::fromStdString(configTemplatePath),
                       QString::fromStdString(outputConfigPath),
                       QString::fromStdString(packagePath));

    ros::spinOnce();
    return 0;
}
