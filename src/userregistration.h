#ifndef USERREGISTRATION_H
#define USERREGISTRATION_H

#include <opencv2/opencv.hpp>
#include <QDir>
#include <string>
#include <vector>

class UserRegistration {
public:
    static std::string createUserDirectory(const std::string& base_path, const std::string& user_name) {
        std::string user_dir = base_path + "/" + user_name + "/";
        QDir().mkpath(QString::fromStdString(user_dir));
        return user_dir;
    }

    static bool saveFaceImages(const std::string& directory, const cv::Mat& face, int index) {
        std::string file_path = directory + "face_" + std::to_string(index) + ".jpg";
        return cv::imwrite(file_path, face);
    }
};

#endif // USERREGISTRATION_H
