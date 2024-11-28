#ifndef MODELHANDLER_H
#define MODELHANDLER_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

class ModelHandler {
private:
    std::string model_path;
    std::vector<float> registered_features; // 등록된 사용자 특징

public:
    ModelHandler(const std::string& modelPath) : model_path(modelPath) {}

    bool isModelAvailable() const {
        return std::ifstream(model_path).good();
    }

    bool trainModel(const std::string& script_path) {
        std::string command = "python3 " + script_path;
        int result = system(command.c_str());
        return (result == 0);
    }

    void loadRegisteredFeatures(const std::string& featureFilePath) {
        registered_features.clear();

        std::ifstream file(featureFilePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open registered feature file: " + featureFilePath);
        }

        float value;
        while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
            registered_features.push_back(value);
        }

        file.close();
    }

    std::vector<float> getRegisteredFeatures() const {
        return registered_features;
    }
};

#endif // MODELHANDLER_H
