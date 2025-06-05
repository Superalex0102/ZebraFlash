#include <filesystem>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "motion-detector/motion_detector.h"

const std::string INPUT_FILE = "../../config/params_input_file.yml";

int main() {
    std::cout << "Current working directory: "
              << std::filesystem::current_path() << std::endl;
    try {
        MotionDetector detector(INPUT_FILE);
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}