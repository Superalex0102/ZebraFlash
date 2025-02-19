#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>

const std::string INPUT_FILE = "..\\config\\params_input_file.yml";
const std::string WINDOW_NAME = "window";

//https://github.com/huihut/OpenCV-MinGW-Build?tab=readme-ov-file

int main() {
    YAML::Node config = YAML::LoadFile(INPUT_FILE);

    if (!config["video_src"]) {
        std::cerr << "Error: 'video_src' key not found in YAML file!" << std::endl;
        return -1;
    }

    std::string video_src = config["video_src"].as<std::string>();
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

    cv::VideoCapture cap(video_src);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file!" << std::endl;
        return -1;
    }

    std::cout << "Video opened successfully." << std::endl;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow(WINDOW_NAME, frame);

        if (cv::waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}