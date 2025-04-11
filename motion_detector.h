#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <opencv2/opencv.hpp>
#include  <yaml-cpp/yaml.h>
#include <string>
#include <vector>

class MotionDetector {
public:
    MotionDetector(const std::string& configFile);
    void run();

private:
    std::string video_src;
    int size;
    int seek;
    int mask_x_min;
    int mask_x_max;
    int mask_y_min;
    int mask_y_max;
    double res_ratio;
    double threshold;
    int angle_min;
    int angle_max;
    int binary_threshold;
    int threshold_count;
    bool show_cropped;
    double pyr_scale;
    int levels;
    int winsize;
    int iterations;
    int poly_n;
    double poly_sigma;

    const std::string WINDOW_NAME = "window";

    std::vector<std::vector<int>> directions_map;
    cv::Ptr<cv::BackgroundSubtractor> backSub;

    void loadConfig(const std::string& configFile);
    float calculateMode(const std::vector<float>& values);
    void roll(std::vector<std::vector<int>>& map);
    int calculateMaxMeanColumn(const std::vector<std::vector<int>>& map);
    void processFrame(cv::Mat& frame, cv::Mat& orig_frame, cv::Mat& gray_previous);
    void detectMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv);
};

#endif //MOTION_DETECTOR_H
