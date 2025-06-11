#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>

#include "../utils/algorithm.h"
#include "../thread-pool/thread_pool.h"



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
    int angle_up_min;
    int angle_up_max;
    int angle_down_min;
    int angle_down_max;
    int binary_threshold;
    int threshold_count;
    double pyr_scale;
    int levels;
    int winsize;
    int iterations;
    int poly_n;
    double poly_sigma;
    bool debug;
    bool use_gpu;
    bool use_multi_thread;
    int thread_amount;
    Algorithm algorithm;

    const std::string WINDOW_NAME = "window";

    std::vector<std::vector<int>> directions_map;
    cv::Ptr<cv::BackgroundSubtractor> backSub;

    std::unique_ptr<ThreadPool> thread_pool;

    void loadConfig(const std::string& configFile);
    void processFrame(cv::Mat& frame, cv::Mat& orig_frame, cv::Mat& gray_previous);
    float detectMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv);
    float detectOpticalFlowMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv);
    float detectYOLOMotion();
};

#endif //MOTION_DETECTOR_H
