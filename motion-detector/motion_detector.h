#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>

#include "../thread-pool/thread_pool.h"

struct AppConfig {
    std::string video_src;
    int size;
    int seek;
    int seek_end;
    int row_start;
    int row_end;
    int col_start;
    int col_end;
    int angle_up_min;
    int angle_up_max;
    int angle_down_min;
    int angle_down_max;
    double res_ratio;
    double threshold;
    int binary_threshold;
    int threshold_count;
    double pyr_scale;
    int levels;
    int winsize;
    int iterations;
    int poly_n;
    double poly_sigma;
    int max_corners;
    double quality_level;
    int min_distance;
    bool debug;
    bool use_gpu;
    bool use_multi_thread;
    int thread_amount;
    std::string algorithm;
    std::string yolo_weights_path;
    std::string yolo_config_path;
    std::string yolo_classes_path;
    float yolo_confidence_threshold;
    float yolo_nms_threshold;
    int yolo_input_size;
};

class MotionDetector {
public:
    MotionDetector(const std::string& configFile);
    void run();

    AppConfig& getConfig();

private:
    AppConfig config_;
    const std::string WINDOW_NAME = "window";

    std::vector<std::vector<int>> directions_map;
    cv::Ptr<cv::BackgroundSubtractor> backSub;

    std::unique_ptr<ThreadPool> thread_pool;

    //YOLO fields, may need a separate file for YOLO
    cv::dnn::Net yolo_network;
    std::vector<std::string> class_names;
    std::vector<cv::Rect> previous_detections;
    bool yolo_initialized = false;

    void loadConfig(const std::string& configFile);
    void initializeParallelProcessing();
    bool processFrame(cv::Mat& frame, cv::Mat& orig_frame, cv::Mat& gray_previous);
    float detectMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv);
    float detectFarneOpticalFlowMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv);
    float detectLKOpticalFlowMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv);

    //YOLO methods
    bool initializeYOLO();
    float detectYOLOMotion(cv::Mat& frame);
    float calculateMotionFromDetections(const std::vector<cv::Rect>& current_detections);
    void updateDirectionsFromYOLO(float motion_magnitude, const std::vector<cv::Rect>& detections);
};

#endif //MOTION_DETECTOR_H
