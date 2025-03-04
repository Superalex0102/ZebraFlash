#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <numeric>

const std::string INPUT_FILE = "..\\config\\params_input_file.yml";
const std::string WINDOW_NAME = "window";

int main() {
    YAML::Node config = YAML::LoadFile(INPUT_FILE);

    std::string video_src = config["video_src"].as<std::string>();
    int size = config["size"].as<int>();
    int seek = config["seek"].as<int>();
    int mask_x_min = config["upper_margin"].as<int>();
    int mask_x_max = config["bottom_margin"].as<int>();
    int mask_y_min = config["left_margin"].as<int>();
    int mask_y_max = config["right_margin"].as<int>();
    double res_ratio = config["res_ratio"].as<double>();
    double threshold = config["threshold"].as<double>(); // changed to double
    int angle_min = config["angle_min"].as<int>();
    int angle_max = config["angle_max"].as<int>();
    int binary_threshold = config["binary_threshold"].as<int>();
    int threshold_count = config["threshold_count"].as<int>();
    bool show_cropped = config["show_cropped"].as<bool>();
    double pyr_scale = config["pyr_scale"].as<double>();
    int levels = config["levels"].as<int>();
    int winsize = config["winsize"].as<int>();
    int iterations = config["iterations"].as<int>();
    int poly_n = config["poly_n"].as<int>();
    double poly_sigma = config["poly_sigma"].as<double>();

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

    cv::VideoCapture cap(video_src);

    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

    cap.set(cv::CAP_PROP_POS_MSEC, seek);

    cv::Mat frame_previous;
    cap >> frame_previous;

    mask_x_max = h - mask_x_max;
    mask_y_max = w - mask_y_max;

    frame_previous = frame_previous(cv::Range(mask_x_min, mask_x_max), cv::Range(mask_y_min, mask_y_max));

    cv::Size res(w * res_ratio, h * res_ratio);
    cv::Mat frame_resized;
    cv::resize(frame_previous, frame_resized, res, 0, 0, cv::INTER_CUBIC);

    cv::Mat gray_previous;
    cv::cvtColor(frame_previous, gray_previous, cv::COLOR_BGR2GRAY);

    cv::Mat hsv(frame_previous.size(), CV_8UC3, cv::Scalar(0, 255, 0));

    cv::Ptr<cv::BackgroundSubtractor> backSub = cv::createBackgroundSubtractorMOG2();

    cv::Mat frame;
    bool grabbed;

    while (true) {
        grabbed = cap.read(frame);

        if (!grabbed) {
            std::cerr << "Error: Failed to grab frame" << std::endl;
            break;
        }

        cv::Mat orig_frame = frame;
        frame = frame(cv::Range(mask_x_min, mask_x_max), cv::Range(mask_y_min, mask_y_max));
        cv::resize(frame, frame_resized, res, 0, 0, cv::INTER_CUBIC);

        cv::imshow(WINDOW_NAME, frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}