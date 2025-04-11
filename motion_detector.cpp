#include <iostream>

#include "motion_detector.h"

MotionDetector::MotionDetector(const std::string &configFile) {
    loadConfig(configFile);
    directions_map.resize(size, std::vector<int>(4, 0));
    backSub = cv::createBackgroundSubtractorMOG2();
}

void MotionDetector::loadConfig(const std::string &configFile) {
    YAML::Node config = YAML::LoadFile(configFile);

    video_src = config["video_src"].as<std::string>();
    size = config["size"].as<int>();
    seek = config["seek"].as<int>();
    mask_x_min = config["upper_margin"].as<int>();
    mask_x_max = config["bottom_margin"].as<int>();
    mask_y_min = config["left_margin"].as<int>();
    mask_y_max = config["right_margin"].as<int>();
    res_ratio = config["res_ratio"].as<double>();
    threshold = config["threshold"].as<double>();
    angle_min = config["angle_min"].as<int>();
    angle_max = config["angle_max"].as<int>();
    binary_threshold = config["binary_threshold"].as<int>();
    threshold_count = config["threshold_count"].as<int>();
    show_cropped = config["show_cropped"].as<bool>();
    pyr_scale = config["pyr_scale"].as<double>();
    levels = config["levels"].as<int>();
    winsize = config["winsize"].as<int>();
    iterations = config["iterations"].as<int>();
    poly_n = config["poly_n"].as<int>();
    poly_sigma = config["poly_sigma"].as<double>();
}

float MotionDetector::calculateMode(const std::vector<float>& values) {
    if (values.empty()) {
        std::cerr << "Error: No data available to calculate mode." << std::endl;
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::map<float, int> frequency_map;
    for (const float& value : values) {
        frequency_map[value]++;
    }

    float mode = values[0];
    int max_count = 0;
    for (const auto& pair : frequency_map) {
        if (pair.second > max_count) {
            mode = pair.first;
            max_count = pair.second;
        }
    }
    return mode;
}

void MotionDetector::roll(std::vector<std::vector<int>>& map) {
    if (map.empty()) {
        return;
    }

    std::vector<int> first_row = map[0];

    for (size_t i = 0; i < map.size() - 1; ++i) {
        map[i] = map[i + 1];
    }

    map[map.size() - 1] = first_row;
}

int MotionDetector::calculateMaxMeanColumn(const std::vector<std::vector<int>>& map) {
    if (map.empty() || map[0].empty()) return -1;

    size_t cols = map[0].size();
    std::vector<float> column_means(cols, 0.0f);

    for (size_t j = 0; j < cols; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < map.size(); ++i) {
            sum += map[i][j];
        }
        column_means[j] = sum / map.size();
    }

    return std::distance(column_means.begin(), std::max_element(column_means.begin(), column_means.end()));
}

void MotionDetector::detectMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv) {
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(gray_previous, gray, flow, pyr_scale, levels,
        winsize, iterations, poly_n, poly_sigma, cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    std::vector<cv::Mat> flow_channels(2);
    cv::split(flow, flow_channels);

    cv::Mat mag, ang;
    cv::cartToPolar(flow_channels[0], flow_channels[1], mag, ang, true);

    cv::Mat ang_180 = ang / 2;
    cv::Mat mask = mag > threshold;

    std::vector<float> move_sense;
    for (int i = 0; i < ang.rows; ++i) {
        for (int j = 0; j < ang.cols; ++j) {
            if (mask.at<uchar>(i, j)) {
                move_sense.push_back(ang.at<float>(i, j));
            }
        }
    }

    float move_mode = calculateMode(move_sense);
    bool is_moving_up = (move_mode >= angle_min && move_mode <= angle_max);

    std::cout << move_mode << std::endl;

    if (is_moving_up) {
        directions_map[directions_map.size() - 1][0] = 3.5f;
        directions_map[directions_map.size() - 1][1] = 0;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else if (move_mode < angle_min || angle_max < move_mode) {
        directions_map[directions_map.size() - 1][0] = 0;
        directions_map[directions_map.size() - 1][1] = 1;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else { //No movement detected
        cv::Mat fg_mask;
        backSub->apply(frame, fg_mask);

        cv::Mat fg_mask_blurred;
        cv::GaussianBlur(fg_mask, fg_mask_blurred, cv::Size(7, 7), 0);

        cv::Mat tresh_frame;
        cv::threshold(fg_mask_blurred, tresh_frame, binary_threshold, 255, cv::THRESH_BINARY);

        if (cv::countNonZero(tresh_frame) > threshold_count) {
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 1;
            directions_map[directions_map.size() - 1][3] = 0;
        }
        else { //No movement, no difference detected
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 1;
        }
    }

    roll(directions_map);

    if (hsv.empty() || hsv.type() != CV_8UC3) {
        hsv = cv::Mat(frame.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    }

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    if (hsv_channels.size() == 3) {
        if (ang_180.size() != hsv_channels[0].size()) {
            cv::resize(ang_180, ang_180, hsv_channels[0].size());
        }
        ang_180.convertTo(hsv_channels[0], hsv_channels[0].type());

        if (mag.size() != hsv_channels[2].size()) {
            cv::resize(mag, mag, hsv_channels[2].size());
        }
        cv::normalize(mag, hsv_channels[2], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::merge(hsv_channels, hsv);
    } else {
        std::cerr << "Error: hsv_channels does not have 3 elements!" << std::endl;
    }
}

void MotionDetector::processFrame(cv::Mat& frame, cv::Mat& orig_frame, cv::Mat& gray_previous) {
    frame = frame(cv::Range(mask_x_min, mask_x_max), cv::Range(mask_y_min, mask_y_max));

    cv::Mat frame_resized;
    cv::Size res(static_cast<int>(frame.cols * res_ratio), static_cast<int>(frame.rows * res_ratio));
    cv::resize(frame, frame_resized, res, 0, 0, cv::INTER_CUBIC);

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat hsv(frame.size(), CV_8UC3, cv::Scalar(0, 255, 0));

    detectMotion(frame, gray, gray_previous, hsv);

    int loc = calculateMaxMeanColumn(directions_map);

    std::string text;
    if (loc == 0) {
        text = "Moving up (LED ON!)";
    }
    else if (loc == 1) {
        text = "Other directions";
    }
    else if (loc == 2) {
        text = "Difference (LED ON!)";
    }
    else {
        text = "WAITING";
    }

    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

    int text_thinkness = 6;
    if (show_cropped) {
        text_thinkness = 2;
    }

    cv::putText(frame, text, cv::Point(30, 90), cv::FONT_HERSHEY_COMPLEX,
    frame.cols / 500.0, cv::Scalar(0, 0, 255), text_thinkness);

    cv::putText(orig_frame, text, cv::Point(30, 90), cv::FONT_HERSHEY_COMPLEX,
        orig_frame.cols / 500.0, cv::Scalar(0, 0, 255), text_thinkness);

    gray_previous = gray;
}

void MotionDetector::run() {
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::VideoCapture cap(video_src);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source: " << video_src << std::endl;
        return;
    }

    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

    mask_x_max = h - mask_x_max;
    mask_y_max = w - mask_y_max;

    cap.set(cv::CAP_PROP_POS_MSEC, seek);

    cv::Mat frame_previous;
    cap >> frame_previous;
    if (frame_previous.empty()) {
        std::cerr << "Error: Failed to grab first frame" << std::endl;
        return;
    }

    cv::Mat gray_previous;
    cv::cvtColor(frame_previous(cv::Range(mask_x_min, mask_x_max), cv::Range(mask_y_min, mask_y_max)),
                 gray_previous, cv::COLOR_BGR2GRAY);

    cv::Mat frame, orig_frame;
    bool grabbed;

    while (true) {
        grabbed = cap.read(frame);

        if (!grabbed || frame.empty()) {
            std::cerr << "Error: Failed to grab frame" << std::endl;
            break;
        }

        orig_frame = frame.clone();
        processFrame(frame, orig_frame, gray_previous);

        if (show_cropped) {
            cv::imshow(WINDOW_NAME, frame);
        }
        else {
            cv::rectangle(orig_frame, cv::Point(mask_y_min, mask_x_min), cv::Point(mask_y_max, mask_x_max),
                cv::Scalar(0, 255, 0), 3);
            cv::imshow(WINDOW_NAME, orig_frame);
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

//cv::setNumThreads(4);