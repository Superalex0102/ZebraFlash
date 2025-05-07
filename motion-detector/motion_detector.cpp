#include <iostream>
#include <opencv2/core/ocl.hpp>

#include "motion_detector.h"

#include <thread>

#include "../benchmark/benchmark.h"

//TODO: better gpu performance
//TODO: custom multithread implementation
//TODO: folyamatosan adogatva legyenek az adatok a GPU-nak, egy threadből, hogy ne sleepeljen

MotionDetector::MotionDetector(const std::string &configFile) {
    loadConfig(configFile);

    if (use_multi_thread) {
        thread_pool = std::make_unique<ThreadPool>(thread_amount == -1 ? std::thread::hardware_concurrency() : thread_amount);
    }

    // cv::setNumThreads(16);

    if (cv::ocl::haveOpenCL() && use_gpu) {
        cv::ocl::setUseOpenCL(true);
        if (cv::ocl::useOpenCL()) {
            std::cout << "Using GPU acceleration via OpenCL" << std::endl;
        } else {
            std::cout << "OpenCL is available, but not in use." << std::endl;
        }
    } else {
        cv::ocl::setUseOpenCL(false);
        std::cout << "GPU acceleration disabled (by confg or unavailable)." << std::endl;
    }

    directions_map.resize(size, std::vector<int>(4, 0));
    backSub = cv::createBackgroundSubtractorMOG2(500, 16, true);
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
    angle_up_min = config["angle_up_min"].as<int>();
    angle_up_max = config["angle_up_max"].as<int>();
    angle_down_min = config["angle_down_min"].as<int>();
    angle_down_max = config["angle_down_max"].as<int>();
    binary_threshold = config["binary_threshold"].as<int>();
    threshold_count = config["threshold_count"].as<int>();
    show_cropped = config["show_cropped"].as<bool>();
    pyr_scale = config["pyr_scale"].as<double>();
    levels = config["levels"].as<int>();
    winsize = config["winsize"].as<int>();
    iterations = config["iterations"].as<int>();
    poly_n = config["poly_n"].as<int>();
    poly_sigma = config["poly_sigma"].as<double>();
    debug = config["debug"].as<bool>();
    use_gpu = config["use_gpu"].as<bool>();
    use_multi_thread = config["use_multi_thread"].as<bool>();
    thread_amount = config["thread_amount"].as<int>();
}

float MotionDetector::calculateMode(const std::vector<float>& values) {
    if (values.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::unordered_map<float, int> frequency_map;
    for (const float& value : values) {
        int binned = static_cast<int>(std::round(value));
        frequency_map[binned]++;
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

float MotionDetector::detectMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::UMat& hsv) {
    int64 start_time = cv::getTickCount();

    cv::Mat flow(gray.size(), CV_32FC2);

    if (use_multi_thread) {
        int cols_per_thread = gray.cols / thread_amount;
        std::vector<std::future<void>> futures;
        std::vector<cv::Mat> flow_parts(thread_amount);

        for (int i = 0; i < thread_amount; i++) {
            int start_col = i * cols_per_thread;
            int end_col = (i == thread_amount - 1) ? gray.cols : (i + 1) * cols_per_thread;

            cv::Range col_range(start_col, end_col);
            cv::Mat gray_section = gray(cv::Range::all(), col_range);
            cv::Mat prev_section = gray_previous(cv::Range::all(), col_range);

            flow_parts[i] = cv::Mat(gray_section.rows, end_col - start_col, CV_32FC2);

            futures.push_back(thread_pool->enqueue([=, &flow_parts]() {
                cv::calcOpticalFlowFarneback(prev_section, gray_section, flow_parts[i],
                    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0);
            }));
        }

        for (auto& future : futures) {
            future.get();
        }

        for (int i = 0; i < thread_amount; i++) {
            int start_col = i * cols_per_thread;
            int end_col = (i == thread_amount - 1) ? gray.cols : (i + 1) * cols_per_thread;

            cv::Range col_range(start_col, end_col);
            flow_parts[i].copyTo(flow(cv::Range::all(), col_range));
        }
    } else {
        cv::calcOpticalFlowFarneback(gray_previous, gray, flow, pyr_scale, levels,
            winsize, iterations, poly_n, poly_sigma, 0);
    }

    int64 end_time = cv::getTickCount();
    double time_taken_ms = (end_time - start_time) * 1000.0 / cv::getTickFrequency();
    std::cout << "calcOpticalFlowFarneback took " << time_taken_ms << " ms" << std::endl;

    std::vector<cv::UMat> flow_channels(2);
    cv::split(flow, flow_channels);

    cv::Mat mag, ang;
    cv::cartToPolar(flow_channels[0], flow_channels[1], mag, ang, true);

    cv::Mat ang_180 = ang / 2;
    cv::Mat mask = mag > threshold;

    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(mask, non_zero_points);

    std::vector<float> move_sense;
    for (const auto& pt : non_zero_points) {
        move_sense.push_back(ang.at<float>(pt));
    }

    float move_mode = calculateMode(move_sense);
    bool is_moving_up =
        (move_mode >= angle_up_min && move_mode <= angle_up_max ||
        move_mode >= angle_down_min && move_mode <= angle_down_max);

    if (debug) {
        std::cout << move_mode << std::endl;
    }

    if (is_moving_up) {
        directions_map[directions_map.size() - 1][0] = 3.5f;
        directions_map[directions_map.size() - 1][1] = 0;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else if (move_mode < angle_up_min || angle_up_max < move_mode ||
        move_mode < angle_down_min || angle_down_max < move_mode) {
        directions_map[directions_map.size() - 1][0] = 0;
        directions_map[directions_map.size() - 1][1] = 1;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else { //No movement detected
        cv::UMat fg_mask;
        backSub->apply(frame, fg_mask);

        cv::UMat fg_mask_blurred;
        cv::GaussianBlur(fg_mask, fg_mask_blurred, cv::Size(7, 7), 0);

        cv::UMat tresh_frame;
        cv::threshold(fg_mask_blurred, tresh_frame, binary_threshold, 255, cv::THRESH_BINARY);

        if (debug) {
            cv::imshow("tresh_frame", tresh_frame);
        }

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
        hsv = cv::UMat(frame.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    }

    std::vector<cv::UMat> hsv_channels;
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

    return move_mode;
}

void MotionDetector::processFrame(cv::Mat& frame, cv::Mat& orig_frame, cv::Mat& gray_previous) {
    frame = frame(cv::Range(mask_x_min, mask_x_max), cv::Range(mask_y_min, mask_y_max));

    cv::UMat frame_resized;
    cv::Size res(static_cast<int>(frame.cols * res_ratio), static_cast<int>(frame.rows * res_ratio));
    cv::resize(frame, frame_resized, res, 0, 0, cv::INTER_CUBIC);

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::UMat hsv(frame.size(), CV_8UC3, cv::Scalar(0, 255, 0));

    float move_mode = detectMotion(frame, gray, gray_previous, hsv);

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

    cv::UMat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

    int text_thinkness = 6;
    if (show_cropped) {
        text_thinkness = 2;
    }

    cv::putText(orig_frame, "Angle: " + std::to_string(static_cast<int>(move_mode)),
            cv::Point(30, 150), cv::FONT_HERSHEY_COMPLEX,
            frame.cols / 500.0, cv::Scalar(0, 0, 255), 6);

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

    Benchmark timer;
    std::vector<BenchmarkResult> results;
    int frame_index = 0;

    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

    mask_x_max = h - mask_x_max;
    mask_y_max = w - mask_y_max;

    cap.set(cv::CAP_PROP_POS_MSEC, seek);

    cv::UMat frame_previous;
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

        timer.start();
        processFrame(frame, orig_frame, gray_previous);
        double elapsed = timer.stop();

        cv::putText(orig_frame, "MS: " + std::to_string(elapsed), cv::Point(30, 200), cv::FONT_HERSHEY_COMPLEX,
                    frame.cols / 500.0, cv::Scalar(0, 255, 0), 3);

        results.push_back({
            frame_index++,
            use_gpu,
            elapsed
        });

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

    saveBenchmarkResults(use_gpu, results);

    cap.release();
    cv::destroyAllWindows();
}