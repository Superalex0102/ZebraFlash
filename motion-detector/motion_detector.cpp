#include <iostream>
#include <opencv2/core/ocl.hpp>

#ifdef HAVE_CUDA
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "motion_detector.h"

#include <fstream>
#include <thread>

#include "../benchmark/benchmark.h"
#include "../utils/motion_utils.h"

MotionDetector::MotionDetector(const std::string &configFile) {
    loadConfig(configFile);

    directions_map.resize(config_.size, std::vector<int>(4, 0));
    backSub = cv::createBackgroundSubtractorMOG2(500, 16, true);
}

AppConfig& MotionDetector::getConfig() {
    return config_;
}

void MotionDetector::loadConfig(const std::string &configFile) {
    YAML::Node config = YAML::LoadFile(configFile);

    config_.video_src = config["video_src"].as<std::string>();
    config_.size = config["size"].as<int>();
    config_.seek = config["seek"].as<int>();
    config_.seek_end = config["seek_end"].as<int>();
    config_.row_start = config["upper_margin"].as<int>();
    config_.row_end = config["bottom_margin"].as<int>();
    config_.col_start = config["left_margin"].as<int>();
    config_.col_end = config["right_margin"].as<int>();
    config_.angle_up_min = config["angle_up_min"].as<int>();
    config_.angle_up_max = config["angle_up_max"].as<int>();
    config_.angle_down_min = config["angle_down_min"].as<int>();
    config_.angle_down_max = config["angle_down_max"].as<int>();
    config_.res_ratio = config["res_ratio"].as<double>();
    config_.threshold = config["threshold"].as<double>();
    config_.binary_threshold = config["binary_threshold"].as<int>();
    config_.threshold_count = config["threshold_count"].as<int>();
    config_.pyr_scale = config["pyr_scale"].as<double>();
    config_.levels = config["levels"].as<int>();
    config_.winsize = config["winsize"].as<int>();
    config_.iterations = config["iterations"].as<int>();
    config_.poly_n = config["poly_n"].as<int>();
    config_.poly_sigma = config["poly_sigma"].as<double>();
    config_.max_corners = config["max_corners"].as<int>();
    config_.quality_level = config["quality_level"].as<double>();
    config_.min_distance = config["min_distance"].as<int>();
    config_.debug = config["debug"].as<bool>();
    config_.use_gpu = config["use_gpu"].as<bool>();
    config_.use_multi_thread = config["use_multi_thread"].as<bool>();
    config_.thread_amount = config["thread_amount"].as<int>();
    config_.algorithm = config["algorithm"].as<std::string>();
    config_.yolo_weights_path = config["yolo_weights_path"].as<std::string>();
    config_.yolo_config_path = config["yolo_config_path"].as<std::string>();
    config_.yolo_classes_path = config["yolo_classes_path"].as<std::string>();
    config_.yolo_confidence_threshold = config["yolo_confidence_threshold"].as<float>();
    config_.yolo_nms_threshold = config["yolo_nms_threshold"].as<float>();
    config_.yolo_input_size = config["yolo_input_size"].as<int>();
}

void MotionDetector::initializeParallelProcessing() {
    if (config_.use_multi_thread) {
        config_.thread_amount = config_.thread_amount == -1 ? std::thread::hardware_concurrency() : config_.thread_amount;
        thread_pool = std::make_unique<ThreadPool>(config_.thread_amount);
    }

    if (cv::ocl::haveOpenCL() && config_.use_gpu) {
        cv::ocl::setUseOpenCL(true);
        if (cv::ocl::useOpenCL()) {
            std::cout << "Using GPU acceleration via OpenCL" << std::endl;
        } else {
            std::cout << "OpenCL is available, but not in use." << std::endl;
        }
    } else {
        cv::ocl::setUseOpenCL(false);
        std::cout << "GPU acceleration disabled (by config or unavailable)." << std::endl;
    }
}

float MotionDetector::detectMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv) {
    float result;
    if (config_.algorithm == "FARNE") {
        result = detectFarneOpticalFlowMotion(frame, gray, gray_previous, hsv);
    }
    else if (config_.algorithm == "LK") {
        result = detectLKOpticalFlowMotion(frame, gray, gray_previous, hsv);
    }
    else if (config_.algorithm == "YOLO") {
        result = detectYOLOMotion(frame);
    }

    return result;
}

static cv::cuda::GpuMat d_gray_previous, d_gray, d_flow;

float MotionDetector::detectFarneOpticalFlowMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv) {
    cv::Mat flow(gray.size(), CV_32FC2);
    cv::Mat mask, ang, ang_180, mag;

    if (config_.use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
#ifdef HAVE_CUDA
        try {
            cv::cuda::Stream stream;
            d_gray.upload(gray, stream);
            d_gray_previous.upload(gray_previous, stream);

            if (d_flow.size() != gray.size() || d_flow.type() != CV_32FC2) {
                d_flow.release();
                d_flow.create(gray.size(), CV_32FC2);
            }

            static auto farneback = cv::cuda::FarnebackOpticalFlow::create(
                config_.levels, config_.pyr_scale, false, config_.winsize, config_.iterations, config_.poly_n, config_.poly_sigma, 0);

            farneback->calc(d_gray_previous, d_gray, d_flow, stream);

            std::vector<cv::cuda::GpuMat> d_flow_channels(3);
            cv::cuda::split(d_flow, d_flow_channels, stream);

            cv::cuda::GpuMat d_mag, d_ang;
            cv::cuda::cartToPolar(d_flow_channels[0], d_flow_channels[1], d_mag, d_ang, true, stream);

            cv::cuda::GpuMat d_ang_180;
            cv::cuda::divide(d_ang, cv::Scalar(2.0), d_ang_180);

            cv::cuda::GpuMat d_mask;
            cv::cuda::threshold(d_mag, d_mask, config_.threshold, 255, cv::THRESH_BINARY, stream);

            d_mask.download(mask, stream);
            d_ang.download(ang, stream);
            d_ang_180.download(ang_180, stream);
            d_mag.download(mag, stream);
            stream.waitForCompletion();
        }
        catch (const cv::Exception& e) {
            std::cerr << "CUDA Optical Flow failed: " << e.what() << std::endl;
            return -1.0f;
        }
#endif
    } else if (config_.use_gpu && cv::ocl::haveOpenCL()) {
        try {
            cv::UMat u_gray_previous, u_gray, u_flow;
            cv::calcOpticalFlowFarneback(gray_previous, gray, flow, config_.pyr_scale,
                config_.levels, config_.winsize, config_.iterations,
                config_.poly_n, config_.poly_sigma, 0);

            flow.copyTo(u_flow);

            std::vector<cv::UMat> u_flow_channels(2);
            cv::split(u_flow, u_flow_channels);

            cv::UMat u_mag, u_ang;
            cv::cartToPolar(u_flow_channels[0], u_flow_channels[1], u_mag, u_ang, true);

            cv::UMat u_ang_180;
            cv::divide(u_ang, cv::Scalar(2.0), u_ang_180);

            cv::UMat u_mask;
            cv::threshold(u_mag, u_mask, config_.threshold, 255, cv::THRESH_BINARY);

            u_mask.copyTo(mask);
            u_ang.copyTo(ang);
            u_ang_180.copyTo(ang_180);
            u_mag.copyTo(mag);
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCL post-processing failed: " << e.what() << std::endl;
            return -1.0f;
        }
    } else if (config_.use_multi_thread) {
        int cols_per_thread = gray.cols / config_.thread_amount;
        std::vector<std::future<void>> futures;
        std::vector<cv::Mat> flow_parts(config_.thread_amount);

        for (int i = 0; i < config_.thread_amount; i++) {
            int start_col = i * cols_per_thread;
            int end_col = (i == config_.thread_amount - 1) ? gray.cols : (i + 1) * cols_per_thread;

            cv::Range col_range(start_col, end_col);
            cv::Mat gray_section = gray(cv::Range::all(), col_range);
            cv::Mat prev_section = gray_previous(cv::Range::all(), col_range);

            flow_parts[i] = cv::Mat(gray_section.rows, end_col - start_col, CV_32FC2);

            futures.push_back(thread_pool->enqueue([=, &flow_parts]() {
                cv::calcOpticalFlowFarneback(prev_section, gray_section, flow_parts[i],
                    config_.pyr_scale, config_.levels, config_.winsize, config_.iterations, config_.poly_n, config_.poly_sigma, 0);
            }));
        }

        for (auto& future : futures) {
            future.get();
        }

        for (int i = 0; i < config_.thread_amount; i++) {
            int start_col = i * cols_per_thread;
            int end_col = (i == config_.thread_amount - 1) ? gray.cols : (i + 1) * cols_per_thread;

            cv::Range col_range(start_col, end_col);
            flow_parts[i].copyTo(flow(cv::Range::all(), col_range));
        }

        std::vector<cv::Mat> flow_channels(2);
        cv::split(flow, flow_channels);

        cv::cartToPolar(flow_channels[0], flow_channels[1], mag, ang, true);

        ang_180 = ang / 2;
        mask = mag > config_.threshold;
    }
    else {
        cv::calcOpticalFlowFarneback(gray_previous, gray, flow, config_.pyr_scale, config_.levels,
            config_.winsize, config_.iterations, config_.poly_n, config_.poly_sigma, 0);

        //TODO: need to cleanup this code
        std::vector<cv::Mat> flow_channels(2);
        cv::split(flow, flow_channels);

        cv::cartToPolar(flow_channels[0], flow_channels[1], mag, ang, true);

        ang_180 = ang / 2;
        mask = mag > config_.threshold;
    }

    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(mask, non_zero_points);

    std::vector<float> move_sense;
    for (const auto& pt : non_zero_points) {
        move_sense.push_back(ang.at<float>(pt));
    }

    float move_mode = MotionUtils::calculateMode(move_sense);
    bool is_moving_up =
        MotionUtils::isAngleInRange(move_mode, config_.angle_up_min, config_.angle_up_max) ||
        MotionUtils::isAngleInRange(move_mode, config_.angle_down_min, config_.angle_down_max);

    if (config_.debug) {
        std::cout << move_mode << std::endl;
    }

    if (is_moving_up) {
        directions_map[directions_map.size() - 1][0] = 3.5f;
        directions_map[directions_map.size() - 1][1] = 0;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else if (move_mode < config_.angle_up_min || config_.angle_up_max < move_mode ||
        move_mode < config_.angle_down_min || config_.angle_down_max < move_mode) {
        directions_map[directions_map.size() - 1][0] = 0;
        directions_map[directions_map.size() - 1][1] = 1;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else { //No movement detected
        // cv::Mat fg_mask;
        // backSub->apply(frame, fg_mask);
        //
        // cv::Mat fg_mask_blurred;
        // cv::GaussianBlur(fg_mask, fg_mask_blurred, cv::Size(7, 7), 0);
        //
        // cv::Mat tresh_frame;
        // cv::threshold(fg_mask_blurred, tresh_frame, config_.binary_threshold, 255, cv::THRESH_BINARY);
        //
        // if (config_.debug) {
        //     cv::imshow("tresh_frame", tresh_frame);
        // }
        //
        // if (cv::countNonZero(tresh_frame) > config_.threshold_count) {
        //     directions_map[directions_map.size() - 1][0] = 0;
        //     directions_map[directions_map.size() - 1][1] = 0;
        //     directions_map[directions_map.size() - 1][2] = 1;
        //     directions_map[directions_map.size() - 1][3] = 0;
        // }
        { //No movement, no difference detected
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 1;
        }
    }

    MotionUtils::roll(directions_map);

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

    return move_mode;
}

float MotionDetector::detectLKOpticalFlowMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv) {
    std::vector<cv::Point2f> prev_pts, curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    if (config_.use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
#ifdef HAVE_CUDA
        cv::goodFeaturesToTrack(gray_previous, prev_pts, config_.max_corners, config_.quality_level, config_.min_distance);
        if (prev_pts.empty()) {
            // No features to track - set to WAITING status
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 1;
            MotionUtils::roll(directions_map);
            return -1.0f;
        }

        cv::cuda::GpuMat d_prev_pts(prev_pts);
        cv::cuda::GpuMat d_gray_prev(gray_previous);
        cv::cuda::GpuMat d_gray(gray);
        cv::cuda::GpuMat d_curr_pts, d_status, d_err;

        auto lk = cv::cuda::SparsePyrLKOpticalFlow::create();
        lk->calc(d_gray_prev, d_gray, d_prev_pts, d_curr_pts, d_status, d_err);

        cv::Mat h_curr_pts, h_status;
        d_curr_pts.download(h_curr_pts);
        d_status.download(h_status);

        curr_pts.resize(prev_pts.size());
        status.resize(prev_pts.size());

        for (size_t i = 0; i < prev_pts.size(); ++i) {
            if (h_status.at<uchar>(i)) {
                curr_pts[i] = h_curr_pts.at<cv::Point2f>(i);
                status[i] = 1;
            } else {
                status[i] = 0;
            }
        }
#endif
    } else if (config_.use_gpu && cv::ocl::haveOpenCL()) {
        cv::UMat u_gray_previous, u_gray;

        gray_previous.copyTo(u_gray_previous);
        gray.copyTo(u_gray);

        cv::goodFeaturesToTrack(u_gray_previous, prev_pts, config_.max_corners, config_.quality_level, config_.min_distance);

        if (prev_pts.empty()) {
            // No features to track - set to WAITING status
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 1;
            MotionUtils::roll(directions_map);
            return -1.0f;
        }
        cv::calcOpticalFlowPyrLK(u_gray_previous, u_gray, prev_pts, curr_pts, status, err);
    } else {
        cv::goodFeaturesToTrack(gray_previous, prev_pts, config_.max_corners, config_.quality_level, config_.min_distance);

        if (prev_pts.empty()) {
            // No features to track - set to WAITING status
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 1;
            MotionUtils::roll(directions_map);
            return -1.0f;
        }

        cv::calcOpticalFlowPyrLK(gray_previous, gray, prev_pts, curr_pts, status, err);
    }

    std::vector<float> move_sense;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            float dx = curr_pts[i].x - prev_pts[i].x;
            float dy = curr_pts[i].y - prev_pts[i].y;
            float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
            if (angle < 0) angle += 360.0f;
            float magnitude = std::sqrt(dx * dx + dy * dy);
            if (magnitude > config_.threshold)
                move_sense.push_back(angle);
        }
    }

    // if (move_sense.empty()) {
        // cv::Mat fg_mask;
        // backSub->apply(frame, fg_mask);
        //
        // cv::Mat blurred, tresh_frame;
        // cv::GaussianBlur(fg_mask, blurred, cv::Size(7, 7), 0);
        // cv::threshold(blurred, tresh_frame, config_.binary_threshold, 255, cv::THRESH_BINARY);
        //
        // if (config_.debug) {
        //     cv::imshow("LK threshold frame", tresh_frame);
        // }
        //
        // if (cv::countNonZero(tresh_frame) > config_.threshold_count) {
        //     // Difference detected
        //     directions_map[directions_map.size() - 1][0] = 0;
        //     directions_map[directions_map.size() - 1][1] = 0;
        //     directions_map[directions_map.size() - 1][2] = 1;
        //     directions_map[directions_map.size() - 1][3] = 0;
        // }
        // else {
        //     // No movement, no difference detected - WAITING
        //     directions_map[directions_map.size() - 1][0] = 0;
        //     directions_map[directions_map.size() - 1][1] = 0;
        //     directions_map[directions_map.size() - 1][2] = 0;
        //     directions_map[directions_map.size() - 1][3] = 1;
        // }
        //
        // MotionUtils::roll(directions_map);
        //
        // if (hsv.empty() || hsv.type() != CV_8UC3) {
        //     hsv = cv::Mat(frame.size(), CV_8UC3, cv::Scalar(0, 255, 0));
        // }
    //     return -1.0f;
    // }

    float move_mode = MotionUtils::calculateMode(move_sense);
    bool is_moving_up =
        MotionUtils::isAngleInRange(move_mode, config_.angle_up_min, config_.angle_up_max) ||
        MotionUtils::isAngleInRange(move_mode, config_.angle_down_min, config_.angle_down_max);

    if (config_.debug) {
        std::cout << "LK Mode: " << move_mode << std::endl;
    }

    if (is_moving_up) {
        directions_map[directions_map.size() - 1][0] = 3.5f;
        directions_map[directions_map.size() - 1][1] = 0;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else if (move_mode < config_.angle_up_min || config_.angle_up_max < move_mode ||
             move_mode < config_.angle_down_min || config_.angle_down_max < move_mode) {
        directions_map[directions_map.size() - 1][0] = 0;
        directions_map[directions_map.size() - 1][1] = 1;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 0;
    }
    else {
        // cv::Mat fg_mask;
        // backSub->apply(frame, fg_mask);
        //
        // cv::Mat blurred, tresh_frame;
        // cv::GaussianBlur(fg_mask, blurred, cv::Size(7, 7), 0);
        // cv::threshold(blurred, tresh_frame, config_.binary_threshold, 255, cv::THRESH_BINARY);
        //
        // if (config_.debug) {
        //     cv::imshow("LK threshold frame", tresh_frame);
        // }
        //
        // if (cv::countNonZero(tresh_frame) > config_.threshold_count) {
        //     directions_map[directions_map.size() - 1][0] = 0;
        //     directions_map[directions_map.size() - 1][1] = 0;
        //     directions_map[directions_map.size() - 1][2] = 1;
        //     directions_map[directions_map.size() - 1][3] = 0;
        // }
        {
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 1;
        }
    }

    MotionUtils::roll(directions_map);

    if (hsv.empty() || hsv.type() != CV_8UC3) {
        hsv = cv::Mat(frame.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    }

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    if (hsv_channels.size() == 3) {
        cv::Mat ang_map(frame.size(), CV_32F, cv::Scalar(0));
        for (size_t i = 0; i < prev_pts.size(); ++i) {
            if (status[i]) {
                float dx = curr_pts[i].x - prev_pts[i].x;
                float dy = curr_pts[i].y - prev_pts[i].y;
                float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
                if (angle < 0) angle += 360.0f;
                ang_map.at<float>(prev_pts[i]) = angle / 2;
            }
        }

        cv::Mat mag_map(frame.size(), CV_32F, cv::Scalar(0));
        for (size_t i = 0; i < prev_pts.size(); ++i) {
            if (status[i]) {
                float dx = curr_pts[i].x - prev_pts[i].x;
                float dy = curr_pts[i].y - prev_pts[i].y;
                float magnitude = std::sqrt(dx * dx + dy * dy);
                mag_map.at<float>(prev_pts[i]) = magnitude;
            }
        }

        ang_map.convertTo(hsv_channels[0], hsv_channels[0].type());
        cv::normalize(mag_map, hsv_channels[2], 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::merge(hsv_channels, hsv);
    }

    if (config_.debug) {
        cv::Mat output_frame = frame.clone();
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::line(output_frame, prev_pts[i], curr_pts[i], cv::Scalar(0, 255, 0), 2);
                cv::circle(output_frame, curr_pts[i], 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::imshow("LK Optical Flow", output_frame);
    }

    return move_mode;
}

bool MotionDetector::initializeYOLO() {
    if (yolo_initialized) {
        return true;
    }
    try {
        yolo_network = cv::dnn::readNetFromDarknet(config_.yolo_config_path, config_.yolo_weights_path);

         if (config_.use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            yolo_network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            yolo_network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "YOLO using GPU acceleration" << std::endl;
         } else if (config_.use_gpu && cv::ocl::haveOpenCL()) {
             yolo_network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
             yolo_network.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
             std::cout << "YOLO using OpenCL GPU acceleration" << std::endl;
         } else {
            yolo_network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            yolo_network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "YOLO using CPU" << std::endl;
        }

        std::ifstream class_file(config_.yolo_classes_path);
        if (class_file.is_open()) {
            std::string line;
            while (getline(class_file, line)) {
                class_names.push_back(line);
            }
            class_file.close();
        }

        yolo_initialized = true;
        std::cout << "YOLO initialized successfully with " << class_names.size() << " classes" << std::endl;
        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "Failed to initialize YOLO: " << e.what() << std::endl;
        return false;
    }
}

float MotionDetector::detectYOLOMotion(cv::Mat& frame) {
    if (!initializeYOLO()) {
        return -1.0f;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(config_.yolo_input_size, config_.yolo_input_size),
        cv::Scalar(0, 0, 0), true, false, CV_32F);

    yolo_network.setInput(blob);

    // int64 start_time = cv::getTickCount();

    std::vector<cv::Mat> outputs;
    try {
        yolo_network.forward(outputs, yolo_network.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
        std::cerr << "YOLO forward pass failed: " << e.what() << std::endl;
        return -1.0f;
    }

    // int64 end_time = cv::getTickCount();
    // double time_taken_ms = (end_time - start_time) * 1000.0 / cv::getTickFrequency();
    // std::cout << "Difference took " << time_taken_ms << " ms" << std::endl;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (auto& output : outputs) {
        for (int i = 0; i < output.rows; ++i) {
            const float* data = output.ptr<float>(i);
            float confidence = data[4];

            if (confidence >= config_.yolo_confidence_threshold) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point class_id_point;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

                if (max_class_score > config_.yolo_confidence_threshold && class_id_point.x == 0) {
                    int center_x = static_cast<int>(data[0] * frame.cols);
                    int center_y = static_cast<int>(data[1] * frame.rows);
                    int width = static_cast<int>(data[2] * frame.cols);
                    int height = static_cast<int>(data[3] * frame.rows);
                    int left = center_x - width / 2;
                    int top = center_y - height / 2;

                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id_point.x);
                }
            }
        }
    }

    if (config_.debug) {
        for (size_t i = 0; i < boxes.size(); ++i) {
            std::cout << "Detection " << i
                      << ": class_id=" << class_ids[i]
                      << ", confidence=" << confidences[i]
                      << ", box=" << boxes[i] << std::endl;
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config_.yolo_confidence_threshold, config_.yolo_nms_threshold, indices);

    std::vector<cv::Rect> current_detections;
    cv::Mat display_frame = frame.clone();

    for (int idx : indices) {
        if (class_ids[idx] == 0) {
            cv::rectangle(display_frame, boxes[idx], cv::Scalar(0, 255, 0), 2);
            std::string label = cv::format("Pedestrian: %.2f", confidences[idx]);
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(boxes[idx].y, labelSize.height);
            cv::putText(display_frame, label, cv::Point(boxes[idx].x, top - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        current_detections.push_back(boxes[idx]);
    }

    if (config_.debug) {
        cv::imshow("Pedestrian Detections", display_frame);
        cv::waitKey(1);
    }

    float move_mode = calculateMotionFromDetections(current_detections);

    previous_detections = current_detections;

    updateDirectionsFromYOLO(move_mode, current_detections);

    return move_mode;
}

float MotionDetector::calculateMotionFromDetections(const std::vector<cv::Rect>& current_detections) {
    if (previous_detections.empty() || current_detections.empty()) {
        return INT_MIN;
    }

    std::vector<float> motion_angles;

    for (const auto& current_box: current_detections) {
        cv::Point2f current_center(current_box.x + current_box.width / 2.0f,
            current_box.y + current_box.height / 2.0f);

        float min_distance = std::numeric_limits<float>::max();
        cv::Point2f best_match_center;
        bool found_match = false;

        for (const auto& prev_box: previous_detections) {
            cv::Point2f prev_center(prev_box.x + prev_box.width / 2.0f,
                prev_box.y + prev_box.height / 2.0f);

            float distance = cv::norm(current_center - prev_center);
            if (distance < min_distance && distance < 100.0f) { //thresold for matching
                min_distance = distance;
                best_match_center = prev_center;
                found_match = true;
            }
        }

        //TODO: better angle detection and calculation
        if (found_match) {
            cv::Point2f motion_vector = current_center - best_match_center;
            float motion_angle = atan2(motion_vector.y, motion_vector.x) * 180.0f / CV_PI;
            if (motion_angle < 0) {
                motion_angle += 360.0f;
            }
            motion_angles.push_back(motion_angle);
        }
    }

    return MotionUtils::calculateMode(motion_angles);
}

void MotionDetector::updateDirectionsFromYOLO(float move_mode, const std::vector<cv::Rect>& detections) {

    //TODO: Compare this to the other detections way, maybe make it to a separate function
    if (detections.empty()) {
        directions_map[directions_map.size() - 1][0] = 0;
        directions_map[directions_map.size() - 1][1] = 0;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 1;
    } else {

        bool is_moving_up =
            MotionUtils::isAngleInRange(move_mode, config_.angle_up_min, config_.angle_up_max) ||
            MotionUtils::isAngleInRange(move_mode, config_.angle_down_min, config_.angle_down_max);

        if (is_moving_up) {
            directions_map[directions_map.size() - 1][0] = 3.5f;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 0;
        } else if (move_mode > 5.0f) {
            // Other directions
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 1;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 0;
        } else {
            directions_map[directions_map.size() - 1][0] = 0;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 1;
            directions_map[directions_map.size() - 1][3] = 0;
        }
    }

    MotionUtils::roll(directions_map);
}

bool MotionDetector::processFrame(cv::Mat& frame, cv::Mat& orig_frame, cv::Mat& gray_previous) {
    frame = frame(cv::Range(config_.row_start, config_.row_end), cv::Range(config_.col_start, config_.col_end));

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat hsv(frame.size(), CV_8UC3, cv::Scalar(0, 255, 0));

    float move_mode = detectMotion(frame, gray, gray_previous, hsv);

    int loc = MotionUtils::calculateMaxMeanColumn(directions_map);

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

    cv::putText(orig_frame, "Angle: " + std::to_string(static_cast<int>(move_mode)),
            cv::Point(30, 150), cv::FONT_HERSHEY_COMPLEX,
            frame.cols / 500.0, cv::Scalar(0, 0, 255), 6);

    cv::putText(orig_frame, text, cv::Point(30, 90), cv::FONT_HERSHEY_COMPLEX,
        orig_frame.cols / 500.0, cv::Scalar(0, 0, 255), text_thinkness);

    gray_previous = gray;

    return loc == 0 || loc == 2;
}

void MotionDetector::run() {
    initializeParallelProcessing();

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::VideoCapture cap(config_.video_src);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source: " << config_.video_src << std::endl;
        return;
    }

    Benchmark timer;
    std::vector<BenchmarkResult> results;
    int frame_index = config_.seek;

    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

    config_.row_end = height - config_.row_end;
    config_.col_end = width - config_.col_end;

    cap.set(cv::CAP_PROP_POS_FRAMES, config_.seek);

    cv::Mat frame_previous;
    cap >> frame_previous;
    if (frame_previous.empty()) {
        std::cerr << "Error: Failed to grab first frame" << std::endl;
        return;
    }

    cv::Mat gray_previous;
    cv::cvtColor(frame_previous(cv::Range(config_.row_start, config_.row_end), cv::Range(config_.col_start, config_.col_end)),
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
        bool crossing_intent = processFrame(frame, orig_frame, gray_previous);
        double elapsed = timer.stop();

        cv::putText(orig_frame, "FPS: " + std::to_string(1000.0 / elapsed), cv::Point(30, 200), cv::FONT_HERSHEY_COMPLEX,
                    frame.cols / 500.0, cv::Scalar(0, 255, 0), 3);

        results.push_back({
            frame_index++,
            config_.use_gpu,
            elapsed,
            crossing_intent
        });

        cv::rectangle(orig_frame, cv::Point(config_.col_start, config_.row_start), cv::Point(config_.col_end, config_.row_end),
            cv::Scalar(0, 255, 0), 3);
        cv::imshow(WINDOW_NAME, orig_frame);

        if (cv::waitKey(1) == 'q' || (config_.seek_end > 0 && cap.get(cv::CAP_PROP_POS_FRAMES) >= config_.seek_end)) {
            break;
        }
    }

    saveBenchmarkResults(config_.use_gpu, config_.algorithm, results);

    cap.release();
    cv::destroyAllWindows();
}