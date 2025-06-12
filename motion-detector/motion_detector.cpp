#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>

#include "motion_detector.h"

#include <fstream>
#include <thread>

#include "../benchmark/benchmark.h"
#include "../utils/motion_utils.h"

MotionDetector::MotionDetector(const std::string &configFile) {
    loadConfig(configFile);

    if (use_multi_thread) {
        thread_amount = thread_amount == -1 ? std::thread::hardware_concurrency() : thread_amount;
        thread_pool = std::make_unique<ThreadPool>(thread_amount);
    }

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
    row_start = config["upper_margin"].as<int>();
    row_end = config["bottom_margin"].as<int>();
    col_start = config["left_margin"].as<int>();
    col_end = config["right_margin"].as<int>();
    res_ratio = config["res_ratio"].as<double>();
    threshold = config["threshold"].as<double>();
    angle_up_min = config["angle_up_min"].as<int>();
    angle_up_max = config["angle_up_max"].as<int>();
    angle_down_min = config["angle_down_min"].as<int>();
    angle_down_max = config["angle_down_max"].as<int>();
    binary_threshold = config["binary_threshold"].as<int>();
    threshold_count = config["threshold_count"].as<int>();
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
    algorithm = config["algorithm"].as<std::string>();
    yolo_weights_path = config["yolo_weights_path"].as<std::string>();
    yolo_config_path = config["yolo_config_path"].as<std::string>();
    yolo_classes_path = config["yolo_classes_path"].as<std::string>();
    yolo_confidence_threshold = config["yolo_confidence_threshold"].as<float>();
    yolo_nms_threshold = config["yolo_nms_threshold"].as<float>();
    yolo_input_size = config["yolo_input_size"].as<int>();
}

float MotionDetector::detectMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv) {
    float result;
    if (algorithm == "OPTICAL") {
        result = detectOpticalFlowMotion(frame, gray, gray_previous, hsv);
    }
    else if (algorithm == "YOLO") {
        result = detectYOLOMotion(frame);
    }

    return result;
}

static cv::cuda::GpuMat d_gray_previous, d_gray, d_flow;

float MotionDetector::detectOpticalFlowMotion(cv::Mat& frame, cv::Mat& gray, cv::Mat& gray_previous, cv::Mat& hsv) {
    cv::Mat flow(gray.size(), CV_32FC2);
    cv::Mat mask, ang, ang_180, mag;

    if (use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        try {
            cv::cuda::Stream stream;
            d_gray.upload(gray, stream);
            d_gray_previous.upload(gray_previous, stream);

            if (d_flow.size() != gray.size() || d_flow.type() != CV_32FC2) {
                d_flow.release();
                d_flow.create(gray.size(), CV_32FC2);
            }

            static auto farneback = cv::cuda::FarnebackOpticalFlow::create(
                levels, pyr_scale, false, winsize, iterations, poly_n, poly_sigma, 0);

            farneback->calc(d_gray_previous, d_gray, d_flow, stream);

            std::vector<cv::cuda::GpuMat> d_flow_channels(3);
            cv::cuda::split(d_flow, d_flow_channels, stream);

            cv::cuda::GpuMat d_mag, d_ang;
            cv::cuda::cartToPolar(d_flow_channels[0], d_flow_channels[1], d_mag, d_ang, true, stream);

            cv::cuda::GpuMat d_ang_180;
            cv::cuda::divide(d_ang, cv::Scalar(2.0), d_ang_180);

            cv::cuda::GpuMat d_mask;
            cv::cuda::threshold(d_mag, d_mask, threshold, 255, cv::THRESH_BINARY, stream);

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
    }
    else if (use_multi_thread) {
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

        std::vector<cv::Mat> flow_channels(2);
        cv::split(flow, flow_channels);

        cv::cartToPolar(flow_channels[0], flow_channels[1], mag, ang, true);

        ang_180 = ang / 2;
        mask = mag > threshold;
    }
    else {
        cv::calcOpticalFlowFarneback(gray_previous, gray, flow, pyr_scale, levels,
            winsize, iterations, poly_n, poly_sigma, 0);
    }

    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(mask, non_zero_points);

    std::vector<float> move_sense;
    for (const auto& pt : non_zero_points) {
        move_sense.push_back(ang.at<float>(pt));
    }

    float move_mode = MotionUtils::calculateMode(move_sense);
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
        cv::Mat fg_mask;
        backSub->apply(frame, fg_mask);

        cv::Mat fg_mask_blurred;
        cv::GaussianBlur(fg_mask, fg_mask_blurred, cv::Size(7, 7), 0);

        cv::Mat tresh_frame;
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

bool MotionDetector::initializeYOLO() {
    if (yolo_initialized) {
        return true;
    }

    try {
        yolo_network = cv::dnn::readNetFromDarknet(yolo_config_path, yolo_weights_path);

         if (use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
             std::cout << cv::getBuildInformation() << std::endl;
            yolo_network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            yolo_network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "YOLO using GPU acceleration" << std::endl;
        } else {
            yolo_network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            yolo_network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "YOLO using CPU" << std::endl;
        }

        std::ifstream class_file(yolo_classes_path);
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
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(yolo_input_size, yolo_input_size),
        cv::Scalar(0, 0, 0), true, true, CV_32F);

    yolo_network.setInput(blob);

    int64 start_time = cv::getTickCount();

    std::vector<cv::Mat> outputs;
    try {
        yolo_network.forward(outputs, yolo_network.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
        std::cerr << "YOLO forward pass failed: " << e.what() << std::endl;
        return -1.0f;
    }

    int64 end_time = cv::getTickCount();
    double time_taken_ms = (end_time - start_time) * 1000.0 / cv::getTickFrequency();
    std::cout << "Difference took " << time_taken_ms << " ms" << std::endl;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float scale_x = static_cast<float>(frame.cols) / yolo_input_size;
    float scale_y = static_cast<float>(frame.rows) / yolo_input_size;

    for (auto& output : outputs) {
        for (int i = 0; i < output.rows; ++i) {
            const float* data = output.ptr<float>(i);
            float confidence = data[4];

            if (confidence >= yolo_confidence_threshold) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point class_id_point;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

                if (max_class_score > yolo_confidence_threshold) {
                    int center_x = static_cast<int>(data[0] * scale_x);
                    int center_y = static_cast<int>(data[1] * scale_y);
                    int width = static_cast<int>(data[2] * scale_x);
                    int height = static_cast<int>(data[3] * scale_y);
                    int left = center_x - width / 2;
                    int top = center_y - height / 2;

                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id_point.x);
                }
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, yolo_confidence_threshold, yolo_nms_threshold, indices);

    std::vector<cv::Rect> current_detections;
    for (int idx : indices) {
        current_detections.push_back(boxes[idx]);

        if (debug) {
            cv::rectangle(frame, boxes[idx], cv::Scalar(0, 255, 0), 2);
            std::string label = class_names.size() > class_ids[idx] ?
                               class_names[class_ids[idx]] : "Unknown";
            cv::putText(frame, label + " " + std::to_string(confidences[idx]),
                       cv::Point(boxes[idx].x, boxes[idx].y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }

    float motion_magnitude = calculateMotionFromDetections(current_detections);

    previous_detections = current_detections;

    updateDirectionsFromYOLO(motion_magnitude, current_detections);

    return motion_magnitude;
}

float MotionDetector::calculateMotionFromDetections(const std::vector<cv::Rect>& current_detections) {
    if (previous_detections.empty() || current_detections.empty()) {
        return 0.0f;
    }

    float total_motion = 0.0f;
    int motion_count = 0;

    for (const auto& current_box: current_detections) {
        cv::Point2f current_center(current_box.x + current_box.width/2.0f,
            current_box.y + current_box.height/2.0f);

        float min_distance = std::numeric_limits<float>::max();
        cv::Point2f best_match_center;
        bool found_match = false;

        for (const auto& prev_box: previous_detections) {
            cv::Point2f prev_center(prev_box.x + prev_box.width/2.0f,
                prev_box.y + prev_box.height/2.0f);

            float distance = cv::norm(current_center - prev_center);
            if (distance < min_distance && distance < 100.0f) { //thresold for matching
                min_distance = distance;
                best_match_center = prev_center;
                found_match = true;
            }
        }

        if (found_match) {
            cv::Point2f motion_vector = current_center - best_match_center;
            float motion_angle = atan2(motion_vector.y, motion_vector.x) * 180.0f / CV_PI;
            if (motion_angle < 0) {
                motion_angle += 360.0f;
            }

            total_motion += motion_angle;
            motion_count++;
        }
    }

    return motion_count > 0 ? total_motion / motion_count : 0.0f;
}

void MotionDetector::updateDirectionsFromYOLO(float motion_magnitude, const std::vector<cv::Rect>& detections) {
    if (detections.empty()) {
        directions_map[directions_map.size() - 1][0] = 0;
        directions_map[directions_map.size() - 1][1] = 0;
        directions_map[directions_map.size() - 1][2] = 0;
        directions_map[directions_map.size() - 1][3] = 1;
    } else {
        if (motion_magnitude >= angle_up_min && motion_magnitude <= angle_up_max) {
            // Upward motion
            directions_map[directions_map.size() - 1][0] = 1;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 0;
        } else if (motion_magnitude >= angle_down_min && motion_magnitude <= angle_down_max) {
            // Downward motion
            directions_map[directions_map.size() - 1][0] = 1;
            directions_map[directions_map.size() - 1][1] = 0;
            directions_map[directions_map.size() - 1][2] = 0;
            directions_map[directions_map.size() - 1][3] = 0;
        } else if (motion_magnitude > 5.0f) {
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

void MotionDetector::processFrame(cv::Mat& frame, cv::Mat& orig_frame, cv::Mat& gray_previous) {
    frame = frame(cv::Range(row_start, row_end), cv::Range(col_start, col_end));

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

    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

    row_end = height - row_end;
    col_end = width - col_end;

    cap.set(cv::CAP_PROP_POS_MSEC, seek);

    cv::Mat frame_previous;
    cap >> frame_previous;
    if (frame_previous.empty()) {
        std::cerr << "Error: Failed to grab first frame" << std::endl;
        return;
    }

    cv::Mat gray_previous;
    cv::cvtColor(frame_previous(cv::Range(row_start, row_end), cv::Range(col_start, col_end)),
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

        cv::rectangle(orig_frame, cv::Point(col_start, row_start), cv::Point(col_end, row_end),
            cv::Scalar(0, 255, 0), 3);
        cv::imshow(WINDOW_NAME, orig_frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    saveBenchmarkResults(use_gpu, results);

    cap.release();
    cv::destroyAllWindows();
}