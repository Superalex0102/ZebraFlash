#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

const std::string INPUT_FILE = "..\\config\\params_input_file.yml";
const std::string WINDOW_NAME = "window";

float calculateMode(const cv::Mat& mat) {
    cv::Mat sorted;
    mat.reshape(1, 1).copyTo(sorted);
    cv::sort(sorted, sorted, cv::SORT_ASCENDING);

    float mode = sorted.at<float>(0);
    int max_count = 1, count = 1;

    for (int i = 1; i < sorted.total(); i++) {
        if (sorted.at<float>(i) == sorted.at<float>(i - 1)) {
            count++;
        } else {
            if (count > max_count) {
                max_count = count;
                mode = sorted.at<float>(i - 1);
            }
            count = 1;
        }
    }

    return mode;
}

void roll(std::vector<std::vector<int>>& map) {
    if (map.empty()) {
        return;
    }

    std::vector<int> first_row = map[0];

    for (size_t i = 0; i < map.size() - 1; ++i) {
        map[i] = map[i + 1];
    }

    map[map.size() - 1] = first_row;
}

int calculateMaxMeanColumn(const std::vector<std::vector<int>>& map) {
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

    std::vector<std::vector<int>> directions_map(size, std::vector<int>(4, 0));

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

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(gray_previous, gray, flow, pyr_scale, levels,
            winsize, iterations, poly_n, poly_sigma, cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

        std::vector<cv::Mat> flow_channels(2);
        cv::split(flow, flow_channels);

        cv::Mat mag, ang;

        cv::cartToPolar(flow_channels[0], flow_channels[1], mag, ang, true);

        cv::Mat ang_180 = ang / 2;

        gray_previous = gray;

        cv::Mat mask = mag > threshold;

        std::vector<float> move_sense;
        ang.reshape(1, 1).copyTo(move_sense);

        cv::Mat move_sense_mat(move_sense);

        float move_mode = calculateMode(move_sense_mat);
        bool is_moving_up = (move_mode >= angle_min && move_mode <= angle_max);

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

        if (hsv.empty() || hsv.type() != CV_8UC3) {
            hsv = cv::Mat(frame_previous.size(), CV_8UC3, cv::Scalar(0, 255, 0));
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

    return 0;
}