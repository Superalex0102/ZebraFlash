#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <sstream>

#include "benchmark.h"

#include <unordered_map>

struct Benchmark::Impl {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};

Benchmark::Benchmark() : impl(new Impl) {}

void Benchmark::start() {
    impl->start_time = std::chrono::high_resolution_clock::now();
}

double Benchmark::stop() {
    impl->end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = impl->end_time - impl->start_time;
    return elapsed.count();
}

Benchmark::~Benchmark() {
    delete impl;
}

std::vector<CrossIntent> loadGroundTruthCrossingIntent(const std::string& csv_filepath) {
    std::vector<CrossIntent> crossing_intent_data;
    std::ifstream file(csv_filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening CSV file: " << csv_filepath << std::endl;
        return crossing_intent_data;
    }

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string frame_str, intent_str;

        if (std::getline(ss, frame_str, ',') && std::getline(ss, intent_str)) {
            int frame = std::stoi(frame_str);
            bool is_crossing = (intent_str == "crossing");
            crossing_intent_data.push_back({frame, is_crossing});
        }
    }

    return crossing_intent_data;
}


double crossingIntentRate(const std::vector<BenchmarkResult>& results, const std::vector<CrossIntent>& ground_truth) {
    int match_count = 0;
    int total_count = 0;

    for (const auto& result : results) {
        for (const auto& truth : ground_truth) {
            if (result.frame_index == truth.frame_index) {
                total_count++;
                if (result.crossing_intent == truth.crossing_intent) {
                    match_count++;
                }
                break;
            }
        }
    }

    if (total_count == 0) {
        return -1;
    }

    return static_cast<double>(match_count) / total_count;
}

void saveResultToCSV(const std::string& filename,
                     const std::vector<BenchmarkResult>& results,
                     const std::vector<CrossIntent>& ground_truth) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    double total_fps = 0.0;
    for (const auto& r : results) {
        if (r.process_time_ms > 0.0) {
            total_fps += 1000.0 / r.process_time_ms;
        }
    }
    double average_fps = results.empty() ? 0.0 : total_fps / results.size();

    double crossing_intent_rate = crossingIntentRate(results, ground_truth);

    double predicted_crossing = 0;
    double predicted_not_crossing = 0;
    double ground_crossing = 0;
    double ground_not_crossing = 0;

    std::unordered_map<int, bool> ground_truth_map;
    for (const auto& r : results) {
        auto it = std::find_if(ground_truth.begin(), ground_truth.end(),
                               [&](const CrossIntent& gt) { return gt.frame_index == r.frame_index; });
        if (it != ground_truth.end()) {
            ground_truth_map[it->frame_index] = it->crossing_intent;
            if (it->crossing_intent)
                ground_crossing++;
            else
                ground_not_crossing++;
        }
    }

    for (const auto& r : results) {
        if (r.crossing_intent)
            predicted_crossing++;
        else
            predicted_not_crossing++;
    }

    file << "Average FPS:," << std::fixed << std::setprecision(3) << average_fps << "\n";
    file << "Crossing Intent Rate (%):," << crossing_intent_rate << "\n";
    file << "Not Crossing Class Error (%):," << (1.0 - predicted_not_crossing / ground_not_crossing) << "\n";
    file << "Crossing Class Error (%):," << (1.0 - predicted_crossing / ground_crossing) << "\n";

    file << "Frame Index,Use GPU,FPS,Predicted Intent,Groundtruth Intent\n";

    for (const auto& r : results) {
        double fps = (r.process_time_ms > 0.0) ? 1000.0 / r.process_time_ms : 0.0;
        bool predicted_intent = r.crossing_intent;
        auto gt_it = ground_truth_map.find(r.frame_index);
        bool groundtruth_intent = (gt_it != ground_truth_map.end()) ? gt_it->second : false;

        file << r.frame_index << ","
             << (r.use_gpu ? "Yes" : "No") << ","
             << std::fixed << std::setprecision(3) << fps << ","
             << (predicted_intent ? "Yes" : "No") << ","
             << (groundtruth_intent ? "Yes" : "No") << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

void saveBenchmarkResults(bool use_gpu, const std::string& algorithm, const std::vector<BenchmarkResult>& results, const std::string& annotationFile) {
    std::string results_dir = "results";
    if (!std::filesystem::exists(results_dir)) {
        if (!std::filesystem::create_directory(results_dir)) {
            std::cerr << "Error: Could not create 'results' directory." << std::endl;
            return;
        }
    }

    std::string timestamp = getTimestamp();
    std::string filename = results_dir + "/" + (use_gpu ? "gpu" : "cpu") + "_" + algorithm + "_benchmark_" + timestamp + ".csv";

    std::cout << "Saving benchmark results to: " << filename << std::endl;
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    auto ground_truth = loadGroundTruthCrossingIntent(annotationFile);

    saveResultToCSV(filename, results, ground_truth);
}