#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <sstream>

#include "benchmark.h"


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

void saveResultToCSV(const std::string& filename, const std::vector<BenchmarkResult>& results) {
    std::ofstream file(filename);

    if (file.is_open()) {
        double total_fps = 0.0;
        for (const auto& r : results) {
            if (r.process_time_ms > 0.0) {
                total_fps += 1000.0 / r.process_time_ms;
            }
        }
        double average_fps = results.empty() ? 0.0 : total_fps / results.size();

        file << "Average FPS:," << std::fixed << std::setprecision(3) << average_fps << "\n";
        file << "Frame Index,Use GPU,FPS\n";
        for (const auto& r : results) {
            double fps = (r.process_time_ms > 0.0) ? 1000.0 / r.process_time_ms : 0.0;

            file << r.frame_index << ","
                 << (r.use_gpu ? "Yes" : "No") << ","
                 << std::fixed << std::setprecision(3) << fps << "\n";
        }
        file.close();
        std::cout << "Results saved to " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
    }
}

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

void saveBenchmarkResults(bool use_gpu, const std::string& algorithm, const std::vector<BenchmarkResult>& results) {
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

    saveResultToCSV(filename, results);
}