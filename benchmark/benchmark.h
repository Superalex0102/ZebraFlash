#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>

class Benchmark {
public:
    Benchmark();
    ~Benchmark();

    void start();
    double stop();

private:
    struct Impl;
    Impl *impl;
};

struct BenchmarkResult {
    int frame_index;
    bool use_gpu;
    double process_time_ms;
    bool crossing_intent;
};

struct CrossIntent {
    int frame_index;
    bool crossing_intent;
};

std::string getTimestamp();
std::vector<CrossIntent> loadGroundTruthCrossingIntent(const std::string& xml_filepath);
double crossingIntentRate(const std::vector<BenchmarkResult>& results, const std::vector<CrossIntent>& ground_truth);
void saveResultToCSV(const std::string& filename, const std::vector<BenchmarkResult>& results);
void saveBenchmarkResults(bool use_gpu, const std::string& algorithm, const std::vector<BenchmarkResult>& results);

#endif //BENCHMARK_H
