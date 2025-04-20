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
};

std::string getTimestamp();
void saveResultToCSV(const std::string& filename, const std::vector<BenchmarkResult>& results);
void saveBenchmarkResults(bool use_gpu, const std::vector<BenchmarkResult>& results);

#endif //BENCHMARK_H
