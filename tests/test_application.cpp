#include <filesystem>
#include <gtest/gtest.h>
#include "../motion-detector/motion_detector.h"
#include "../benchmark/benchmark.h"

TEST(BenchmarksTest, TestFarneSingleCPU) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    std::cout << "Current working directory: "
          << std::filesystem::current_path() << std::endl;

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneMultiCPU) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    std::cout << "Current working directory: "
          << std::filesystem::current_path() << std::endl;

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneGPU) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    std::cout << "Current working directory: "
          << std::filesystem::current_path() << std::endl;

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKMultiCPU) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    std::cout << "Current working directory: "
          << std::filesystem::current_path() << std::endl;

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKGPU) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    std::cout << "Current working directory: "
          << std::filesystem::current_path() << std::endl;

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOSingleCPU) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    std::cout << "Current working directory: "
          << std::filesystem::current_path() << std::endl;

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOGPU) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    std::cout << "Current working directory: "
          << std::filesystem::current_path() << std::endl;

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}