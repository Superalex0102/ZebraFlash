#include <filesystem>
#include <gtest/gtest.h>
#include "../motion-detector/motion_detector.h"
#include "../benchmark/benchmark.h"

TEST(BenchmarksTest, TestFarneSingleCPUCrowd) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad1.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad1_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneSingleCPUDayShadow) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad2.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad2_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneSingleCPUNight) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad3.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad3_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneMultiCPUCrowd) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad1.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad1_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneMultiCPUDayShadow) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad2.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad2_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneMultiCPUNight) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad3.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad3_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneGPUCrowd) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad1.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad1_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneGPUDayShadow) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad2.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad2_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestFarneGPUNight) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "FARNE";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad3.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad3_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_FARNE_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKMultiCPUCrowd) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad1.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad1_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKMultiCPUDayShadow) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad2.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad2_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKMultiCPUNight) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = true;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad3.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad3_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKGPUCrowd) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad1.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad1_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKGPUDayShadow) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad2.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad2_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestLKGPUNight) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "LK";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad3.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad3_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_LK_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOSingleCPUCrowd) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad1.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad1_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOSingleCPUDayShadow) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad2.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad2_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOSingleCPUNight) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = false;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad3.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad3_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/cpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOGPUCrowd) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad1.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad1_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOGPUDayShadow) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad2.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad2_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}

TEST(BenchmarksTest, TestYOLOGPUNight) {
    const std::string INPUT_FILE = "../../config/params_input_file.yml";

    try {
        MotionDetector detector(INPUT_FILE);
        detector.getConfig().algorithm = "YOLO";
        detector.getConfig().use_gpu = true;
        detector.getConfig().use_multi_thread = false;
        detector.getConfig().debug = false;
        detector.getConfig().video_src = "../../input/abbeyroad3.mp4";
        detector.getConfig().video_annot = "../../input/abbeyroad3_annotation.csv";
        detector.getConfig().seek_end = 2500;
        detector.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ASSERT_TRUE(std::filesystem::exists("results/gpu_YOLO_benchmark_" + getTimestamp()) + ".csv") << "Benchmark file was not created.";
}