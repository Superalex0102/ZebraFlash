#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <opencv2/opencv.hpp>
#include  <yaml-cpp/yaml.h>
#include <string>
#include <vector>

class MotionDetector {
public:
    MotionDetector(const std::string& configFile);
    void run();

private:
    std::string video_src;
}

#endif //MOTION_DETECTOR_H
