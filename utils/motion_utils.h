#ifndef MOTION_UTILS_H
#define MOTION_UTILS_H

#include <vector>

class MotionUtils {
public:
    static float calculateMode(const std::vector<float>& values);
    static void roll(std::vector<std::vector<int>>& map);
    static int calculateMaxMeanColumn(const std::vector<std::vector<int>>& map);
    static bool isAngleInRange(float angle, int min, int max);
};

#endif //MOTION_UTILS_H
