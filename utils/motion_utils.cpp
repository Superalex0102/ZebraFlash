#include <unordered_map>

#include "motion_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

float MotionUtils::calculateMode(const std::vector<float>& values) {
    if (values.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::unordered_map<float, int> frequency_map;
    for (const float& value : values) {
        int binned = static_cast<int>(std::round(value));
        frequency_map[binned]++;
    }

    float mode = values[0];
    int max_count = 0;
    for (const auto& pair : frequency_map) {
        if (pair.second > max_count) {
            mode = pair.first;
            max_count = pair.second;
        }
    }
    return mode;
}

void MotionUtils::roll(std::vector<std::vector<int>>& map) {
    if (map.empty()) {
        return;
    }

    std::vector<int> first_row = map[0];

    for (size_t i = 0; i < map.size() - 1; ++i) {
        map[i] = map[i + 1];
    }

    map[map.size() - 1] = first_row;
}

int MotionUtils::calculateMaxMeanColumn(const std::vector<std::vector<int>>& map) {
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