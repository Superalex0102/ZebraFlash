#include "Algorithm.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>

namespace YAML {
    template<>
    struct convert<Algorithm> {
        static bool decode(const Node& node, Algorithm& rhs) {
            std::string value = node.as<std::string>();
            std::transform(value.begin(), value.end(), value.begin(), ::toupper);

            if (value == "OPTICAL") {
                rhs = Algorithm::OPTICAL;
            } else if (value == "YOLO") {
                rhs = Algorithm::YOLO;
            } else {
                throw std::runtime_error("Unknown algorithm: " + value);
            }
            return true;
        }
    };
}