#include <yaml-cpp/yaml.h>

const std::string INPUT_FILE = "../config/params_input_file.yaml";

int main() {
    YAML::Node config = YAML::LoadFile(INPUT_FILE);
}