#pragma once

#include <vector>

class MovingAverageModel {
public:
    static std::vector<double> predict(const std::vector<std::vector<double>>& windows);
};
