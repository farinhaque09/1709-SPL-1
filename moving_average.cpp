#include "moving_average.h"

#include <stdexcept>

std::vector<double> MovingAverageModel::predict(const std::vector<std::vector<double>>& windows) {
    std::vector<double> predictions;
    predictions.reserve(windows.size());

    for (const std::vector<double>& window : windows) {
        if (window.empty()) {
            throw std::runtime_error("MovingAverageModel received an empty window");
        }
        double sum = 0.0;
        for (double v : window) {
            sum += v;
        }
        predictions.push_back(sum / static_cast<double>(window.size()));
    }

    return predictions;
}
