#include "windowing.h"

#include <stdexcept>

WindowedDataset createSlidingWindows(const std::vector<double>& series, int windowSize) {
    if (windowSize <= 0) {
        throw std::runtime_error("windowSize must be positive");
    }
    if (static_cast<int>(series.size()) <= windowSize) {
        throw std::runtime_error("Not enough data points to build even one window.");
    }

    WindowedDataset dataset;
    int numWindows = static_cast<int>(series.size()) - windowSize;
    dataset.X.reserve(numWindows);
    dataset.y.reserve(numWindows);

    for (int start = 0; start < numWindows; ++start) {
        std::vector<double> window(series.begin() + start,
                                    series.begin() + start + windowSize);
        double target = series[start + windowSize];
        dataset.X.push_back(std::move(window));
        dataset.y.push_back(target);
    }

    return dataset;
}

TrainTestSplit trainTestSplit(const WindowedDataset& data, double trainRatio) {
    if (trainRatio <= 0.0 || trainRatio >= 1.0) {
        throw std::runtime_error("trainRatio must be between 0 and 1 (exclusive)");
    }

    int total = static_cast<int>(data.X.size());
    int trainSize = static_cast<int>(total * trainRatio);

    if (trainSize <= 0) trainSize = 1;
    if (trainSize >= total) trainSize = total - 1;

    TrainTestSplit split;
    split.train.X.assign(data.X.begin(), data.X.begin() + trainSize);
    split.train.y.assign(data.y.begin(), data.y.begin() + trainSize);

    split.test.X.assign(data.X.begin() + trainSize, data.X.end());
    split.test.y.assign(data.y.begin() + trainSize, data.y.end());

    return split;
}
