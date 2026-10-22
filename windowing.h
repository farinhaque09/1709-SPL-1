#pragma once

#include <vector>

struct WindowedDataset {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

WindowedDataset createSlidingWindows(const std::vector<double>& series, int windowSize);

struct TrainTestSplit {
    WindowedDataset train;
    WindowedDataset test;
};

TrainTestSplit trainTestSplit(const WindowedDataset& data, double trainRatio);
