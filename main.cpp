#include <torch/torch.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "csv_utils.h"
#include "metrics.h"
#include "moving_average.h"
#include "scaler.h"
#include "simple_nn.h"
#include "windowing.h"

struct Config {
    std::string csvPath = "data/sales_data.csv";
    int windowSize = 14;
    double trainRatio = 0.8;
    double validationRatio = 0.1;
    int epochs = 150;
    int hiddenSize = 32;
    double learningRate = 0.01;
    int patience = 20;
    int previewRows = 5;
};

Config parseArgs(int argc, char** argv) {
    Config config;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv") {
            config.csvPath = argv[++i];
        } else if (arg == "--window") {
            config.windowSize = std::stoi(argv[++i]);
        } else if (arg == "--train_ratio") {
            config.trainRatio = std::stod(argv[++i]);
        } else if (arg == "--val_ratio") {
            config.validationRatio = std::stod(argv[++i]);
        } else if (arg == "--epochs") {
            config.epochs = std::stoi(argv[++i]);
        } else if (arg == "--hidden") {
            config.hiddenSize = std::stoi(argv[++i]);
        } else if (arg == "--lr") {
            config.learningRate = std::stod(argv[++i]);
        } else if (arg == "--patience") {
            config.patience = std::stoi(argv[++i]);
        } else if (arg == "--preview") {
            config.previewRows = std::stoi(argv[++i]);
        }
    }
    return config;
}

torch::Tensor windowsToTensor(const std::vector<std::vector<double>>& windows) {
    int64_t rows = static_cast<int64_t>(windows.size());
    int64_t cols = static_cast<int64_t>(windows.empty() ? 0 : windows[0].size());
    torch::Tensor tensor = torch::empty({rows, cols}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            accessor[i][j] = static_cast<float>(windows[i][j]);
        }
    }
    return tensor;
}

torch::Tensor targetsToTensor(const std::vector<double>& targets) {
    int64_t rows = static_cast<int64_t>(targets.size());
    torch::Tensor tensor = torch::empty({rows, 1}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    for (int64_t i = 0; i < rows; ++i) {
        accessor[i][0] = static_cast<float>(targets[i]);
    }
    return tensor;
}

void printLine() {
    std::cout << "------------------------------------------------------------\n";
}

void printMetricsRow(const std::string& name, const MetricSet& metrics) {
    std::cout << "    " << std::left << std::setw(14) << name << std::right
              << std::setw(10) << metrics.mae << std::setw(10) << metrics.rmse
              << std::setw(9) << metrics.mape << "%\n";
}

int main(int argc, char** argv) {
    try {
        Config config = parseArgs(argc, argv);
        std::cout << std::fixed << std::setprecision(2);
        torch::manual_seed(42);

        std::cout << "Time-Series Demand Forecasting Using LSTM - progress demo (Basic NN)\n";
        printLine();

        std::cout << "[1] Loading and cleaning data from " << config.csvPath << "\n";
        std::vector<SalesRecord> records = loadSalesCSV(config.csvPath);
        int dropped = cleanSalesRecords(records);
        std::cout << "    rows remaining: " << records.size() << " (" << dropped << " dropped)\n";

        std::vector<double> salesSeries;
        salesSeries.reserve(records.size());
        for (const SalesRecord& r : records) {
            salesSeries.push_back(r.sales);
        }

        std::cout << "\n[2] Min-max scaling (fitted on the training portion only)\n";
        int trainCutoff = static_cast<int>(salesSeries.size() * config.trainRatio);
        std::vector<double> trainPortionRaw(salesSeries.begin(), salesSeries.begin() + trainCutoff);
        MinMaxScaler scaler;
        scaler.fit(trainPortionRaw);
        std::vector<double> scaledSeries = scaler.transform(salesSeries);
        std::cout << "    training min: " << scaler.getMin() << " | training max: " << scaler.getMax() << "\n";

        std::cout << "\n[3] Sliding windows and splits (window = " << config.windowSize << " days)\n";
        WindowedDataset windowed = createSlidingWindows(scaledSeries, config.windowSize);
        TrainTestSplit split = trainTestSplit(windowed, config.trainRatio);

        WindowedDataset fitData = split.train;
        WindowedDataset validationData;
        if (config.validationRatio > 0.0) {
            TrainTestSplit inner = trainTestSplit(split.train, 1.0 - config.validationRatio);
            fitData = inner.train;
            validationData = inner.test;
        }
        std::cout << "    total windows: " << windowed.X.size() << "\n";
        std::cout << "    training: " << fitData.X.size()
                  << " | validation: " << validationData.X.size()
                  << " | testing: " << split.test.X.size() << "\n";

        std::cout << "\n[4] Moving Average baseline\n";
        std::vector<double> actualReal = scaler.inverseTransform(split.test.y);
        std::vector<double> maReal = scaler.inverseTransform(MovingAverageModel::predict(split.test.X));
        std::cout << "    predictions made for " << maReal.size() << " test windows\n";

        std::cout << "\n[5] Training the Basic Neural Network (" << config.windowSize << " -> " << config.hiddenSize
                  << " -> " << config.hiddenSize << " -> 1, Adam, MSE loss)\n";
        torch::Tensor xFit = windowsToTensor(fitData.X);
        torch::Tensor yFit = targetsToTensor(fitData.y);
        torch::Tensor xValidation = windowsToTensor(validationData.X);
        torch::Tensor yValidation = targetsToTensor(validationData.y);
        torch::Tensor xTest = windowsToTensor(split.test.X);

        SimpleNN simpleNN(config.windowSize, config.hiddenSize);
        TrainingHistory history = trainSimpleNN(simpleNN, xFit, yFit, xValidation, yValidation,
                                                config.epochs, config.learningRate, config.patience);

        std::cout << std::setprecision(5);
        std::cout << "    epochs run: " << history.epochsRun << " | best epoch: " << history.bestEpoch
                  << " | stopped early: " << (history.stoppedEarly ? "yes" : "no") << "\n";
        std::cout << "    first training loss: " << history.trainLoss.front()
                  << " | last training loss: " << history.trainLoss.back() << "\n";
        if (!history.validationLoss.empty()) {
            double bestValidation = *std::min_element(history.validationLoss.begin(), history.validationLoss.end());
            std::cout << "    best validation loss: " << bestValidation << "\n";
        }
        std::cout << std::setprecision(2);

        std::cout << "\n[6] Predictions on the test windows (real units)\n";
        std::vector<double> nnReal = scaler.inverseTransform(predictSimpleNN(simpleNN, xTest));

        int preview = std::min(config.previewRows, static_cast<int>(actualReal.size()));
        std::cout << "    " << std::setw(6) << "test#" << std::setw(11) << "actual"
                  << std::setw(13) << "moving avg" << std::setw(11) << "basic NN" << "\n";
        for (int i = 0; i < preview; ++i) {
            std::cout << "    " << std::setw(6) << i << std::setw(11) << actualReal[i]
                      << std::setw(13) << maReal[i] << std::setw(11) << nnReal[i] << "\n";
        }

        std::cout << "\n[7] Evaluation on the test set\n";
        MetricSet maMetrics = computeAllMetrics(actualReal, maReal);
        MetricSet nnMetrics = computeAllMetrics(actualReal, nnReal);
        std::cout << "    " << std::left << std::setw(14) << "Model" << std::right
                  << std::setw(10) << "MAE" << std::setw(10) << "RMSE" << std::setw(10) << "MAPE" << "\n";
        printMetricsRow("MovingAvg", maMetrics);
        printMetricsRow("BasicNN", nnMetrics);

        double change = (maMetrics.rmse - nnMetrics.rmse) / maMetrics.rmse * 100.0;
        if (change >= 0.0) {
            std::cout << "\n    Basic NN RMSE is " << change << "% lower than the Moving Average\n";
        } else {
            std::cout << "\n    Basic NN RMSE is " << -change << "% higher than the Moving Average\n";
        }

        printLine();
        std::cout << "Completed so far: data pipeline, Moving Average baseline, Basic NN with early stopping.\n";
        std::cout << "Next: LSTM model, extra baselines, replenishment logic.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
