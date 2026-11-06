#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "csv_utils.h"
#include "scaler.h"
#include "windowing.h"
#include "metrics.h"
#include "moving_average.h"

struct Config {
    std::string csvPath = "data/sales_data.csv";
    int windowSize = 14;
    double trainRatio = 0.8;
    int previewRows = 5;
};

Config parseArgs(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (i + 1 >= argc) {
            break;
        }
        if (arg == "--csv") {
            config.csvPath = argv[++i];
        } else if (arg == "--window") {
            config.windowSize = std::stoi(argv[++i]);
        } else if (arg == "--train_ratio") {
            config.trainRatio = std::stod(argv[++i]);
        } else if (arg == "--preview") {
            config.previewRows = std::stoi(argv[++i]);
        }
    }
    return config;
}

void printLine() {
    std::cout << "------------------------------------------------------------\n";
}

void printWindowRow(const std::vector<double>& window, double target, const MinMaxScaler& scaler) {
    std::cout << "    input (last " << window.size() << " days, scaled): [";
    int shown = static_cast<int>(window.size()) < 4 ? static_cast<int>(window.size()) : 4;
    for (int i = 0; i < shown; ++i) {
        std::cout << window[i] << (i + 1 < shown ? ", " : ", ...");
    }
    std::cout << "]\n";
    std::cout << "    next-day target: " << target << " (scaled) = "
              << scaler.inverseSingle(target) << " units\n";
}

int main(int argc, char** argv) {
    Config config = parseArgs(argc, argv);
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Time-Series Demand Forecasting Using LSTM - progress demo\n";
    printLine();

    std::cout << "[1] Loading data from " << config.csvPath << "\n";
    std::vector<SalesRecord> records = loadSalesCSV(config.csvPath);
    std::cout << "    rows loaded: " << records.size() << "\n";
    std::cout << "    first row: " << records.front().date << " | sales "
              << records.front().sales << " | stock " << records.front().stock << "\n";
    std::cout << "    last row : " << records.back().date << " | sales "
              << records.back().sales << " | stock " << records.back().stock << "\n";

    std::cout << "\n[2] Cleaning data\n";
    int dropped = cleanSalesRecords(records);
    std::cout << "    rows dropped: " << dropped << " | rows remaining: " << records.size() << "\n";

    std::vector<double> salesSeries;
    salesSeries.reserve(records.size());
    for (const SalesRecord& r : records) {
        salesSeries.push_back(r.sales);
    }

    std::cout << "\n[3] Min-max scaling (fitted on the training portion only)\n";
    int trainCutoff = static_cast<int>(salesSeries.size() * config.trainRatio);
    std::vector<double> trainPortionRaw(salesSeries.begin(), salesSeries.begin() + trainCutoff);
    MinMaxScaler scaler;
    scaler.fit(trainPortionRaw);
    std::vector<double> scaledSeries = scaler.transform(salesSeries);
    std::cout << "    training min: " << scaler.getMin() << " | training max: " << scaler.getMax() << "\n";
    std::cout << "    example: " << salesSeries.front() << " units -> "
              << scaledSeries.front() << " (scaled) -> "
              << scaler.inverseSingle(scaledSeries.front()) << " units (restored)\n";

    std::cout << "\n[4] Sliding windows (window size = " << config.windowSize << " days)\n";
    WindowedDataset windowed = createSlidingWindows(scaledSeries, config.windowSize);
    std::cout << "    total windows: " << windowed.X.size() << "\n";
    printWindowRow(windowed.X.front(), windowed.y.front(), scaler);

    std::cout << "\n[5] Time-ordered train/test split (" << static_cast<int>(config.trainRatio * 100)
              << "% / " << static_cast<int>((1.0 - config.trainRatio) * 100 + 0.5) << "%)\n";
    TrainTestSplit split = trainTestSplit(windowed, config.trainRatio);
    std::cout << "    training windows: " << split.train.X.size()
              << " | testing windows: " << split.test.X.size() << "\n";

    std::cout << "\n[6] Moving Average baseline on the test windows\n";
    std::vector<double> maScaled = MovingAverageModel::predict(split.test.X);
    std::vector<double> actualReal = scaler.inverseTransform(split.test.y);
    std::vector<double> maReal = scaler.inverseTransform(maScaled);

    int preview = config.previewRows;
    if (preview > static_cast<int>(actualReal.size())) {
        preview = static_cast<int>(actualReal.size());
    }
    std::cout << "    " << std::setw(6) << "test#" << std::setw(12) << "actual"
              << std::setw(14) << "moving avg" << std::setw(10) << "error" << "\n";
    for (int i = 0; i < preview; ++i) {
        double err = maReal[i] - actualReal[i];
        std::cout << "    " << std::setw(6) << i << std::setw(12) << actualReal[i]
                  << std::setw(14) << maReal[i] << std::setw(10) << err << "\n";
    }

    std::cout << "\n[7] Evaluation metrics\n";
    MetricSet ma = computeAllMetrics(actualReal, maReal);
    std::cout << "    Model\t\tMAE\tRMSE\tMAPE\n";
    std::cout << "    MovingAvg\t\t" << ma.mae << "\t" << ma.rmse << "\t" << ma.mape << "%\n";

    printLine();
    std::cout << "Completed so far: data loading, cleaning, scaling, windowing,\n";
    std::cout << "train/test split, Moving Average baseline, MAE/RMSE/MAPE.\n";
    std::cout << "Next: Basic NN baseline, LSTM model, replenishment logic.\n";

    return 0;
}
