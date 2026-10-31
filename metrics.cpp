#include "metrics.h"

#include <cmath>
#include <stdexcept>

namespace {
void checkSameSize(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        throw std::runtime_error("Metric inputs must be non-empty and the same size");
    }
}
}

double computeMAE(const std::vector<double>& actual, const std::vector<double>& predicted) {
    checkSameSize(actual, predicted);
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        sum += std::fabs(actual[i] - predicted[i]);
    }
    return sum / static_cast<double>(actual.size());
}

double computeRMSE(const std::vector<double>& actual, const std::vector<double>& predicted) {
    checkSameSize(actual, predicted);
    double sumSquares = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = actual[i] - predicted[i];
        sumSquares += diff * diff;
    }
    return std::sqrt(sumSquares / static_cast<double>(actual.size()));
}

double computeMAPE(const std::vector<double>& actual, const std::vector<double>& predicted) {
    checkSameSize(actual, predicted);
    double sum = 0.0;
    int countedTerms = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i]) < 1e-9) {
            continue;
        }
        sum += std::fabs((actual[i] - predicted[i]) / actual[i]);
        countedTerms++;
    }
    if (countedTerms == 0) {
        return 0.0;
    }
    return (sum / countedTerms) * 100.0;
}

MetricSet computeAllMetrics(const std::vector<double>& actual, const std::vector<double>& predicted) {
    MetricSet metrics;
    metrics.mae = computeMAE(actual, predicted);
    metrics.rmse = computeRMSE(actual, predicted);
    metrics.mape = computeMAPE(actual, predicted);
    return metrics;
}
