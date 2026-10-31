#pragma once

#include <vector>

double computeMAE(const std::vector<double>& actual, const std::vector<double>& predicted);
double computeRMSE(const std::vector<double>& actual, const std::vector<double>& predicted);
double computeMAPE(const std::vector<double>& actual, const std::vector<double>& predicted);

struct MetricSet {
    double mae = 0.0;
    double rmse = 0.0;
    double mape = 0.0;
};

MetricSet computeAllMetrics(const std::vector<double>& actual, const std::vector<double>& predicted);
