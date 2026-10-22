#include "scaler.h"

#include <algorithm>
#include <stdexcept>

void MinMaxScaler::fit(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::runtime_error("MinMaxScaler::fit called with empty data");
    }
    auto minIt = std::min_element(data.begin(), data.end());
    auto maxIt = std::max_element(data.begin(), data.end());
    min_ = *minIt;
    max_ = *maxIt;

    if (max_ - min_ < 1e-9) {
        max_ = min_ + 1.0;
    }
    fitted_ = true;
}

double MinMaxScaler::transformSingle(double value) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler used before fit()");
    }
    return (value - min_) / (max_ - min_);
}

double MinMaxScaler::inverseSingle(double scaledValue) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler used before fit()");
    }
    return scaledValue * (max_ - min_) + min_;
}

std::vector<double> MinMaxScaler::transform(const std::vector<double>& data) const {
    std::vector<double> out;
    out.reserve(data.size());
    for (double v : data) {
        out.push_back(transformSingle(v));
    }
    return out;
}

std::vector<double> MinMaxScaler::inverseTransform(const std::vector<double>& scaled) const {
    std::vector<double> out;
    out.reserve(scaled.size());
    for (double v : scaled) {
        out.push_back(inverseSingle(v));
    }
    return out;
}
