#pragma once

#include <vector>

class MinMaxScaler {
public:
    void fit(const std::vector<double>& data);
    std::vector<double> transform(const std::vector<double>& data) const;
    std::vector<double> inverseTransform(const std::vector<double>& scaled) const;
    double transformSingle(double value) const;
    double inverseSingle(double scaledValue) const;
    double getMin() const { return min_; }
    double getMax() const { return max_; }

private:
    double min_ = 0.0;
    double max_ = 1.0;
    bool fitted_ = false;
};
