#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <iomanip>

struct Row {
    std::string date;
    double sales;
    double stock;
};

std::string formatDate(int year, int month, int day) {
    char buffer[11];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d", year, month, day);
    return std::string(buffer);
}

int daysInMonth(int year, int month) {
    static const int lengths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    bool isLeap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    if (month == 2 && isLeap) return 29;
    return lengths[month - 1];
}

int weekdayFromIndex(int dayIndex) {
    return (dayIndex + 2) % 7;
}

int main() {
    const int numDays = 365;
    const double baseDemand = 120.0;
    const double trendPerDay = 0.05;
    const double weekendBoost = 25.0;
    const double noiseStd = 8.0;
    const int restockEveryNDays = 14;
    const double restockAmount = 2300.0;
    double stock = 1600.0;

    std::mt19937 rng(42);
    std::normal_distribution<double> noiseDist(0.0, noiseStd);

    std::vector<Row> rows;
    rows.reserve(numDays);

    int year = 2025, month = 1, day = 1;

    for (int i = 0; i < numDays; ++i) {
        int weekday = weekdayFromIndex(i);
        double trend = trendPerDay * i;
        double seasonal = (weekday == 5 || weekday == 6) ? weekendBoost : 0.0;
        double monthlyWave = 10.0 * std::sin(2.0 * M_PI * i / 30.0);
        double noise = noiseDist(rng);

        double sales = baseDemand + trend + seasonal + monthlyWave + noise;
        if (sales < 0.0) sales = 0.0;
        sales = std::round(sales * 10.0) / 10.0;

        if (i % restockEveryNDays == 0 && i != 0) {
            stock += restockAmount;
        }
        stock -= sales;
        if (stock < 0.0) stock = 0.0;
        stock = std::round(stock * 10.0) / 10.0;

        Row row;
        row.date = formatDate(year, month, day);
        row.sales = sales;
        row.stock = stock;
        rows.push_back(row);

        day++;
        if (day > daysInMonth(year, month)) {
            day = 1;
            month++;
            if (month > 12) {
                month = 1;
                year++;
            }
        }
    }

    std::ofstream file("sales_data.csv");
    if (!file.is_open()) {
        std::cerr << "Could not open sales_data.csv for writing\n";
        return 1;
    }

    file << "date,sales,stock\n";
    file << std::fixed << std::setprecision(1);
    for (const Row& r : rows) {
        file << r.date << "," << r.sales << "," << r.stock << "\n";
    }

    std::cout << "Wrote " << rows.size() << " rows to sales_data.csv\n";
    return 0;
}
