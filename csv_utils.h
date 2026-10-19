#pragma once

#include <string>
#include <vector>

struct SalesRecord {
    std::string date;
    double sales = 0.0;
    double stock = 0.0;
};

std::vector<SalesRecord> loadSalesCSV(const std::string& path);
int cleanSalesRecords(std::vector<SalesRecord>& records);
