#include "csv_utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace {

std::vector<std::string> splitLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

bool tryParseDouble(const std::string& text, double& out) {
    try {
        size_t idx = 0;
        out = std::stod(text, &idx);
        return idx == text.size();
    } catch (...) {
        return false;
    }
}

}

std::vector<SalesRecord> loadSalesCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }

    std::vector<SalesRecord> records;
    std::string line;

    bool isHeader = true;
    int lineNumber = 0;
    int skippedRows = 0;

    while (std::getline(file, line)) {
        lineNumber++;

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) {
            continue;
        }

        if (isHeader) {
            isHeader = false;
            continue;
        }

        std::vector<std::string> fields = splitLine(line);
        if (fields.size() != 3) {
            std::cerr << "[warn] line " << lineNumber
                      << " does not have 3 fields, skipping: " << line << "\n";
            skippedRows++;
            continue;
        }

        SalesRecord record;
        record.date = fields[0];

        double sales = 0.0, stock = 0.0;
        bool salesOk = tryParseDouble(fields[1], sales);
        bool stockOk = tryParseDouble(fields[2], stock);

        if (!salesOk || !stockOk) {
            std::cerr << "[warn] line " << lineNumber
                      << " has a non-numeric sales/stock value, skipping: " << line << "\n";
            skippedRows++;
            continue;
        }

        record.sales = sales;
        record.stock = stock;
        records.push_back(record);
    }

    if (skippedRows > 0) {
        std::cerr << "[info] skipped " << skippedRows
                  << " malformed row(s) while loading " << path << "\n";
    }

    if (records.empty()) {
        throw std::runtime_error("No valid rows were loaded from: " + path);
    }

    return records;
}

int cleanSalesRecords(std::vector<SalesRecord>& records) {
    std::vector<SalesRecord> cleaned;
    cleaned.reserve(records.size());
    int dropped = 0;

    for (const SalesRecord& r : records) {
        bool salesFinite = std::isfinite(r.sales);
        bool stockFinite = std::isfinite(r.stock);

        if (!salesFinite || !stockFinite || r.sales < 0.0 || r.stock < 0.0) {
            dropped++;
            continue;
        }
        cleaned.push_back(r);
    }

    records = std::move(cleaned);
    return dropped;
}
