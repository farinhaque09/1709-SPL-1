#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> splitLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: prepare_dataset <train.csv> <store> <item> <output.csv>\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string store = argv[2];
    const std::string item = argv[3];
    const std::string outputPath = argv[4];

    std::ifstream input(inputPath);
    if (!input.is_open()) {
        std::cerr << "Could not open " << inputPath << "\n";
        return 1;
    }

    std::string line;
    std::getline(input, line);
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line != "date,store,item,sales") {
        std::cerr << "Unexpected header, expected: date,store,item,sales\n";
        return 1;
    }

    std::ofstream output(outputPath);
    if (!output.is_open()) {
        std::cerr << "Could not open " << outputPath << " for writing\n";
        return 1;
    }
    output << "date,sales\n";

    long long scanned = 0;
    long long written = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        scanned++;

        std::vector<std::string> fields = splitLine(line);
        if (fields.size() != 4) {
            continue;
        }
        if (fields[1] == store && fields[2] == item) {
            output << fields[0] << "," << fields[3] << "\n";
            written++;
        }
    }

    std::cout << "Scanned " << scanned << " rows, wrote " << written
              << " rows for store " << store << ", item " << item << " to " << outputPath << "\n";
    if (written == 0) {
        std::cerr << "No rows matched, check the store and item numbers\n";
        return 1;
    }
    return 0;
}
