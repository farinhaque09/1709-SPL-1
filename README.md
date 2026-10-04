# Time-Series Demand Forecasting Using LSTM

C++ implementation of demand forecasting using an LSTM network,
benchmarked against a Moving Average baseline and a basic feedforward
Neural Network, using LibTorch (PyTorch's C++ API).

## Requirements

- CMake 3.18+
- A C++17 compiler
- LibTorch — download from https://pytorch.org/get-started/locally/
  (select "LibTorch" under Package, pick your OS, and either CPU or
  CUDA depending on your machine)

## Build

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
cmake --build build --config Release
```

## Generate the sample dataset (optional — one is already included)

```
cd data
g++ -std=c++17 -O2 -o generate_data generate_data.cpp
./generate_data
```

To use a real dataset instead, replace `data/sales_data.csv` with any
CSV that has the same three columns in the same order: `date,sales,stock`.

## Run

```
./build/forecast --csv data/sales_data.csv --window 14 --epochs 150 --hidden 32
```

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | `data/sales_data.csv` | Path to the input CSV |
| `--window` | `14` | How many past days are used to predict the next day |
| `--epochs` | `150` | Training iterations |
| `--hidden` | `32` | Hidden layer size for both the NN and LSTM |
| `--train_ratio` | `0.8` | Fraction of data used for training (rest is test) |
| `--lr` | `0.01` | Learning rate for the Adam optimizer |

## Output

- Console: training progress, MAE/RMSE/MAPE for all three models, and a
  next-period demand forecast compared against current stock.
- `results/forecast_results.csv` — per-test-point predictions from all
  three models alongside the actual value.
- `results/metrics_summary.csv` — MAE/RMSE/MAPE per model.

## Project structure

```
include/      Header files (.h)
src/          Implementation files (.cpp)
data/         Dataset + generator script
results/      Created automatically when the program runs
CMakeLists.txt
```
