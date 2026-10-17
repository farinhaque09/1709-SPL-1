# Time-Series Demand Forecasting Using LSTM

C++ implementation of demand forecasting using an LSTM network, benchmarked
against a Moving Average baseline and a basic feedforward Neural Network.
Built with LibTorch (the PyTorch C++ API) and CMake. Past sales are used as
the time series, and the forecast is compared with current stock to show
whether replenishment is needed.

## Requirements

Ubuntu (WSL or native Linux) with:

- g++ and CMake 3.18+ (`sudo apt install build-essential cmake`)
- LibTorch for Linux, C++, CPU, cxx11 ABI, from https://pytorch.org/get-started/locally/
  extracted to `~/libtorch`

## Build

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/libtorch
cmake --build build -j2
```

## Dataset

A sample dataset is included at `data/sales_data.csv`. To regenerate it:

```
cd data
g++ -std=c++17 -O2 -o generate_data generate_data.cpp
./generate_data
```

A real dataset can be used instead if it has the columns `date,sales,stock`.

## Run

```
./build/forecast --csv data/sales_data.csv --window 14 --epochs 150 --hidden 32
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | `data/sales_data.csv` | Path to the input CSV |
| `--window` | `14` | Number of past days used to predict the next day |
| `--epochs` | `150` | Training iterations |
| `--hidden` | `32` | Hidden layer size for the NN and the LSTM |
| `--train_ratio` | `0.8` | Fraction of data used for training |
| `--lr` | `0.01` | Learning rate of the Adam optimizer |

## Output

- Console: training loss, MAE / RMSE / MAPE of all three models, and the
  next-period forecast compared with current stock.
- `results/forecast_results.csv`: actual and predicted values of every model.
- `results/metrics_summary.csv`: MAE, RMSE and MAPE of every model.

## Project structure

```
include/   header files
src/       implementation files
data/      dataset and dataset generator
results/   created when the program runs
CMakeLists.txt
```
