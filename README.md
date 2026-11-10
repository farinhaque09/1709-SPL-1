# Time-Series Demand Forecasting Using LSTM

C++ implementation of demand forecasting using an LSTM network, benchmarked
against five baselines: Naive, Seasonal Naive, Exponential Smoothing, Moving
Average and a basic feedforward Neural Network. Built with LibTorch (the
PyTorch C++ API) and CMake. Past sales are used as the time series, and the
forecast is compared with current stock to show whether replenishment is
needed.

## Requirements

Ubuntu (WSL or native Linux) with:

- g++ 10 or newer and CMake 3.18+ (`sudo apt install build-essential cmake`)
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
g++ -std=c++20 -O2 -o generate_data generate_data.cpp
./generate_data
```

A real dataset can be used instead if it has the columns `date,sales,stock`.

## Run

```
./build/forecast --csv data/sales_data.csv --window 14 --epochs 150 --hidden 32
./build/forecast --help
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | `data/sales_data.csv` | Path to the input CSV |
| `--window` | `14` | Past days used to predict the next day (7 to 365) |
| `--epochs` | `150` | Maximum training epochs |
| `--hidden` | `32` | Hidden layer size for the NN and the LSTM |
| `--train_ratio` | `0.8` | Fraction of windows used for training |
| `--val_ratio` | `0.1` | Fraction of training windows kept for early stopping (0 disables it) |
| `--patience` | `20` | Epochs without improvement before training stops |
| `--lr` | `0.01` | Learning rate of the Adam optimizer |
| `--alpha` | `0.3` | Smoothing factor of the exponential smoothing baseline |

## Tests

The unit tests cover every module that does not need LibTorch:

```
cmake --build build --target run_tests
./build/run_tests
```

## Output

- Console: data summary, training loss with early stopping, MAE / RMSE / MAPE
  of all six models, the best model by RMSE, and the next-period forecast
  compared with current stock.
- `results/forecast_results.csv`: actual and predicted values of every model.
- `results/metrics_summary.csv`: MAE, RMSE and MAPE of every model.

## Project structure

```
include/   header files
src/       implementation files
tests/     unit tests
data/      dataset and dataset generator
results/   created when the program runs
CMakeLists.txt
```

## Sample output

Console output has this shape (numbers depend on the run):

```
Model                    MAE      RMSE      MAPE
naive                <value>   <value>   <value>%
seasonal_naive       <value>   <value>   <value>%
exp_smoothing        <value>   <value>   <value>%
moving_average       <value>   <value>   <value>%
basic_nn             <value>   <value>   <value>%
lstm                 <value>   <value>   <value>%

Best model by RMSE: <name>

Next-period forecasted demand: <value> units
Current stock: <value> units
Replenishment needed: order approximately <value> more units
```
