#include "lstm_model.h"

LSTMModelImpl::LSTMModelImpl(int inputSize, int hiddenSize, int numLayers)
    : hiddenSize_(hiddenSize), numLayers_(numLayers) {
    auto options = torch::nn::LSTMOptions(inputSize, hiddenSize)
                       .num_layers(numLayers)
                       .batch_first(true);
    lstm = register_module("lstm", torch::nn::LSTM(options));
    fc = register_module("fc", torch::nn::Linear(hiddenSize, 1));
}

torch::Tensor LSTMModelImpl::forward(torch::Tensor x) {
    auto lstmOut = lstm->forward(x);
    torch::Tensor output = std::get<0>(lstmOut);
    torch::Tensor lastStep = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
    torch::Tensor prediction = fc->forward(lastStep);
    return prediction;
}

TrainingHistory trainLSTM(LSTMModel& model,
                          const torch::Tensor& xTrain,
                          const torch::Tensor& yTrain,
                          const torch::Tensor& xVal,
                          const torch::Tensor& yVal,
                          int epochs,
                          double learningRate,
                          int patience) {
    return trainWithEarlyStopping(model, "LSTM", xTrain, yTrain, xVal, yVal,
                                  epochs, learningRate, patience);
}

std::vector<double> predictLSTM(LSTMModel& model, const torch::Tensor& xInput) {
    model->eval();
    torch::NoGradGuard noGrad;

    torch::Tensor output = model->forward(xInput).squeeze(1);
    output = output.contiguous();

    std::vector<double> predictions;
    predictions.reserve(output.numel());
    auto accessor = output.accessor<float, 1>();
    for (int64_t i = 0; i < output.size(0); ++i) {
        predictions.push_back(static_cast<double>(accessor[i]));
    }
    return predictions;
}
