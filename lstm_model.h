#pragma once

#include <torch/torch.h>

#include "training.h"

struct LSTMModelImpl : torch::nn::Module {
    LSTMModelImpl(int inputSize, int hiddenSize, int numLayers);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
    int hiddenSize_;
    int numLayers_;
};

TORCH_MODULE(LSTMModel);

TrainingHistory trainLSTM(LSTMModel& model,
                          const torch::Tensor& xTrain,
                          const torch::Tensor& yTrain,
                          const torch::Tensor& xVal,
                          const torch::Tensor& yVal,
                          int epochs,
                          double learningRate,
                          int patience);

std::vector<double> predictLSTM(LSTMModel& model, const torch::Tensor& xInput);
