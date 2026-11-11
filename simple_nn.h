#pragma once

#include <torch/torch.h>

#include "training.h"

struct SimpleNNImpl : torch::nn::Module {
    SimpleNNImpl(int inputSize, int hiddenSize);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

TORCH_MODULE(SimpleNN);

TrainingHistory trainSimpleNN(SimpleNN& model,
                              const torch::Tensor& xTrain,
                              const torch::Tensor& yTrain,
                              const torch::Tensor& xVal,
                              const torch::Tensor& yVal,
                              int epochs,
                              double learningRate,
                              int patience);

std::vector<double> predictSimpleNN(SimpleNN& model, const torch::Tensor& xInput);
