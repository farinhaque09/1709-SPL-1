#include "simple_nn.h"

SimpleNNImpl::SimpleNNImpl(int inputSize, int hiddenSize) {
    fc1 = register_module("fc1", torch::nn::Linear(inputSize, hiddenSize));
    fc2 = register_module("fc2", torch::nn::Linear(hiddenSize, hiddenSize));
    fc3 = register_module("fc3", torch::nn::Linear(hiddenSize, 1));
}

torch::Tensor SimpleNNImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
}

TrainingHistory trainSimpleNN(SimpleNN& model,
                              const torch::Tensor& xTrain,
                              const torch::Tensor& yTrain,
                              const torch::Tensor& xVal,
                              const torch::Tensor& yVal,
                              int epochs,
                              double learningRate,
                              int patience) {
    return trainWithEarlyStopping(model, "Basic NN", xTrain, yTrain, xVal, yVal,
                                  epochs, learningRate, patience);
}

std::vector<double> predictSimpleNN(SimpleNN& model, const torch::Tensor& xInput) {
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
