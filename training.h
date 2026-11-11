#pragma once

#include <torch/torch.h>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct TrainingHistory {
    std::vector<double> trainLoss;
    std::vector<double> validationLoss;
    int epochsRun = 0;
    int bestEpoch = 0;
    bool stoppedEarly = false;
};

template <typename Model>
TrainingHistory trainWithEarlyStopping(Model& model,
                                       const std::string& label,
                                       const torch::Tensor& xTrain,
                                       const torch::Tensor& yTrain,
                                       const torch::Tensor& xVal,
                                       const torch::Tensor& yVal,
                                       int epochs,
                                       double learningRate,
                                       int patience) {
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learningRate));

    TrainingHistory history;
    const bool useValidation = xVal.size(0) > 0;
    double bestLoss = std::numeric_limits<double>::infinity();
    int epochsWithoutImprovement = 0;
    std::vector<torch::Tensor> bestParameters;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        optimizer.zero_grad();

        torch::Tensor prediction = model->forward(xTrain);
        torch::Tensor loss = torch::mse_loss(prediction, yTrain);

        loss.backward();
        optimizer.step();

        double trainLossValue = loss.item<double>();
        history.trainLoss.push_back(trainLossValue);
        history.epochsRun = epoch;

        double monitoredLoss = trainLossValue;
        if (useValidation) {
            model->eval();
            torch::NoGradGuard noGrad;
            torch::Tensor validationPrediction = model->forward(xVal);
            monitoredLoss = torch::mse_loss(validationPrediction, yVal).item<double>();
            history.validationLoss.push_back(monitoredLoss);
        }

        if (monitoredLoss < bestLoss) {
            bestLoss = monitoredLoss;
            history.bestEpoch = epoch;
            epochsWithoutImprovement = 0;
            if (useValidation) {
                bestParameters.clear();
                for (const torch::Tensor& parameter : model->parameters()) {
                    bestParameters.push_back(parameter.detach().clone());
                }
            }
        } else {
            epochsWithoutImprovement++;
        }

        if (epoch % 20 == 0 || epoch == epochs) {
            std::cout << "  [" << label << "] epoch " << epoch << "/" << epochs
                      << " - training loss: " << trainLossValue;
            if (useValidation) {
                std::cout << " | validation loss: " << monitoredLoss;
            }
            std::cout << "\n";
        }

        if (useValidation && epochsWithoutImprovement >= patience) {
            history.stoppedEarly = true;
            std::cout << "  [" << label << "] early stopping at epoch " << epoch
                      << " (best epoch " << history.bestEpoch << ")\n";
            break;
        }
    }

    if (useValidation && !bestParameters.empty()) {
        torch::NoGradGuard noGrad;
        std::vector<torch::Tensor> parameters = model->parameters();
        for (size_t i = 0; i < parameters.size(); ++i) {
            parameters[i].copy_(bestParameters[i]);
        }
    }

    return history;
}
