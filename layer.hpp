#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <math.h>
#include <random>

#include "activationFunctions.hpp"
#include "library/json.hpp"

class Layer {
public:
    size_t nbrNodesIn;
    size_t nbrNodesOut;

    std::vector<double> weights;
    std::vector<double> biases;

    std::vector<double> gradientWeights;
    std::vector<double> gradientBiases;

    std::vector<double> weightedSum;
    std::vector<double> outputs;

    ActivationFunction* activationFunction;


public:
    Layer(size_t nbrNodesIn, size_t nbrNodesOut, ActivationFunction* activationFunction);

    nlohmann::json toJson() const;

    std::vector<double> CalculateOutputs(const std::vector<double>& inputs);
    std::vector<double> UpdateGradient(const Layer& oldLayer, const std::vector<double>& oldNodeValues, const std::vector<double>& previousOutput);
    void ApplyGradient(double learningRate);
    std::vector<double>  InitializeWeights(size_t nbrNodesOut, size_t nbrNodesIn);
};
