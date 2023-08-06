#pragma once

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <cstring>
#include <math.h>
#include <random>

#include "activationFunctions.h"

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
    std::vector<double> CalculateOutputs(std::vector<double> inputs);
    std::vector<double> UpdateGradient(Layer oldLayer, std::vector<double>  oldNodeValues, std::vector<double>  previousOutput);
    void ApplyGradient(double learningRate);
    std::vector<double>  InitializeWeights(size_t nbrNodesOut, size_t nbrNodesIn);
};
