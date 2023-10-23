#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <math.h>
#include <random>

#include "activationFunctions.hpp"
#include "json.hpp"
#include "data.hpp"

class Layer 
{
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
        //-----------------Constructor-------------------
        Layer(size_t nbrNodesIn, size_t nbrNodesOut, ActivationFunction* activationFunction);

        //-----------------Serialization-------------------
        nlohmann::json toJson() const;

        //-----------------Forward Pass-------------------
        std::vector<double> CalculateOutputs(std::vector<double>& inputs);

        //-----------------Backward Pass-------------------
        std::vector<double> UpdateGradient(Layer &oldLayer, std::vector<double> &oldNodeValues, 
                                           std::vector<double>& previousOutput, LayerGradient* layerGradient);
        void ApplyGradient(LayerGradient layerGradient, size_t batchSize, double learningRate);

        //-----------------Weights Initialization-------------------
        std::vector<double>  InitializeWeights(size_t nbrNodesOut, size_t nbrNodesIn);
};

