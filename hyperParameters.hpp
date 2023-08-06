#pragma once
#include "costFunctions.hpp"
#include "activationFunctions.hpp"
#include <cstdio>

class HyperParameters {
public:
    ActivationFunction* activationFunction;
    ActivationFunction* outputActivationFunction;
    CostFunction* costFunction;

    double initialLearningRate;
    double learnRateDecay;
    int batchSize;
    int epoch;

    HyperParameters();

    // Copy Constructor to avoid segfault on destructor when HyperParameters is pass by value
    HyperParameters(const HyperParameters& other);

    ~HyperParameters();
};

