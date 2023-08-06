#pragma once
#include "costFunctions.h"
#include "activationFunctions.h"
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

