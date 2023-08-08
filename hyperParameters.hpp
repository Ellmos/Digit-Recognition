#pragma once
#include "costFunctions.hpp"
#include "activationFunctions.hpp"


class HyperParameters {
public:
    ActivationFunction* activationFunction = new Relu();
    ActivationFunction* outputActivationFunction = new Softmax();;
    CostFunction* costFunction = new CrossEntropy();

    double initialLearningRate = 0.25; 
    double learnRateDecay = 0.075;
    int batchSize = 64;
    int epoch = 1;

public:
    HyperParameters(){};
    ~HyperParameters(){
        delete activationFunction;
        delete outputActivationFunction;
        delete costFunction;
    }
};

