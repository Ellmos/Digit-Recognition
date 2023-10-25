#pragma once

#include "activationFunctions.hpp"
#include "costFunctions.hpp"


class HyperParameters {
public:
    ActivationFunction* activationFunction = new Relu();
    ActivationFunction* outputActivationFunction = new Softmax();;
    CostFunction* costFunction = new CrossEntropy();


    std::vector<size_t> layersSize = {784, 16, 10};
    double initialLearningRate = 0.025; 
    double learnRateDecay = 0.075;
    int batchSize = 1000;
    int epoch = 1;

public:
    HyperParameters(){};
    ~HyperParameters(){
        delete activationFunction;
        delete outputActivationFunction;
        delete costFunction;
    }
};

