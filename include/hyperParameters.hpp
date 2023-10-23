#pragma once
#include "costFunctions.hpp"
#include "activationFunctions.hpp"


class HyperParameters {
public:
    ActivationFunction* activationFunction = new Relu();
    ActivationFunction* outputActivationFunction = new Softmax();;
    CostFunction* costFunction = new CrossEntropy();


    std::vector<size_t> layersSize = {784, 16, 10};
    double initialLearningRate = 0.25; 
    double learnRateDecay = 0.075;
    int batchSize = 10000;
    int epoch = 5;

public:
    HyperParameters(){};
    ~HyperParameters(){
        delete activationFunction;
        delete outputActivationFunction;
        delete costFunction;
    }
};

