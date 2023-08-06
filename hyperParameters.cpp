#include "hyperParameters.hpp"

// Constructor and variables
HyperParameters::HyperParameters(){
    activationFunction = new Relu();
    outputActivationFunction = new Softmax();
    costFunction = new CrossEntropy();
        
    initialLearningRate = 0.25;
    learnRateDecay = 0.075;
    batchSize = 64;
    epoch = 50;
}

// Copy Constructor to avoid segfault on destructor when HyperParameters is pass by value
HyperParameters::HyperParameters(const HyperParameters& other)
    : activationFunction(new Relu()),
    outputActivationFunction(new Softmax()),
    costFunction(new CrossEntropy()),
    initialLearningRate(other.initialLearningRate),
    learnRateDecay(other.learnRateDecay),
    batchSize(other.batchSize),
    epoch(other.epoch) {}

// Destructor
HyperParameters::~HyperParameters(){
    delete activationFunction;
    delete outputActivationFunction;
    delete costFunction;
}
