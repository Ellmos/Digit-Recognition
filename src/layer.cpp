#include "layer.hpp"
#include <chrono>
#include <random>
#include <thread>


using json = nlohmann::json;
using namespace std;


//---------------------------Serialization--------------------------------

json Layer::toJson() const {
    return {
        {"weights", weights},
        {"biases", biases}
    };
}

//---------------------------Constructor--------------------------------
Layer::Layer(size_t nbrNodesIn, size_t nbrNodesOut, ActivationFunction* activationFunction){
    this->nbrNodesIn = nbrNodesIn;
    this->nbrNodesOut = nbrNodesOut;

    this->activationFunction = activationFunction;

    weights = InitializeWeights(nbrNodesOut, nbrNodesIn);
    biases.assign(nbrNodesOut, 0);

    gradientWeights.assign(nbrNodesOut * nbrNodesIn, 0);
    gradientBiases.assign(nbrNodesOut, 0);

    weightedSum.assign(nbrNodesOut, 0);
    outputs.assign(nbrNodesOut, 0);
}



//---------------------------Forward Pass--------------------------------
vector<double> Layer::CalculateOutputs(vector<double>& inputs){
    for (size_t nodesOut = 0; nodesOut < nbrNodesOut; nodesOut++){
        double iOutput = biases[nodesOut];
        for (size_t nodesIn = 0; nodesIn < nbrNodesIn; nodesIn++){
            iOutput += inputs[nodesIn] * weights[nodesOut * nbrNodesIn + nodesIn];
        }
        weightedSum[nodesOut] = iOutput;
    }


    for (size_t nodesOut = 0; nodesOut < nbrNodesOut; nodesOut++){
        outputs[nodesOut] = activationFunction->Function(weightedSum, nodesOut);
    }

    return outputs;
}


//---------------------------Backward Pass--------------------------------
vector<double> Layer::UpdateGradient(Layer &oldLayer, vector<double> &oldNodeValues, 
                                     vector<double> &previousOutput, LayerGradient* layerGradient) {

    // cout << "(-------------------------------------)\n";
    vector<double> newNodeValues(nbrNodesOut, 0);

    for (size_t nodesOut = 0; nodesOut < nbrNodesOut; nodesOut++) {
        double newNodeValue = 0;
        for (size_t oldNodesOut = 0; oldNodesOut < oldLayer.nbrNodesOut; oldNodesOut++)
            newNodeValue += oldLayer.weights[oldNodesOut * oldLayer.nbrNodesIn + nodesOut] * oldNodeValues[oldNodesOut];

        newNodeValue *= activationFunction->Derivative(weightedSum, nodesOut);
        newNodeValues[nodesOut] = newNodeValue;

        // gradientBiases[nodesOut] += newNodeValue;
        layerGradient->biases[nodesOut] += newNodeValue;
        // cout << newNodeValue << endl;
        // this_thread::sleep_for(chrono::milliseconds(200));
        for (size_t nodesIn = 0; nodesIn < nbrNodesIn; nodesIn++){
            // gradientWeights[nodesOut * nbrNodesIn + nodesIn] += previousOutput[nodesIn] * newNodeValue;
            layerGradient->weights[nodesOut * nbrNodesIn + nodesIn] += previousOutput[nodesIn] * newNodeValue;
        }

    }

    return newNodeValues;
}

void Layer::ApplyGradient(LayerGradient layerGradient, size_t batchSize, double learningRate) {
    for (size_t nodesOut = 0; nodesOut < nbrNodesOut; nodesOut++){
        for (size_t nodesIn = 0; nodesIn < nbrNodesIn; nodesIn++)
            weights[nodesOut * nbrNodesIn + nodesIn] -= layerGradient.weights[nodesOut * nbrNodesIn + nodesIn] / batchSize * learningRate;

        biases[nodesOut] -= layerGradient.biases[nodesOut] / batchSize * learningRate;
    }
    
    gradientWeights.assign(nbrNodesOut * nbrNodesIn, 0);
    gradientBiases.assign(nbrNodesOut, 0);
}



//---------------------------Weights Initialization--------------------------------
std::default_random_engine rnd{std::random_device{}()};

double RandomNormalDistribution(double mean, double standardDeviation){
    std::normal_distribution<double> dist(mean, standardDeviation);
    return dist(rnd);
}


vector<double> Layer::InitializeWeights(size_t nbrNodesOut, size_t nbrNodesIn){
    vector<double> weights(nbrNodesOut * nbrNodesIn);
    for (size_t nodesOut = 0; nodesOut < nbrNodesOut; nodesOut++){
        for (size_t nodesIn = 0; nodesIn < nbrNodesIn; nodesIn++){
            weights[nodesOut * nbrNodesIn + nodesIn] = RandomNormalDistribution(0, 1);
        }
    }

    return weights;
}

