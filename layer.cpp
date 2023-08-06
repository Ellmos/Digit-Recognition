#include "layer.h"

using namespace std;

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


vector<double> Layer::CalculateOutputs(vector<double> inputs){
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


vector<double> Layer::UpdateGradient(Layer oldLayer, vector<double> oldNodeValues, vector<double>  previousOutput) {
    vector<double> newNodeValues(nbrNodesOut, 0);

    for (size_t nodesOut = 0; nodesOut < nbrNodesOut; nodesOut++)
        {
            double newNodeValue = 0;
            for (size_t oldNodesOut = 0; oldNodesOut < oldLayer.nbrNodesOut; oldNodesOut++)
                newNodeValue += oldLayer.weights[oldNodesOut * oldLayer.nbrNodesIn + nodesOut] * oldNodeValues[oldNodesOut];

            newNodeValue *= activationFunction->Derivative(weightedSum, nodesOut);
            newNodeValues[nodesOut] = newNodeValue;

            gradientBiases[nodesOut] += newNodeValue;
            for (size_t nodesIn = 0; nodesIn < nbrNodesIn; nodesIn++)
                gradientWeights[nodesOut * nbrNodesIn + nodesIn] += previousOutput[nodesIn] * newNodeValue;
        }

    return newNodeValues;
}


void Layer::ApplyGradient(double learningRate) {
    for (size_t nodesOut = 0; nodesOut < nbrNodesOut; nodesOut++){
        for (size_t nodesIn = 0; nodesIn < nbrNodesIn; nodesIn++)
            weights[nodesOut * nbrNodesIn + nodesIn] -= gradientWeights[nodesOut * nbrNodesIn + nodesIn] * learningRate;

        biases[nodesOut] -= gradientBiases[nodesOut] * learningRate;
    }
    
    gradientWeights.assign(nbrNodesOut * nbrNodesIn, 0);
    gradientBiases.assign(nbrNodesOut, 0);
}



//-----------------Weights Initialization------------------//
double RandomDouble(){
    double lower = 0;
    double upper = 1;

    default_random_engine rnd{random_device{}()};
    uniform_real_distribution<double> dist(lower, upper);

    return dist(rnd);
}

double RandomNormalDistribution(double mean, double standardDeviation){
    double lower = 0;
    double upper = 1;

    default_random_engine rnd{random_device{}()};
    uniform_real_distribution<double> dist(lower, upper);

    double x1 = 1 - dist(rnd);
    double x2 = 1 - dist(rnd);

    double y1 = sqrt(-2.0 * log(x1)) * cos(2.0 * M_PI * x2);
    return y1 * standardDeviation + mean;
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

