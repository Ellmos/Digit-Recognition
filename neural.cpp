#include "neural.h"
#include <cstdlib>
using namespace std;

Neural::Neural(size_t layerSizes[], size_t nbrLayers, HyperParameters* hyperParameters) {
    this->layersSize = layerSizes;
    this->nbrLayers = nbrLayers;
    this->costFunction = hyperParameters->costFunction;
    // this->layers.reserve(nbrLayers);


    for (size_t i = 0; i < nbrLayers; i++)
        layers.push_back(Layer(layersSize[i], layersSize[i + 1], hyperParameters->activationFunction));
    layers[nbrLayers - 1].activationFunction = hyperParameters->outputActivationFunction;
}

//---------------------------Back Propagation--------------------------------
vector<double> Neural::CalculateOutputs(vector<double> inputs){
    for (size_t i = 0; i < nbrLayers; i++) {
        inputs = layers[i].CalculateOutputs(inputs);
    }
    return inputs;
}

void Neural::FeedBatch(Batch batch, size_t batchSize, double learningRate) {
    for (size_t i = 0; i < batchSize; i++) {
        Data dataPoint = batch.dataPoints[i];
        CalculateOutputs(dataPoint.inputs);

        Layer* outputLayer = &layers[nbrLayers - 1];
        vector<double> previousOutputs = nbrLayers >= 2 ? layers[nbrLayers - 2].outputs : dataPoint.inputs;

        // Compute the nodes values of the output layer and update its gradients
        vector<double> nodeValues(outputLayer->nbrNodesOut);
        for (size_t nodesOut = 0; nodesOut < outputLayer->nbrNodesOut; nodesOut++) {

            double activationDerivative = outputLayer->activationFunction->Derivative(outputLayer->weightedSum, nodesOut);
            double costDerivative = costFunction->Derivative(outputLayer->outputs[nodesOut], dataPoint.targets[nodesOut]);
            double currentNodeValue = activationDerivative * costDerivative;
            nodeValues[nodesOut] = currentNodeValue;



            for (size_t nodesIn = 0; nodesIn < outputLayer->nbrNodesIn; nodesIn++)
                outputLayer->gradientWeights[nodesOut * outputLayer->nbrNodesIn + nodesIn] += previousOutputs[nodesIn] * currentNodeValue;



        }

        // Go back through the layers, compute the corresponding node values and
        // update the gradient at the same time
        for (size_t i = 2; i < nbrLayers + 1; i++) {
            previousOutputs = i < nbrLayers ? layers[nbrLayers - i - 1].outputs : dataPoint.inputs;
            Layer* currentLayer = &layers[nbrLayers - i];
           
            nodeValues = currentLayer->UpdateGradient(layers[nbrLayers - i + 1], nodeValues, previousOutputs);
        }

        for (size_t i = 0; i < nbrLayers; i++) {
            Layer* layer = &layers[i];
            layer->ApplyGradient(learningRate / batch.batchSize);
        }
    }
}

void Neural::Learn(DataSet trainDataSet, DataSet testDataSet, HyperParameters hp) {
    int nbrBatch = trainDataSet.nbrBatch;
    int printBatch = nbrBatch / 10 != 0 ? nbrBatch / 10 : 1;

    vector<double> accuracyTrain(hp.epoch);
    vector<double> accuracyTest(hp.epoch);

    auto rng = default_random_engine{};

    cout << "------------------Learning-----------------------\n";
    for (int currentEpoch = 0; currentEpoch < hp.epoch; currentEpoch++) {
        cout << "--Epoch " << currentEpoch + 1 << " out of " << hp.epoch << "--\n";
        double learningRate = hp.initialLearningRate * (1 / (1 + hp.learnRateDecay * currentEpoch));

        shuffle(begin(trainDataSet.batches), end(trainDataSet.batches), rng);
        for (int i = 0; i < nbrBatch; i++) {
            if (i % printBatch == 0)
                cout << "Batch " << i << " out of " << nbrBatch << "\n";
            FeedBatch(trainDataSet.batches[i], trainDataSet.batches[i].batchSize, learningRate);
        }

        // Used to visualize progression of dataSet and check overfitting
        accuracyTrain[currentEpoch] = DataSetAccuracy(trainDataSet);
        accuracyTest[currentEpoch] = DataSetAccuracy(testDataSet);
    }


    cout << "Accuracy on training DataSet: " << accuracyTrain[hp.epoch - 1] << "%\n";
    cout << "Accuracy on test DataSet: " << accuracyTest[hp.epoch - 1] << "%\n";

    // ToJson("NewSave");
}

//-----------------Cost---------------
double Neural::DataPointCost(Data dataPoint) {
    vector<double> outputs = CalculateOutputs(dataPoint.inputs);
    return costFunction->Function(outputs, dataPoint.targets);
}
double Neural::BatchCost(Batch batch) {
    double cost = 0;
    for (size_t i = 0; i < batch.batchSize; i++) {
        cost += DataPointCost(batch.dataPoints[i]);
    }

    return cost;
}
double Neural::DataSetCost(DataSet dataSet) {
    double cost = 0;
    for (size_t i = 0; i < dataSet.nbrBatch; i++) {
        cost += BatchCost(dataSet.batches[i]);
    }

    return cost;
}

//-----------------Accuracy---------------
double Neural::BatchAccuracy(Batch batch) {
    double nbrGood = 0;
    for (size_t i = 0; i < batch.batchSize; i++) {
        if (Classify(batch.dataPoints[i].inputs) == GetMaxIndex(batch.dataPoints[i].targets))
            nbrGood++;
    }

    return nbrGood * 100 / batch.batchSize;
}
double Neural::DataSetAccuracy(DataSet dataSet) {
    double averageAccuracy = 0;
    for (size_t i = 0; i < dataSet.nbrBatch; i++) {
        averageAccuracy += BatchAccuracy(dataSet.batches[i]);
    }
    
    cout << "Accuracy: " << averageAccuracy / dataSet.nbrBatch << endl;
    return averageAccuracy / dataSet.nbrBatch;
}

//-----------------Classify---------------
int Neural::Classify(vector<double> inputs) {
    vector<double> outputs = CalculateOutputs(inputs);
    return GetMaxIndex(outputs);
}
int Neural::GetMaxIndex(vector<double> outputs) {
    double max = outputs[0];
    int index = 0;
    for (size_t i = 1; i < outputs.size(); i++) {
        if (outputs[i] > max) {
            max = outputs[i];
            index = i;
        }
    }
    return index;
}
