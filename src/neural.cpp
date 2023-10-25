#include "neural.hpp"

#include <cstdlib>
#include <filesystem>
#include <future>

using namespace std;
using json = nlohmann::json;


//---------------------------Serialization--------------------------------
json Neural::Serialize() const {
    nlohmann::json jLayers;
    for (const auto& layer : layers) {
        jLayers.push_back(layer.toJson());
    }

    return {
        {"nbrLayers", nbrLayers},
        {"layersSize", layersSize},
        {"layers", jLayers}
    };
}

void Neural::ToJson(std::string fileName){
    filesystem::path path = filesystem::current_path();
    ofstream file (path.string() + "/src/data/saves/" + fileName + ".json");
    if (!file.is_open())
        throw runtime_error("Neural::ToJson: Failed to create save file");

    file << this->Serialize().dump();

    file.close();
}


//-----------------Deserialization---------------
Neural NeuralFromJson(std::string fileName, HyperParameters& hyperParameters){
    // Read File
    filesystem::path path = filesystem::current_path();
    string filePath = path.string() + "/src/data/saves/" + fileName + ".json";
    ifstream file(filePath);
    if (!file.is_open()) 
        throw runtime_error("NeuralFromJson: Failed to open \"" + filePath + "\"");

    std::string jsonString((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Parse JSON
    json parsedJson;
    try {
        parsedJson = json::parse(jsonString);
    } catch (const json::parse_error& e) {
        throw runtime_error("NeuralFromJson: Failed to parse JSON file");
    }

    // Create neural class
    size_t nbrLayers = parsedJson["nbrLayers"];
    vector<size_t> layersSize = vector<size_t>(nbrLayers+1);
    for (size_t i = 0; i <= nbrLayers; i++) {
        layersSize[i] = parsedJson["layersSize"][i];
    }
    hyperParameters.layersSize = layersSize;

    Neural neural = Neural(hyperParameters);
    for (size_t i = 0; i < neural.nbrLayers; i++) {
        vector<double> weights = parsedJson["layers"][i]["weights"];
        vector<double> biases = parsedJson["layers"][i]["biases"];

        neural.layers[i].weights = weights;
        neural.layers[i].biases = biases;
    }

    file.close();
    return neural;
}




//---------------------------Neural class --------------------------------

Neural::Neural(HyperParameters& hyperParameters) {
    this->layersSize = hyperParameters.layersSize;
    this->nbrLayers = layersSize.size() - 1;
    this->costFunction = hyperParameters.costFunction;

    for (size_t i = 0; i < nbrLayers; i++)
        layers.push_back(Layer(layersSize[i], layersSize[i + 1], hyperParameters.activationFunction));
    layers[nbrLayers - 1].activationFunction = hyperParameters.outputActivationFunction;
}



//---------------------------BackPropagation--------------------------------
void Neural::Learn(DataSet& trainDataSet, DataSet& testDataSet, HyperParameters& hp) {
    int nbrBatch = trainDataSet.nbrBatch;
    int printBatch = nbrBatch / 10 != 0 ? nbrBatch / 10 : 1;

    vector<double> accuracyTrain(hp.epoch);
    vector<double> accuracyTest(hp.epoch);

    auto rng = default_random_engine{};

    for (int currentEpoch = 0; currentEpoch < hp.epoch; currentEpoch++) {

        cout << "--Epoch " << currentEpoch + 1 << " out of " << hp.epoch << "--\n";

        // Shuffle DataSet to improve learning
        shuffle(begin(trainDataSet.batches), end(trainDataSet.batches), rng);

        //-----------------------Single Thread------------------------
        // for (int i = 0; i < nbrBatch; i++) {
        //     FeedBatch(trainDataSet.batches[i], hp.layersSize, learningRate);
        //     if (i % printBatch == 0)
        //         cout << "Batch " << i << " out of " << nbrBatch << "\n";
        //
        // }

        //-----------------------Multi Thread------------------------
        // Run every batches on a different thread
        vector<future<BatchGradient>> futureBatchGradient(nbrBatch);
        for (int i = 0; i < nbrBatch; i++) {
            futureBatchGradient[i] = async(launch::async, &Neural::FeedBatch, this, trainDataSet.batches[i], hp.layersSize);
            if ((i+1) % printBatch == 0)
                cout << "Launching async batch " << (i+1) << " out of " << nbrBatch << "\n";
        }

        // Wait for every threads to finish
        vector<BatchGradient> batchGradients(nbrBatch);
        for (int i = 0; i < nbrBatch; i++) {
            batchGradients[i] = futureBatchGradient[i].get();
        }

        // cout << batchGradients[0].layersGradient[0].nbrNodesIn << endl << endl;
        // for (size_t i = 0; i < batchGradients[0].layersGradient[0].nbrNodesIn; i++){
        //     cout << batchGradients[0].layersGradient[0].weights[i] << endl;
        // }

        // Apply gradient of every batches
        double learningRate = hp.initialLearningRate * (1 / (1 + hp.learnRateDecay * currentEpoch));
        for (int i = 0; i < nbrBatch; i++) {
            ApplyBatchGradient(batchGradients[i], hp.batchSize, learningRate);
        }

        //-----------Accuracy--------------

        // Used to visualize progression of dataSet and check overfitting 
        future<double> trainAccuracyFuture = async(launch::async, &Neural::DataSetAccuracy, this, trainDataSet);
        future<double> testAccuracyFuture = async(launch::async, &Neural::DataSetAccuracy, this, testDataSet);

        accuracyTrain[currentEpoch] = trainAccuracyFuture.get();
        accuracyTest[currentEpoch] = testAccuracyFuture.get();
        cout << "Accuracy on training DataSet: " << accuracyTrain[currentEpoch] << "%\n";
        cout << "Accuracy on test DataSet: " << accuracyTest[currentEpoch] << "%\n\n";
    }
}





BatchGradient Neural::FeedBatch(const Batch& batch, const vector<size_t>& layersSize) {
    BatchGradient batchGradient = BatchGradient(layersSize);
    for (size_t i = 0; i < batch.batchSize; i++){
        Data dataPoint = batch.dataPoints[i];
        CalculateOutputs(dataPoint.inputs);

        Layer* outputLayer = &layers[nbrLayers - 1];
        LayerGradient* outputLayerGradient = &batchGradient.layersGradient[nbrLayers - 1];
        vector<double> previousOutputs = nbrLayers >= 2 ? layers[nbrLayers - 2].outputs : dataPoint.inputs;

        // Compute the nodes values of the output layer and update its gradients
        vector<double> nodeValues(outputLayer->nbrNodesOut);
        for (size_t nodesOut = 0; nodesOut < outputLayer->nbrNodesOut; nodesOut++) {

            double activationDerivative = outputLayer->activationFunction->Derivative(outputLayer->weightedSum, nodesOut);
            double costDerivative = costFunction->Derivative(outputLayer->outputs[nodesOut], dataPoint.targets[nodesOut]);
            double currentNodeValue = activationDerivative * costDerivative;
            nodeValues[nodesOut] = currentNodeValue;


            // outputLayer->gradientBiases[nodesOut] += currentNodeValue;
            outputLayerGradient->biases[nodesOut] += currentNodeValue;
            for (size_t nodesIn = 0; nodesIn < outputLayer->nbrNodesIn; nodesIn++){
                // outputLayer->gradientWeights[nodesOut * outputLayer->nbrNodesIn + nodesIn] += previousOutputs[nodesIn] * currentNodeValue;
                outputLayerGradient->weights[nodesOut * outputLayer->nbrNodesIn + nodesIn] += previousOutputs[nodesIn] * currentNodeValue;
            }
        }
        // exit(0);



        // Go back through the layers, compute the corresponding node values and update the gradient at the same time
        for (size_t j = nbrLayers-2; j <= nbrLayers-2; j--) {
            previousOutputs = j > 0 ? layers[j - 1].outputs : dataPoint.inputs;
            Layer* currentLayer = &layers[j];
            LayerGradient* layerGradient = &batchGradient.layersGradient[j];

            nodeValues = currentLayer->UpdateGradient(layers[j + 1], nodeValues, previousOutputs, layerGradient);
        }

        // for (size_t i = 0; i < nbrLayers; i++)
        //     layers[i].ApplyGradient(learningRate / batch.batchSize);
    }
    for (size_t i = 0; i < batchGradient.layersGradient[nbrLayers - 1].nbrNodesIn; i++) {
        cout << batchGradient.layersGradient[nbrLayers - 1].weights[i] << endl;
    }

    // cout << "end batch" << endl;
    return batchGradient;
}





void Neural::ApplyBatchGradient(BatchGradient batchGradient, size_t batchSize, double learningRate) {
    for (size_t i = 0; i < nbrLayers; i++){
        layers[i].ApplyGradient(batchGradient.layersGradient[i], batchSize, learningRate);
    }
}



vector<double> Neural::CalculateOutputs(vector<double> inputs){
    for (size_t i = 0; i < nbrLayers; i++) {
        inputs = layers[i].CalculateOutputs(inputs);
    }
    return inputs;
}

//-----------------Cost---------------
double Neural::DataPointCost(const Data& dataPoint) {
    vector<double> outputs = CalculateOutputs(dataPoint.inputs);
    return costFunction->Function(outputs, dataPoint.targets);
}
double Neural::BatchCost(const Batch& batch) {
    double cost = 0;
    for (size_t i = 0; i < batch.batchSize; i++) {
        cost += DataPointCost(batch.dataPoints[i]);
    }

    return cost;
}
double Neural::DataSetCost(const DataSet& dataSet) {
    vector<future<double>> batchesCost(dataSet.nbrBatch);

    for (size_t i = 0; i < dataSet.nbrBatch; i++){
        batchesCost[i] = async(launch::async, &Neural::BatchCost, this, dataSet.batches[i]);
    }
    
    double cost = 0;
    for (size_t i = 0; i < dataSet.nbrBatch; i++){
        cost += batchesCost[i].get();
    }
    return cost;

}

//-----------------Accuracy---------------
double Neural::BatchAccuracy(const Batch& batch) {
    double nbrGood = 0;
    for (size_t i = 0; i < batch.batchSize; i++) {
        if (Classify(batch.dataPoints[i].inputs) == GetMaxIndex(batch.dataPoints[i].targets))
            nbrGood++;
    }

    return nbrGood * 100 / batch.batchSize;
}
double Neural::DataSetAccuracy(const DataSet& dataSet) {
    vector<future<double>> batchesAccuracyValues(dataSet.nbrBatch);

    for (size_t i = 0; i < dataSet.nbrBatch; i++) {
        batchesAccuracyValues[i] = async(launch::async, &Neural::BatchAccuracy, this, dataSet.batches[i]);
    }

    double totalAccuracy = 0;
    for (auto& future : batchesAccuracyValues) {
        totalAccuracy += future.get(); // Retrieve the result from the future
    }
    return totalAccuracy / dataSet.nbrBatch;
}

//-----------------Classify---------------
int Neural::Classify(const vector<double>& inputs) {
    vector<double> outputs = CalculateOutputs(inputs);
    return GetMaxIndex(outputs);
}
int Neural::GetMaxIndex(const vector<double>& outputs) {
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
