#include "neural.hpp"

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
    ofstream file ("data/saves/" + fileName + ".json");
    if (!file.is_open())
        throw runtime_error("Neural::ToJson: Failed to create save file");

    file << this->Serialize().dump();

    file.close();
}


//-----------------Deserialization---------------
Neural NeuralFromJson(std::string fileName, HyperParameters *hyperParameters){
    //Read File
    string filePath = "data/saves/" + fileName + ".json";
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

    Neural neural = Neural(layersSize, hyperParameters);
    for (size_t i = 0; i < neural.nbrLayers; i++) {
        vector<double> tmp = parsedJson["layers"][i]["weights"];
        vector<double> tmp2 = parsedJson["layers"][i]["biases"];

        neural.layers[i].weights = tmp;
        neural.layers[i].biases = tmp2;
    }

    file.close();
    return neural;
}




//---------------------------Neural class --------------------------------

Neural::Neural(vector<size_t> layerSizes, HyperParameters* hyperParameters) {
    this->layersSize = layerSizes;
    this->nbrLayers = layersSize.size() - 1;
    this->costFunction = hyperParameters->costFunction;

    for (size_t i = 0; i < nbrLayers; i++)
        layers.push_back(Layer(layersSize[i], layersSize[i + 1], hyperParameters->activationFunction));
    layers[nbrLayers - 1].activationFunction = hyperParameters->outputActivationFunction;
}



//---------------------------BackPropagation--------------------------------
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


            outputLayer->gradientBiases[nodesOut] += currentNodeValue;
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

        for (size_t i = 0; i < nbrLayers; i++)
            layers[i].ApplyGradient(learningRate / batch.batchSize);
    }
}

void Neural::Learn(DataSet trainDataSet, DataSet testDataSet, HyperParameters hp) {
    int nbrBatch = trainDataSet.nbrBatch;
    int printBatch = nbrBatch / 10 != 0 ? nbrBatch / 10 : 1;

    vector<double> accuracyTrain(hp.epoch);
    vector<double> accuracyTest(hp.epoch);

    auto rng = default_random_engine{};

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
        cout << "Accuracy on training DataSet: " << accuracyTrain[currentEpoch] << "%\n";
        cout << "Accuracy on test DataSet: " << accuracyTest[currentEpoch] << "%\n";
    }
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
