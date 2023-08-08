#pragma once

#include <cstdlib>
#include <algorithm>
#include <random>
#include <iostream>

#include "layer.hpp"
#include "costFunctions.hpp"
#include "hyperParameters.hpp"
#include "data/dataLoader.hpp"



class Neural
{
    public:
        size_t nbrLayers;
        std::vector<size_t> layersSize;
        std::vector<Layer> layers;
        CostFunction* costFunction;

    public:
        //-----------------Serialization---------------
        void ToJson(std::string fileName);
        nlohmann::json Serialize() const;

        //-----------------Constructor---------------
        Neural(std::vector<size_t> layerSizes, HyperParameters *hyperParameters);

        //-----------------BackPropagation---------------
        std::vector<double> CalculateOutputs(std::vector<double> inputs);
        void FeedBatch(Batch batch, size_t batchSize, double learningRate);
        void Learn(DataSet trainDataSet, DataSet testDataSet, HyperParameters hp);

        //-----------------Cost---------------
        double DataPointCost(Data dataPoint);
        double BatchCost(Batch batch);
        double DataSetCost(DataSet dataSet);

        //-----------------Accuracy---------------
        double BatchAccuracy(Batch batch);
        double DataSetAccuracy(DataSet dataSet);

        //-----------------Classify---------------
        int Classify(std::vector<double> inputs);
        int GetMaxIndex(std::vector<double> outputs);
}; 


//-----------------Deserialization---------------
Neural NeuralFromJson(std::string fileName, HyperParameters *hyperParameters);

