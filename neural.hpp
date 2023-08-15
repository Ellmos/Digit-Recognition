#pragma once

#include <cstdlib>
#include <algorithm>
#include <random>
#include <iostream>
#include <future>

#include "layer.hpp"
#include "costFunctions.hpp"
#include "hyperParameters.hpp"
#include "data/dataLoader.hpp"




//-----------------Neural Class---------------
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
        Neural(HyperParameters& hyperParameters);

        //-----------------BackPropagation---------------
        void Learn(DataSet& trainDataSet, DataSet& testDataSet, HyperParameters& hp);
        BatchGradient FeedBatch(const Batch& batch, const std::vector<size_t>& layersSize);
        void ApplyBatchGradient(BatchGradient batchGradient, size_t batchSize, double learningRate);
        std::vector<double> CalculateOutputs(std::vector<double> inputs);

        //-----------------Cost---------------
        double DataPointCost(const Data& dataPoint);
        double BatchCost(const Batch& batch);
        double DataSetCost(const DataSet& dataSet);

        //-----------------Accuracy---------------
        double BatchAccuracy(const Batch& batch);
        double DataSetAccuracy(const DataSet& dataSet);

        //-----------------Classify---------------
        int Classify(const std::vector<double>& inputs);
        int GetMaxIndex(const std::vector<double>& outputs);
}; 


//-----------------Deserialization---------------
Neural NeuralFromJson(std::string fileName, HyperParameters& hyperParameters);



