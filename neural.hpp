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
        Neural(std::vector<size_t> layerSizes, const HyperParameters& hyperParameters);

        //-----------------BackPropagation---------------
        std::vector<double> CalculateOutputs(std::vector<double> inputs);
        void FeedBatch(const Batch& batch, double learningRate);
        void Learn(DataSet& trainDataSet, const DataSet& testDataSet, const HyperParameters& hp);

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
Neural NeuralFromJson(std::string fileName, const HyperParameters& hyperParameters);

