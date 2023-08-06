#pragma once

#include <cstdlib>
#include <algorithm>
#include <random>

#include "layer.h"
#include "costFunctions.h"
#include "hyperParameters.h"
#include "data/dataLoader.h"


class Neural
{
    public:
        size_t nbrLayers;
        size_t* layersSize;
        std::vector<Layer> layers;
        CostFunction* costFunction;

    public:
        Neural(size_t layerSizes[], size_t nbrLayers, HyperParameters *hyperParameters);
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
