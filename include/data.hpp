#pragma once

#include <stddef.h>
#include <vector>

//-------------------Data Class-----------------
class Data{
public:
    std::vector<double> inputs;
    std::vector<double> targets;

public:
    Data(){};
};


//-------------------Batch Class-----------------
class Batch{
public:
    std::vector<Data> dataPoints;
    size_t batchSize;

public:
    Batch(){};
    Batch(std::vector<Data> dataPoints){
        this->dataPoints = dataPoints;
        this->batchSize = dataPoints.size();
    }
    Batch(size_t length){
        this->dataPoints = std::vector<Data>(length);
        this->batchSize = length;
    }
};


//-------------------DataSet Class-----------------
class DataSet{
public:
    std::vector<Batch> batches;
    size_t nbrBatch;

public:
    DataSet(size_t length){
        this->batches.reserve(length);
        this->nbrBatch = length;
    }
    DataSet(size_t nbrBatch, size_t batchSize){
        this->nbrBatch = nbrBatch;
        this->batches = std::vector<Batch>(nbrBatch, Batch(batchSize));
    }

    void Extend(size_t nbrBatchToAdd, size_t batchSize){
        this->nbrBatch += nbrBatchToAdd;
        this->batches.resize(this->nbrBatch, Batch(batchSize));
    }
};

//-----------------Gradient Data class---------------
class LayerGradient
{
    public:
        size_t nbrNodesIn;
        size_t nbrNodesOut; 
        std::vector<double> weights;
        std::vector<double> biases;

    public:
        LayerGradient(size_t nbrNodesIn, size_t nbrNodesOut){
            this->nbrNodesIn = nbrNodesIn;
            this->nbrNodesOut = nbrNodesOut;
            this->weights.assign(nbrNodesOut * nbrNodesIn, 0);
            this->biases.assign(nbrNodesOut, 0);
        }
};

class BatchGradient
{
    public:
        size_t nbrLayers;
        std::vector<LayerGradient> layersGradient;

    public:
        BatchGradient(){};
        BatchGradient(std::vector<size_t> layersSize){
            this->nbrLayers = layersSize.size()-1;
            for (size_t i = 0; i < nbrLayers; i++)
                layersGradient.push_back(LayerGradient(layersSize[i], layersSize[i + 1]));
        }
};
