#pragma once

#include <vector>

//-------------------Data Class-----------------
class Data{
public:
    std::vector<double> inputs;
    std::vector<double> targets;

public:
    Data();
};


//-------------------Batch Class-----------------
class Batch{
public:
    std::vector<Data> dataPoints;
    size_t batchSize;

public:
    Batch();
    Batch(std::vector<Data> dataPoints);
    Batch(size_t length);
};


//-------------------DataSet Class-----------------
class DataSet{
public:
    std::vector<Batch> batches;
    size_t nbrBatch;

public:
    DataSet(size_t length);
    DataSet(size_t length, size_t batchSize);

    void Extend(size_t nbrBatchToAdd, size_t batchSize);
};

