#include "data.h"

using namespace std; 

//-------------------Data Class-----------------
Data::Data(){}

//-------------------Batch Class-----------------
Batch::Batch(){};
Batch::Batch(std::vector<Data> dataPoints){
    this->dataPoints = dataPoints;
    this->batchSize = dataPoints.size();
}
Batch::Batch(size_t length){
    this->dataPoints = vector<Data>(length);
    this->batchSize = length;
}

//-------------------DataSet Class-----------------
DataSet::DataSet(size_t length){
    this->batches.reserve(length);
    this->nbrBatch = length;
}
DataSet::DataSet(size_t nbrBatch, size_t batchSize){
    this->nbrBatch = nbrBatch;
    this->batches = vector<Batch>(nbrBatch, Batch(batchSize));
}

void DataSet::Extend(size_t nbrBatchToAdd, size_t batchSize){
     
    this->nbrBatch += nbrBatchToAdd;
    this->batches.resize(this->nbrBatch, Batch(batchSize));
}
