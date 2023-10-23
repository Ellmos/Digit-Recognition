#include "dataLoader.hpp"

using namespace std;



//-------------------DataLoader-----------------
using namespace std;

size_t ReadBigInt32(ifstream &file){
    size_t number;
    file.read((char *)&number, 4);
    
    //Reverse int
    unsigned char c1 = number & 255;
    unsigned char c2 = (number >> 8) & 255;
    unsigned char c3 = (number >> 16) & 255;
    unsigned char c4 = (number >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void ReadFiles(DataSet* dataSet, string imagesFilePath, string labelsFilePath, size_t batchSize, size_t proportion){
    ifstream imageFile(imagesFilePath);
    ifstream labelFile(labelsFilePath);

    if (!imageFile.is_open() || !labelFile.is_open()) 
        throw runtime_error("Failed to open dataSet files");

    //Checking file integrity and getting headers
    size_t magicNumber = ReadBigInt32(imageFile);
    size_t magicNumberLabel = ReadBigInt32(labelFile);
    if (magicNumber != 2051 || magicNumberLabel != 2049)
        throw std::runtime_error("Invalid magic number in data files");


    //Getting information about the datset files 
    size_t numberOfImages = ReadBigInt32(imageFile);
    ReadBigInt32(labelFile);  //Next number of labelFile is numberOfLabels = numberOfImages
    
    numberOfImages = numberOfImages * proportion / 100;

    // width and height of the images (28x228)
    size_t width = ReadBigInt32(imageFile);
    size_t height = ReadBigInt32(imageFile);



    size_t nbrBatch;
    size_t iDataPointOffset;
    size_t iBatchOffset;

    // batchSize = 0 correspond to a signle batch of the size of the dataset (used in the test dataset)  
    if (batchSize == 0){
        batchSize = numberOfImages;
        nbrBatch = 1;
        iBatchOffset = 0;
        if (dataSet->nbrBatch == 0){
            iDataPointOffset = 0;
            dataSet->Extend(nbrBatch, batchSize);
        }
        else {
            iDataPointOffset = dataSet->batches[0].dataPoints.size();
            dataSet->batches[0].dataPoints.resize(dataSet->batches[0].batchSize + batchSize);
        }
    }
    else {
        nbrBatch =  numberOfImages / batchSize;
        iDataPointOffset = 0;
        iBatchOffset = dataSet->nbrBatch;
        dataSet->Extend(nbrBatch, batchSize);
    }

    
    


    // Reading files data
    size_t numberOfPixels = width * height;
    double ratio = 1.0 / 255;

    for (size_t iBatch = iBatchOffset; iBatch < iBatchOffset+nbrBatch; iBatch++){
        for (size_t iDataPoint = iDataPointOffset; iDataPoint < iDataPointOffset + batchSize; iDataPoint++){
            //Read target number
            dataSet->batches[iBatch].dataPoints[iDataPoint].targets.resize(10);
            unsigned char target = 0;
            labelFile.read((char *)&target, 1);
            dataSet->batches[iBatch].dataPoints[iDataPoint].targets[target] = 1;


            //Read inputs numbers
            dataSet->batches[iBatch].dataPoints[iDataPoint].inputs.resize(numberOfPixels);
            for (size_t i = 0; i < numberOfPixels; i++){
                // Read pixel value in file
                unsigned char pixel = 0;
                imageFile.read((char *)&pixel, 1);

                // Normalize pixel and put it in correct batch->dataPoint->input index
                dataSet->batches[iBatch].dataPoints[iDataPoint].inputs[i] = (double)pixel * ratio;
            }
        }
    }
}

void LoadDataSets(DataSet* trainDataSet, DataSet* testDataSet, size_t batchSize, char mnist, char const& modMnist){
    string trainImages = "train-images.idx3-ubyte";
    string trainLabels = "train-labels.idx1-ubyte";
    string testImages = "t10k-images.idx3-ubyte";
    string testLabels = "t10k-labels.idx1-ubyte";

    if (mnist){
        string mnistDir = "./data/mnist/";
        ReadFiles(trainDataSet, mnistDir+trainImages, mnistDir+trainLabels, batchSize, mnist);
        ReadFiles(testDataSet, mnistDir+testImages, mnistDir+testLabels, 0, mnist);
    }

    if (modMnist){
        string modMnistdir = "./data/modifiedMnist/";
        ReadFiles(trainDataSet, modMnistdir+trainImages, modMnistdir+trainLabels, batchSize, modMnist);
        ReadFiles(testDataSet, modMnistdir+testImages, modMnistdir+testLabels, 0, modMnist);
    }
}
