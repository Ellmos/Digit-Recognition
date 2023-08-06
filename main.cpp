#include "activationFunctions.h"
#include "layer.h"
#include "neural.h"
#include "hyperParameters.h"
#include "data/dataLoader.h"

#include <chrono>

using namespace std;

int main() {
    size_t layersSize[] = {784, 64, 10};       
    size_t nbrLayers = sizeof(layersSize) / sizeof(layersSize[0]) - 1;

    HyperParameters hyperParameters = HyperParameters();
    Neural neural = Neural(layersSize, nbrLayers, &hyperParameters);


    cout << "----------------Generating Dataset-------------------\n";
    auto start_time = chrono::high_resolution_clock::now();

    DataSet trainDataSet = DataSet(0);
    DataSet testDataSet = DataSet(0);

    // first int is proportion of mnist and second is proportion of modMnist
    LoadDataSets(&trainDataSet, &testDataSet, hyperParameters.batchSize, 0, 100);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end_time - start_time;
    cout << "Time taken:  " << duration.count() << " seconds" << endl;



    neural.Learn(trainDataSet, testDataSet, hyperParameters);


    cout << "End main\n";
    return 0;
}
