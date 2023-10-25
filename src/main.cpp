#include <chrono>
#include <iterator>

#include "activationFunctions.hpp"
#include "dataLoader.hpp"
#include "hyperParameters.hpp"
#include "layer.hpp"
#include "neural.hpp"


using namespace std;


int main() {
    HyperParameters hyperParameters = HyperParameters();

    // Create a new neural Network
    Neural neural = Neural(hyperParameters);

    // Load a save
    // Neural neural = NeuralFromJson("aller", hyperParameters);


    cout << "----------------Generating Dataset-------------------\n";
    auto start_time = chrono::high_resolution_clock::now();

    DataSet trainDataSet = DataSet(0);
    DataSet testDataSet = DataSet(0);

    // first int is proportion of mnist and second is proportion of modifiedMnist
    LoadDataSets(&trainDataSet, &testDataSet, hyperParameters.batchSize, 0, 100);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end_time - start_time;
    cout << "Time taken: " << duration.count() << " seconds" << endl;


    // Data oui = trainDataSet.batches[0].dataPoints[0];
    // vector<Data> vec;
    // vec.push_back(oui);
    // Batch batch = Batch(vec);
    // // neural.FeedBatch(batch, hyperParameters.layersSize);
    //
    // DataSet dataSet = DataSet(1, 1);
    // dataSet.batches[0] = batch;
    // neural.Learn(dataSet, dataSet, hyperParameters);
    // exit(0);



    start_time = chrono::high_resolution_clock::now();

    neural.Learn(trainDataSet, testDataSet, hyperParameters);

    end_time = chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    cout << "Time taken: " << duration.count() << " seconds" << endl;


    // cout << "------------------Learning-----------------------\n";
    // neural.Learn(trainDataSet, testDataSet, hyperParameters);
    // neural.ToJson("newSave");

    cout << "End main\n";
    return 0;
}
