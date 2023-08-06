#include "activationFunctions.hpp"
#include "layer.hpp"
#include "neural.hpp"
#include "hyperParameters.hpp"
#include "data/dataLoader.hpp"

#include <chrono>

using namespace std;


int main() {
    vector<size_t> layersSize = {784, 100, 10};

    HyperParameters hyperParameters = HyperParameters();
    Neural neural = Neural(layersSize, &hyperParameters);


    cout << "----------------Generating Dataset-------------------\n";
    auto start_time = chrono::high_resolution_clock::now();

    DataSet trainDataSet = DataSet(0);
    DataSet testDataSet = DataSet(0);

    // first int is proportion of mnist and second is proportion of modMnist
    LoadDataSets(&trainDataSet, &testDataSet, hyperParameters.batchSize, 0, 100);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end_time - start_time;
    cout << "Time taken:  " << duration.count() << " seconds" << endl;


    cout << "------------------Learning-----------------------\n";
    neural.Learn(trainDataSet, testDataSet, hyperParameters);

    neural.ToJson("aller");

    cout << "End main\n";
    return 0;
}
