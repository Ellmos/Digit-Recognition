#include "../include/activationFunctions.hpp"
#include <iostream>

using namespace std;


//--------------Relu------------
double Relu::Function(vector<double> inputs, size_t index) {
    return inputs[index] > 0 ? inputs[index] : 0;
}

double Relu::Derivative(vector<double> inputs, size_t index) {
    return inputs[index] > 0 ? 1 : 0;
}




//--------------Relu------------
void Normalize(vector<double> &inputs){
    double max = *max_element(inputs.begin(), inputs.end());
    for (size_t i = 0; i < inputs.size(); i++)
        inputs[i] -= max;
}

double Softmax::Function(vector<double> inputs, size_t index) {
    Normalize(inputs);

    double expSum = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        expSum += exp(inputs[i]);
    }
    
    double output = exp(inputs[index]) / expSum;

    return output >= 0.99 ? 1 : output;
}

double Softmax::Derivative(vector<double> inputs, size_t index) {
    Normalize(inputs);

    // double oui[] = {-90.175820824531741, 80.117045290677922, 49.409126101882251, 14.388140904427623, -2.2061784887730411, 
    //     92.485586674441578, 38.63878810063359, -52.393199982827312, -75.143634881036348, -21.559602296244492};
    //
    // double test = 0;
    // for (size_t i = 0; i < 10; i++) {
    //     test += exp(oui[i]);
    // }
    //
    // double test2 = exp(oui[0]);
    //
    // cout << "aaaaaaaaaaaaaaaaa  " << (test) << endl;

    double expSum = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        expSum += exp(inputs[i]);
    }

    double ex = exp(inputs[index]);

    return (ex * expSum - ex * ex) / (expSum * expSum);
}
