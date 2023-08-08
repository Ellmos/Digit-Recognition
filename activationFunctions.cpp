#include "activationFunctions.hpp"
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

    return exp(inputs[index]) / expSum;
}

double Softmax::Derivative(vector<double> inputs, size_t index) {
    Normalize(inputs);

    double expSum = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        expSum += exp(inputs[i]);
    }

    double ex = exp(inputs[index]);

    return (ex * expSum - ex * ex) / (expSum * expSum);
}
