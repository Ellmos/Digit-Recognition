#include "activationFunctions.h"

using namespace std;

double Relu::Function(vector<double> inputs, size_t index) {
    return inputs[index] > 0 ? inputs[index] : 0;
}

double Relu::Derivative(vector<double> inputs, size_t index) {
    return inputs[index] > 0 ? 1 : 0;
}


double Softmax::Function(vector<double> inputs, size_t index) {
    double expSum = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        expSum += std::exp(inputs[i]);
    }

    return exp(inputs[index]) / expSum;
}

double Softmax::Derivative(vector<double> inputs, size_t index) {
    double expSum = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        expSum += exp(inputs[i]);
    }

    double ex = exp(inputs[index]);

    return (ex * expSum - ex * ex) / (expSum * expSum);
}
