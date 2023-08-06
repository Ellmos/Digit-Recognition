#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>


class ActivationFunction {
public:
    virtual double Function(std::vector<double> inputs, size_t index) = 0;
    virtual double Derivative(std::vector<double> inputs, size_t index) = 0;
    virtual ~ActivationFunction() = default;
};


class Relu : public ActivationFunction { 
public:
    double Function(std::vector<double> inputs, size_t index);
    double Derivative(std::vector<double> inputs, size_t index);
};


class Softmax : public ActivationFunction {
public:
    double Function(std::vector<double> inputs, size_t index);
    double Derivative(std::vector<double> inputs, size_t index);
};




