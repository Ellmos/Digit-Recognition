#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>


class CostFunction {
public:
    virtual double Function(std::vector<double> outputValues, std::vector<double> targetValues) = 0;
    virtual double Derivative(double outputValue, double targetValue) = 0;
    virtual ~CostFunction() = default;
};


class MeanSquare : public CostFunction { 
public:
    double Function(std::vector<double> outputValues, std::vector<double> targetValues);
    double Derivative(double outputValue, double targetValue);
};


class CrossEntropy : public CostFunction {
public:
    double Function(std::vector<double> outputValues, std::vector<double> targetValues);
    double Derivative(double outputValue, double targetValue);
};
