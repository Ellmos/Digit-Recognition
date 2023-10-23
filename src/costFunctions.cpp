#include "costFunctions.hpp"

using namespace std;

double MeanSquare::Function(vector<double> outputValues, vector<double> targetValues) 
{
    double cost = 0;

    for (size_t i = 0; i < outputValues.size(); i++)
        {
            double diff = targetValues[i] - outputValues[i];
            cost += diff * diff;
        }

    // Divide by 0.5 so the square when derivated is cancelled
    return cost * 0.5;
}

double MeanSquare::Derivative(double outputValue, double targetValue)
{
    return outputValue - targetValue;
}



double CrossEntropy::Function(vector<double> outputValues, vector<double> targetValues)
{
    double cost = 0;

    for (size_t i = 0; i < outputValues.size(); i++){
        double output = outputValues[i];
        double target = targetValues[i];

        double tmp = (target == 1) ? -std::log(output) : -std::log(1 - output);
        if (!std::isnan(tmp)) 
            cost += tmp;
    }

    return cost;
}

double CrossEntropy::Derivative(double outputValue, double targetValue)
{
    if (outputValue == 0 || outputValue == 1)
        return 0;

    return (-outputValue + targetValue) / (outputValue * (outputValue - 1));
}
