#pragma once

#include "data.hpp"
#include <fstream>


void ReadFiles(DataSet* dataSet, std::string imagesFilePath, std::string labelsFilePath, size_t batchSize, size_t proportion);
void LoadDataSets(DataSet* trainDataSet, DataSet* testDataSet, size_t batchSize, char mnist, char const& modMnist);
