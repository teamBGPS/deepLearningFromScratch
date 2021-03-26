#pragma once

#include "Layer.h"

void trainADAM(Layer* layer, unsigned int nData, double *input, double* output, unsigned int batchSize, unsigned int nEpoch, double learningRate, double beta1 = 0.9, double beta2 = 0.999);
void LayerToRGB(std::string aFileName, Layer* aLayer, double aScale);