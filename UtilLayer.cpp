#include "UtilLayer.h"

void trainADAM(Layer* layer, unsigned int nData, double *input, double* target, unsigned int batchSize, unsigned int nEpoch, double learningRate, double beta1, double beta2) {
	std::cerr << "TrainADAM..." << std::endl;

	auto inputLayer = layer->getInputLayer();
	auto outputLayer = layer->getOutputLayer();
	auto l = inputLayer->nextLayer;
	do {
		if (l->adamWM == nullptr) {
			l->adamWM = new double[l->width * l->prevWidth];
			l->adamWV = new double[l->width * l->prevWidth];
			l->adamBM = new double[l->width];
			l->adamBV = new double[l->width];
		}
		std::fill_n(l->adamWM, l->width * l->prevWidth, 0.0);
		std::fill_n(l->adamWV, l->width * l->prevWidth, 0.0);
		std::fill_n(l->adamBM, l->width, 0.0);
		std::fill_n(l->adamBV, l->width, 0.0);
	} while ((l = l->nextLayer) != nullptr);

	auto sampleIndex = new unsigned int[nData];
	for (unsigned int i = 0; i < nData; i++) sampleIndex[i] = i;

	delete[] inputLayer->outputValue;
	delete[] outputLayer->targetValue;
	for (size_t epoch = 0; epoch < nEpoch && !inputLayer->toExit; epoch++) {
		std::cerr << '\r' << epoch;
		
		for (size_t i = nData - 1; i > 0; i--) {
			auto j = randUniform64() % (i + 1);
			auto temp = sampleIndex[i];	sampleIndex[i] = sampleIndex[j]; sampleIndex[j] = temp;
		}
		
		for (size_t index = 0; index < nData; index++) {
			inputLayer->outputValue = input + sampleIndex[index] * inputLayer->width;
			outputLayer->targetValue = target + sampleIndex[index] * outputLayer->width;
			
			inputLayer->feedForwardAndBackward();

			if ((index + 1) % batchSize == 0 || index == nData - 1) {
				auto l = inputLayer->nextLayer;
				do {
					for (size_t i = 0; i < l->width * l->prevWidth; i++) {
						l->adamWM[i] = beta1 * l->adamWM[i] + (1.0 - beta1) * l->dW[i];
						l->adamWV[i] = beta2 * l->adamWV[i] + (1.0 - beta2) * l->dW[i] * l->dW[i];
						double m = l->adamWM[i] / (1 - beta1);
						double v = l->adamWV[i] / (1 - beta2);
						if (l->isTrainable) l->W[i] = l->W[i] - learningRate * m / (sqrt(v) + 1e-8);
						l->dW[i] = 0;
					}
					for (size_t i = 0; i < l->width; i++) {
						l->adamBM[i] = beta1 * l->adamBM[i] + (1.0 - beta1) * l->dB[i];
						l->adamBV[i] = beta2 * l->adamBV[i] + (1.0 - beta2) * l->dB[i] * l->dB[i];
						double m = l->adamBM[i] / (1 - beta1);
						double v = l->adamBV[i] / (1 - beta2);
						if (l->isTrainable) l->B[i] = l->B[i] - learningRate * m / (sqrt(v) + 1e-8);
						l->dB[i] = 0;
					}
				} while ((l = l->nextLayer) != nullptr);
			}
		}

		if (inputLayer->toDump) {
			inputLayer->toDump = false;
			std::cerr << "Creating dump file..." << std::endl;
			std::ofstream outputFile("Dump-" + std::to_string(epoch) + ".txt");
			if(!outputFile.fail()) inputLayer->saveAll(&outputFile, true);
			std::cerr << "Dump-" << std::to_string(epoch) << ".txt file created!" << std::endl;
		}
	}
	std::cerr << '\r';

	delete[] sampleIndex;
	inputLayer->outputValue = nullptr;;
	outputLayer->targetValue = nullptr;
}

void LayerToRGB(std::string aFileName, Layer* aLayer, double aScale) {
	unsigned int sumNeuron = 0;
	unsigned int maxY = 0;

	Layer* layer = aLayer->getInputLayer();
	while ((layer = layer->nextLayer) != nullptr) {
		sumNeuron += layer->width;
		if (layer->prevWidth > maxY) maxY = layer->prevWidth;
	}
	maxY++; // add b

	unsigned char* R = new unsigned char[sumNeuron * maxY];
	unsigned char* G = new unsigned char[sumNeuron * maxY];
	unsigned char* B = new unsigned char[sumNeuron * maxY];
	memset(R, 0, sumNeuron * maxY);
	memset(G, 0, sumNeuron * maxY);
	memset(B, 255, sumNeuron * maxY);

	unsigned int baseX = 0;
	layer = aLayer->getInputLayer();
	while ((layer = layer->nextLayer) != nullptr) {
		for (size_t x = 0; x < layer->width; x++) {
			for (size_t y = 0; y < layer->prevWidth + 1; y++) {
				double value;
				if (y == 0) value = layer->B[x]; else value = layer->W[(y - 1) * layer->width + x];
				bool isNegative = false;
				if (value < 0) { isNegative = true; value = -value; }
				unsigned int byteValue = 0;
				if (value <= 1.0e-3) {
					R[y * sumNeuron + (baseX + x)] = 255;
					G[y * sumNeuron + (baseX + x)] = 255;
					B[y * sumNeuron + (baseX + x)] = 255;
				}
				else {
					double dValue = (log(value) - log(1e-3)) *  aScale;
					if (dValue < 255) byteValue = (unsigned int)dValue; else byteValue = 255;
					if (isNegative) {
						R[y * sumNeuron + (baseX + x)] = 255 - byteValue;
						G[y * sumNeuron + (baseX + x)] = 0;
						B[y * sumNeuron + (baseX + x)] = 0;
					}
					else {
						R[y * sumNeuron + (baseX + x)] = 0;
						G[y * sumNeuron + (baseX + x)] = 255 - byteValue;
						B[y * sumNeuron + (baseX + x)] = 0;
					}
				}
			}
		}
		baseX += layer->width;
	}
	printArrayToBmp(sumNeuron, maxY, R, G, B, aFileName);

	delete[] R; delete[] G; delete[] B;
}