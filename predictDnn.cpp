#include "UtilLayer.h"

int main(int argc, const char *argv[]) {
	if (argc < 5) problemAndExit("predictDnn <testData[:<startIndexA>:<startIndexB>]> <dnnDescriptor> <modelVersionId> <P|p|c|v> [expand]");

	auto ModelFile = new std::ifstream(std::string(argv[2]) + "-" + argv[3] + ".txt");
	if (ModelFile->fail()) problemAndExit("Model file does not exists!");
	auto DNN = Layer::getLayer()->loadAll(ModelFile, true);
	ModelFile->close();
	auto InputLayer = DNN->getInputLayer();
	auto OutputLayer = DNN->getOutputLayer();

	int indexA, indexB, outputWidth;
	if (argc < 6) {
		indexA = 0;
		indexB = InputLayer->width;
		outputWidth = OutputLayer->width;
	}
	else {
		indexA = 1; 
		indexB = 0;
		outputWidth = 1;
	}
	auto res = split(argv[1], ":");
	auto fileName = res[0];
	if (res.size() >= 2) indexA = (int)paresInt(res[1]);
	if (res.size() >= 3) indexB = (int)paresInt(res[2]);
	if (indexA < 0 || indexB < 0) problemAndExit("Invalid index parmeter!");

	auto InputFile = new std::ifstream(fileName);
	if (InputFile->fail()) problemAndExit("Input file does not exists!");

	double* dataInput = nullptr;
	double* dataTarget = nullptr;
	unsigned int nData = 0;
	double sumLoss = 0.0;
	if (argv[4][0] == 'c' || argv[4][0] == 'v') {
		nData = loadCsvFile(InputFile, 0, dataInput, InputLayer->width, nullptr, dataTarget, outputWidth, nullptr, indexA, indexB);
	}
	else {
		nData = loadCsvFile(InputFile, 0, dataInput, InputLayer->width, nullptr, dataTarget, 0, nullptr, indexA, indexB);
	}
	InputFile->close();
	if (nData <= 0) problemAndExit("No data!");
	delete InputLayer->outputValue;
	delete OutputLayer->targetValue;
	OutputLayer->targetValue = nullptr;
	
	if (argc < 6) {
		for (size_t index = 0; index < nData; index++) {
			InputLayer->outputValue = dataInput + index * InputLayer->width;
			if(dataTarget != nullptr) OutputLayer->targetValue = dataTarget + index * OutputLayer->width;
			
			InputLayer->feedForwardAndBackward(false);
			sumLoss += OutputLayer->loss;

			if (argv[4][0] == 'P' || argv[4][0] == 'p' || argv[4][0] == 'v') {
				if (argv[4][0] == 'P') {
					std::cout << OutputLayer->outputValue[0];
					for (unsigned int i = 1; i < OutputLayer->width; i++) std::cout << "," << OutputLayer->outputValue[i];
				}
				else {
					std::cout << index;
					for (unsigned int i = 0; i < InputLayer->width; i++) std::cout << "," << InputLayer->outputValue[i];
					if (dataTarget != nullptr) for (unsigned int i = 0; i < OutputLayer->width; i++) std::cout << "," << OutputLayer->targetValue[i];
					for (unsigned int i = 0; i < OutputLayer->width; i++) std::cout << "," << OutputLayer->outputValue[i];
					if (dataTarget != nullptr) std::cout << "," << OutputLayer->loss;
				}
				std::cout << std::endl;
			}
		}
		if (dataTarget != nullptr) std::cout << "Loss: " << (sumLoss / nData) << std::endl;
	}
	else {
		unsigned int failedPrediction = 0;
		unsigned int* errorSamples = nullptr;
		if (dataTarget != nullptr) {
			OutputLayer->targetValue = new double[OutputLayer->width];
			errorSamples = new unsigned int[OutputLayer->width * OutputLayer->width];
			std::fill_n(errorSamples, OutputLayer->width * OutputLayer->width, (unsigned int)0);
		}

		for (size_t index = 0; index < nData; index++) {
			InputLayer->outputValue = dataInput + index * InputLayer->width;
			if (dataTarget != nullptr) {
				for (unsigned int i = 0; i < OutputLayer->width; i++) OutputLayer->targetValue[i] = 0;
				OutputLayer->targetValue[(int)(round(dataTarget[index]))] = 1.0;
			}
			InputLayer->feedForwardAndBackward(false);
			sumLoss += OutputLayer->loss;

			double maxValue = 0;
			unsigned int maxIndex = 0;
			for (unsigned int i = 0; i < OutputLayer->width; i++) {
				if (OutputLayer->outputValue[i] > maxValue) {
					maxValue = OutputLayer->outputValue[i];
					maxIndex = i;
				}
			}

			if (dataTarget != nullptr && argv[4][0] == 'c' && maxIndex != (int)(round(dataTarget[index]))) {
				errorSamples[(int)(round(dataTarget[index])) * OutputLayer->width + maxIndex]++;
				failedPrediction++;
			}

			if (argv[4][0] == 'P' || argv[4][0] == 'p' || argv[4][0] == 'v') {
				if (argv[4][0] == 'P') {
					std::cout << maxIndex;
				}
				else {
					std::cout << index << "," << maxIndex;
					if (dataTarget != nullptr) std::cout << "," << (int)(round(dataTarget[index]));
					for (unsigned int i = 0; i < InputLayer->width; i++) std::cout << "," << InputLayer->outputValue[i];
					if (dataTarget != nullptr) std::cout << "," << OutputLayer->loss;
				}
				std::cout << std::endl;
			}
		}

		if (dataTarget != nullptr && argv[4][0] == 'c') {
			for (size_t targetIndex = 0; targetIndex < OutputLayer->width; targetIndex++) {
				for (size_t maxIndex = 0; maxIndex < OutputLayer->width; maxIndex++) {
					if (errorSamples[targetIndex * OutputLayer->width + maxIndex] > 0) std::cout << targetIndex << "->" << maxIndex << "=" << errorSamples[targetIndex * OutputLayer->width + maxIndex] << " ";
				}
			}
			std::cout << std::endl;
		}

		if (dataTarget != nullptr) std::cout << "Loss: " << (sumLoss / nData) << "  #FailedPrediction: " << failedPrediction << "  Accuracy: " << (1.0 - ((double)failedPrediction) / nData) << std::endl;
	}

	InputLayer->outputValue = nullptr;
	OutputLayer->targetValue = nullptr;
	Layer::deleteAllLayer(DNN);
	delete dataInput;
	delete dataTarget;
	return 0;
}