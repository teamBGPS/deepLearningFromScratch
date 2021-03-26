#include "Layer.h"

Layer::Layer() { }

Layer::~Layer() {
	delete W; delete B; delete dW; delete dB; delete activationValue; delete derivateActivation; delete outputValue; delete backpropagetedError;
}

void Layer::deleteAllLayer(Layer* l){
	Layer* nextLayer;
	auto currentLayer = l->getInputLayer();
	while ((nextLayer = currentLayer->nextLayer) != nullptr) {
		delete currentLayer;
		currentLayer = nextLayer;
	}
	delete currentLayer;
}

Layer* Layer::getLayer() {
	return new Layer;
}

Layer* Layer::getInputLayer() {
	auto layer = this;
	while (layer->layerType != INPUT) layer = layer->prevLayer;
	return layer;
}

Layer* Layer::getOutputLayer() {
	auto layer = this;
	while (layer->layerType != OUTPUT) layer = layer->nextLayer;
	return layer;
}

Layer* Layer::allocate() {
	outputValue = new double[width];
	memset(outputValue, 0, width * sizeof(double));
	if (layerType == INPUT) return this;

	W = new double[width * prevWidth];
	B = new double[width];
	dW = new double[width * prevWidth];
	dB = new double[width];
	activationValue = new double[width];
	derivateActivation = new double[width];
	backpropagetedError = new double[width];
	memset(W, 0, width * prevWidth * sizeof(double)); memset(dW, 0, width * prevWidth * sizeof(double));
	memset(B, 0, width * sizeof(double)); memset(dB, 0, width * sizeof(double)); memset(activationValue, 0, width * sizeof(double));
	memset(derivateActivation, 0, width * sizeof(double)); memset(backpropagetedError, 0, width * sizeof(double));
	if (layerType != OUTPUT)  return this;

	targetValue = new double[width];
	memset(targetValue, 0, width * sizeof(double));
	return this;
}

Layer* Layer::initW(std::string initMethod) {
	if (initMethod == "random") for (size_t i = 0; i < width * prevWidth; i++) W[i] = 2.0 * std::ceil(1000.0*randUniform()) / 1000.0 - 1.0;
	else if (initMethod == "positiveRandom") for (size_t i = 0; i < width * prevWidth; i++) W[i] = std::ceil(1000.0*randUniform()) / 1000.0;
	else if (initMethod == "zero") memset(W, 0, width * prevWidth * sizeof(double));
	return this;
}

Layer* Layer::initB(std::string initMethod) {
	if (initMethod == "random") for (size_t i = 0; i < width; i++) B[i] = 2.0 * std::ceil(1000.0*randUniform()) / 1000.0 - 1.0;
	else if (initMethod == "positiveRandom") for (size_t i = 0; i < width; i++) B[i] = std::ceil(1000.0*randUniform()) / 1000.0;
	else if (initMethod == "zero") 	memset(B, 0, width * sizeof(double));
	return this;
}

Layer* Layer::save(std::ofstream* output, bool hex) {
	std::string nonLinearityStr;
	switch (nonLinearity) {
	case LINEAR: nonLinearityStr = "linear"; break;
	case RELU: nonLinearityStr = "relu"; break;
	case SIGMOID: nonLinearityStr = "sigmoid"; break;
	case SOFTMAX_CROSSENTROPY: nonLinearityStr = "softmaxCrossentropy"; break;
	case SIGMOID_MSE: nonLinearityStr = "sigmoidMse"; break;
	case LINEAR_MSE: nonLinearityStr = "linearMse"; break;
	case UNDEFINED: nonLinearityStr = "undefined"; break;
	};

	*output << "BeginLayer," << name << "," << width << "," << nonLinearityStr;
	if (!isTrainable) *output << ",const";
	*output << std::endl;
	if (W != nullptr) {
		*output << "MatrixW," << prevWidth << std::endl;
		for (size_t i = 0; i < width * prevWidth; i++) {
			if (hex) *output << doubleToHexString(W[i]); else *output << W[i];
			if ((i + 1) % width == 0) *output << std::endl; else *output << ",";
		}
	}
	if (B != nullptr) {
		*output << "VectorB" << std::endl;
		for (size_t i = 0; i < width; i++) {
			if (hex) *output << doubleToHexString(B[i]); else *output << B[i];
			if ((i + 1) % width == 0) *output << std::endl; else *output << ",";
		}
	}
	*output << "EndLayer" << std::endl;

	return this;
}

Layer* Layer::saveAll(std::ofstream* output, bool hex) {
	auto l = getInputLayer();
	while ((l = l->nextLayer) != nullptr) l->save(output, hex);
	return this;
}

Layer* Layer::load(std::ifstream* input, bool hex) {
	std::string line, item;
	std::stringstream ss;

	std::getline(*input, line);
	ss.str(line);
	getLine(ss, item, ',');
	if (item != "BeginLayer") return nullptr;
	getLine(ss, item, ',');
	name = item;
	getLine(ss, item, ',');
	width = (unsigned int)paresInt(item);
	getLine(ss, item, ',');;
	if (item == "linear") nonLinearity = LINEAR;
	else if (item == "relu") nonLinearity = RELU;
	else if (item == "sigmoid") nonLinearity = SIGMOID;
	else if (item == "softmaxCrossentropy") nonLinearity = SOFTMAX_CROSSENTROPY;
	else if (item == "sigmoidMse") nonLinearity = SIGMOID_MSE;
	else if (item == "linearMse") nonLinearity = LINEAR_MSE;
	else problemAndExit("Unknown nonlinearity: " + item);

	getLine(ss, item, ',');
	if (item == "const") isTrainable = false;

	std::getline(*input, line);
	ss.clear();
	ss.str(line);
	getLine(ss, item, ',');
	if (item != "MatrixW") return nullptr;
	getLine(ss, item, ',');
	prevWidth = std::stoi(item);

	allocate();

	item = "";
	getLine(ss, item, ',');
	if (item.length() > 0) {
		initW(item);
	}
	else {
		for (size_t n = 0; n < prevWidth; n++) {
			std::getline(*input, line);
			ss.clear();
			ss.str(line);
			for (size_t neuron = 0; neuron < width; neuron++) {
				getLine(ss, item, ',');
				if (hex) W[n * width + neuron] = hexStringToDouble(item); else W[n * width + neuron] = std::stod(item);
			}
		}
	}

	std::getline(*input, line);
	ss.clear();
	ss.str(line);
	getLine(ss, item, ',');
	if (item != "VectorB") return nullptr;

	item = "";
	getLine(ss, item, ',');
	if (item.length() > 0) {
		initB(item);
	}
	else {
		std::getline(*input, line);
		ss.clear();
		ss.str(line);
		for (size_t neuron = 0; neuron < width; neuron++) {
			getLine(ss, item, ',');
			if (hex) B[neuron] = hexStringToDouble(item); else B[neuron] = std::stod(item);
		}
	}
	return this;
}

Layer* Layer::loadAll(std::ifstream* input, bool hex) {
	std::string line;
	std::getline(*input, line); 
	if (line.rfind("BeginLayer", 0) == 0) {
		input->seekg(0, std::ios_base::beg);
		load(input, hex);
		layerType = HIDDEN;
		Layer* inputLayer = new Layer();
		inputLayer->name = "Input";
		inputLayer->layerType = INPUT;
		inputLayer->nextLayer = this;
		inputLayer->width = prevWidth;
		prevLayer = inputLayer;

		Layer* prev = this;
		Layer* l = nullptr;
		unsigned int i = 2;
		while (true) {
			std::string line;
			std::getline(*input, line); // EndLayer
			auto len = input->tellg();
			std::getline(*input, line);
			input->seekg(len, std::ios_base::beg);
			if (line.rfind("BeginLayer", 0) != 0) break;

			Layer* l = new Layer();
			l->load(input, hex);
			l->layerType = HIDDEN;
			l->name = "Hidden-" + std::to_string(i);
			l->prevLayer = prev;
			l->prevWidth = prev->width;
			prev->nextLayer = l;
			prev = l;
			i++;
		}
		prev->layerType = OUTPUT;
		prev->name = "Output";
		return this;
	}
	else {
		input->seekg(0, std::ios_base::beg);
		std::string line, item;
		std::stringstream ss;
		unsigned int nLayer = 0;
		Layer* currentLayer = nullptr;;
		Layer* prevLayer = nullptr;
		while (!input->eof()) {
			std::getline(*input, line);
			if (line.size() == 0) continue;
			nLayer++;

			if (nLayer == 1) currentLayer = this; else currentLayer = new Layer();
			ss.clear();
			ss.str(line);
			getLine(ss, item, ',');
			currentLayer->name = item;

			if (nLayer == 1) {
				getLine(ss, item, ',');
				currentLayer->width = std::stoi(item);
				currentLayer->layerType = INPUT;
				currentLayer->prevLayer = nullptr;
			}
			else {
				getLine(ss, item, ',');
				if (item == "include") {
					getLine(ss, item, ',');
					if (item.length() > 0) {
						auto in = new std::ifstream(item);
						currentLayer->load(in, hex);
						in->close();
					}
				}
				else {
					currentLayer->width = std::stoi(item);
					getLine(ss, item, ',');
					if (item == "linear") currentLayer->nonLinearity = LINEAR;
					else if (item == "relu") currentLayer->nonLinearity = RELU;
					else if (item == "sigmoid") currentLayer->nonLinearity = SIGMOID;
					else if (item == "softmaxCrossentropy") currentLayer->nonLinearity = SOFTMAX_CROSSENTROPY;
					else if (item == "sigmoidMse") currentLayer->nonLinearity = SIGMOID_MSE;
					else if (item == "linearMse") currentLayer->nonLinearity = LINEAR_MSE;
					else problemAndExit("Unknown nonlinearity: " + item);
				}
				
				currentLayer->layerType = HIDDEN;
				currentLayer->prevLayer = prevLayer;
				currentLayer->prevWidth = prevLayer->width;
				prevLayer->nextLayer = currentLayer;
			}
			prevLayer = currentLayer;
		}
		prevLayer->layerType = OUTPUT;
		prevLayer->nextLayer = nullptr;

		auto l = getInputLayer();
		l->allocate();
		while ((l = l->nextLayer) != nullptr) {
			if (l->W == nullptr) {
				l->allocate();
				l->initW("random");
				l->initB("random");
			}
		}
		return prevLayer->getInputLayer();
	}
}

void Layer::feedForwardAndBackward(bool doBackpropagation) {
	if (prevLayer == nullptr) {
		nextLayer->feedForwardAndBackward();
		return;
	}

	auto baseIndex = 0;
	for (size_t neuron = 0; neuron < width; neuron++) {
		activationValue[neuron] = B[neuron];
		for (size_t n = 0; n < prevWidth; n++) activationValue[neuron] += prevLayer->outputValue[n] * W[baseIndex + n];
		baseIndex += prevWidth;
	}

	if (nextLayer != nullptr) {
		if (nonLinearity == SIGMOID) {
			for (size_t neuron = 0; neuron < width; neuron++) {
				outputValue[neuron] = 1.0 / (1.0 + exp(-activationValue[neuron]));
				derivateActivation[neuron] = outputValue[neuron] * (1.0 - outputValue[neuron]);
			}
		}
		else if (nonLinearity == RELU) {
			for (size_t neuron = 0; neuron < width; neuron++) {
				if (activationValue[neuron] > 0) {
					outputValue[neuron] = activationValue[neuron];
					derivateActivation[neuron] = 1;
				}
				else {
					outputValue[neuron] = 0;
					derivateActivation[neuron] = 0;
				}
			}
		}
		else if (nonLinearity == LINEAR) {
			for (size_t neuron = 0; neuron < width; neuron++) {
				outputValue[neuron] = activationValue[neuron];
				derivateActivation[neuron] = 1;
			}
		}
		else problemAndExit("Unknown nonlinearity: " + nonLinearity);

		nextLayer->feedForwardAndBackward();
	}
	else {
		loss = 0;
		if (nonLinearity == SOFTMAX_CROSSENTROPY) {
			// target value should be 0.0 or 1.0
			double denom = 0;
			double maxValue = -1e308;
			unsigned int maxIndex = 0;
			for (size_t neuron = 0; neuron < width; neuron++) if (maxValue < activationValue[neuron]) maxValue = activationValue[neuron];
			for (size_t neuron = 0; neuron < width; neuron++) {
				activationValue[neuron] = activationValue[neuron] - maxValue;
				denom += exp(activationValue[neuron]);
			}
			for (size_t neuron = 0; neuron < width; neuron++) {
				outputValue[neuron] = exp(activationValue[neuron]) / denom;
				if (targetValue != nullptr) {
					backpropagetedError[neuron] = outputValue[neuron] - targetValue[neuron];
					if (targetValue[neuron] > 0.95) {
						if (outputValue[neuron] < 1e-10) loss += 24; else loss -= log(outputValue[neuron]);
					}
					else {
						if (outputValue[neuron] > 0.9999999999) loss += 24; else loss -= log(1.0 - outputValue[neuron]);
					}
				}
			}
		}
		else if (nonLinearity == SIGMOID_MSE) {
			for (size_t neuron = 0; neuron < width; neuron++) {
				outputValue[neuron] = 1.0 / (1.0 + exp(-activationValue[neuron]));
				if (targetValue!=nullptr) {
					backpropagetedError[neuron] = (outputValue[neuron] - targetValue[neuron]) * outputValue[neuron] * (1.0 - outputValue[neuron]);
					double dv = targetValue[neuron] - outputValue[neuron];
					loss += dv * dv;
				}
			}
			loss /= 2.0;
		}
		else if (nonLinearity == LINEAR_MSE) {
			for (size_t neuron = 0; neuron < width; neuron++) {
				outputValue[neuron] = activationValue[neuron];
				if (targetValue != nullptr) {
					backpropagetedError[neuron] = outputValue[neuron] - targetValue[neuron];
					double dv = targetValue[neuron] - outputValue[neuron];
					loss += dv * dv;
				}
			}
			loss /= 2.0;
		}
		else problemAndExit("Unknown nonlinearity: " + nonLinearity);

		if (doBackpropagation) backpropagate();
	}
}

void Layer::backpropagate() {
	if (prevLayer != nullptr) {
		if (nextLayer != nullptr) {
			for (size_t neuron = 0; neuron < width; neuron++) {
				double value = 0.0;
				for (size_t i = 0; i < nextLayer->width; i++) value += nextLayer->backpropagetedError[i] * nextLayer->W[i * width + neuron];
				backpropagetedError[neuron] = value * derivateActivation[neuron];
			}
		}
		auto baseIndex = 0;
		for (size_t neuron = 0; neuron < width; neuron++) {
			for (size_t i = 0; i < prevWidth; i++) dW[baseIndex + i] += backpropagetedError[neuron] * prevLayer->outputValue[i];
			baseIndex += prevWidth;
			dB[neuron] = backpropagetedError[neuron];
		}
		prevLayer->backpropagate();
	}
};