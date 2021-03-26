#pragma once

#include "Util.h"

class Layer {

public:

	enum LayerType { NONE, INPUT, HIDDEN, OUTPUT };
	enum NonLinearity { UNDEFINED, LINEAR, RELU, SIGMOID, SOFTMAX_CROSSENTROPY, SIGMOID_MSE, LINEAR_MSE };

	std::string name{ "noName" };
	LayerType layerType{ NONE };
	unsigned int width{ 0 };
	unsigned int prevWidth{ 0 };
	double* W{ nullptr };  // matrix width is the number of neurons of this layer, number of rows is the number of the neurons of the prevouse layer
	double* B{ nullptr };
	double* activationValue{ nullptr };
	double* outputValue{ nullptr };
	NonLinearity nonLinearity{ UNDEFINED };

	double* derivateActivation{ nullptr };
	double* backpropagetedError{ nullptr};
	double* targetValue{ nullptr };
	double loss{ 0.0 };

	double* dW{ nullptr };
	double* dB{ nullptr };

	Layer* prevLayer{ nullptr };
	Layer* nextLayer{ nullptr };

	double* adamWM{ nullptr };
	double* adamWV{ nullptr };
	double* adamBM{ nullptr };
	double* adamBV{ nullptr };

	bool isTrainable{ true };

	volatile bool toExit{ false };
	volatile bool toDump{ false };

	Layer();
	~Layer();
	
	static void deleteAllLayer(Layer* l);
	static Layer* getLayer();

	Layer* getInputLayer();
	Layer* getOutputLayer();
	Layer* allocate();
	Layer* initW(std::string initMethod);
	Layer* initB(std::string initMethod);
	Layer* save(std::ofstream* output, bool hex = true);
	Layer* saveAll(std::ofstream* output, bool hex);
	Layer* load(std::ifstream* input, bool hex = true);
	Layer* loadAll(std::ifstream* input, bool hex);
	void feedForwardAndBackward(bool doBackpropagation = true);

private:

	void backpropagate();

};