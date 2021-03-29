#include "UtilLayer.h"

Layer* interruptableDnn = nullptr;

#ifdef _WIN32
BOOL WINAPI signalHandler(DWORD signal) {
	if (signal == CTRL_C_EVENT) {
		std::cerr << std::endl << "CTRL_C_EVENT" << std::endl;
		if (interruptableDnn != nullptr) interruptableDnn->toExit = true;
	}
	else if (signal == CTRL_BREAK_EVENT) {
		std::cerr << std::endl << "CTRL_BREAK_EVENT" << std::endl;
		if (interruptableDnn != nullptr) interruptableDnn->toDump = true;
	}
	return TRUE;
}
#else
void signalHandler(int signal) {
	if (signal == SIGINT) {
		std::cerr << std::endl << "SIGINT" << std::endl;
		if (interruptableDnn != nullptr) interruptableDnn->toExit = true;
	}
}
#endif

int main(int argc, const char *argv[]) {
	if (argc < 4) problemAndExit("trainDnn <trainData[:<startIndexA>:<startIndexB>]> <dnnDescriptor> <modelVersionId> <epoch> [expand]");

#ifdef _WIN32
	SetConsoleCtrlHandler(signalHandler, TRUE);
#else
	struct sigaction sigactionHandler;
	sigactionHandler.sa_handler = signalHandler;
	sigemptyset(&sigactionHandler.sa_mask);
	sigactionHandler.sa_flags = 0;
	sigaction(SIGINT, &sigactionHandler, NULL);
#endif

	int modelVersionId = (int)paresInt(argv[3]);
	int epoch = (int)paresInt(argv[4]);
	if (modelVersionId < 0 || epoch <= 0) problemAndExit("Invalid parmeter!");

	auto ModelFile = new std::ifstream(std::string(argv[2]) + "-" + std::to_string(modelVersionId) + ".txt");
	if (ModelFile->fail()) problemAndExit("Model file does not exists!");
	auto DNN = Layer::getLayer()->loadAll(ModelFile, true);
	ModelFile->close();
	auto InputLayer = DNN->getInputLayer();
	auto OutputLayer = DNN->getOutputLayer();
	interruptableDnn = InputLayer;

	int indexA, indexB;
	if (argc < 6) {
		indexA = 0;
		indexB = InputLayer->width;
	}
	else {
		indexA = 1;
		indexB = 0;
	}
	auto inputDescriptor = split(argv[1], ":");
	auto InputFileName = inputDescriptor[0];
	if (inputDescriptor.size() > 1) indexA = (int)paresInt(inputDescriptor[1]);
	if (inputDescriptor.size() > 2) indexB = (int)paresInt(inputDescriptor[2]);
	if (indexA < 0 || indexB < 0) problemAndExit("Invalid index parmeter!");
	auto InputFile = new std::ifstream(InputFileName);
	if (InputFile->fail()) problemAndExit("Input file does not exists!");

	new std::thread([] {
		while (true) {
			std::this_thread::sleep_for(std::chrono::seconds(120));
			interruptableDnn->toDump = true;
		}
	});

	double* dataInput = nullptr;
	double* dataTarget = nullptr;
	setSeed(SystemCurrentTimeMillis());
	if (argc < 6) {
		auto nData = loadCsvFile(InputFile, 0, dataInput, InputLayer->width, nullptr, dataTarget, OutputLayer->width, nullptr, indexA, indexB);
		InputFile->close();
		if (nData <= 0) problemAndExit("No data!");
		trainADAM(DNN, nData, dataInput, dataTarget, 32, epoch, 0.001);
	}
	else {
	    auto nData = loadCsvFile(InputFile, 0, dataInput, InputLayer->width, nullptr, dataTarget, 1, nullptr, indexA, indexB);
		InputFile->close();
		if (nData <= 0) problemAndExit("No data!");
		double* dataTargetCategory = nullptr;
		expandCategories(nData, OutputLayer->width, dataTarget, dataTargetCategory);
		trainADAM(DNN, nData, dataInput, dataTargetCategory, 32, epoch, 0.001);
		delete[] dataTargetCategory;
	}

	DNN->saveAll(new std::ofstream(std::string(argv[2]) + "-" + std::to_string(modelVersionId + 1) + ".txt"), true);

	Layer::deleteAllLayer(DNN);
	delete[] dataInput;
	delete[] dataTarget;
	return 0;
}