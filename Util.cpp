#include "Util.h"

Xorshift64_state state;

void setSeed(uint64_t seed) {
	state.a = seed;
}

uint64_t rand64(struct Xorshift64_state *state) {
	uint64_t x = state->a;
	x ^= x << 13; x ^= x >> 7; x ^= x << 17;
	return state->a = x;
}

double randUniform() { return ((double)rand64(&state)) / 1.8446744073709551616e+19; }

uint64_t randUniform64() { return rand64(&state); }

long long paresInt(std::string item) {
	try { return std::stoi(item); }
	catch (const std::exception&) { return INT_MIN; }
}

double paresDouble(std::string item) {
	try { return std::stod(item); }
	catch (const std::exception&) { return NAN; }
}

void getLine(std::stringstream& ss, std::string& item, char delimiter){
	std::getline(ss, item, delimiter);
	item.erase(remove_if(item.begin(), item.end(), isspace), item.end());
}

void printArrayToBmp(int w, int h, unsigned char* R, unsigned char* G, unsigned char* B, std::string aFileName) {
	auto output = std::ofstream(aFileName, std::ofstream::binary);
	unsigned char header[14] = { 'B','M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char info[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
	int paddingSize = (4 - (w * 3) % 4) % 4;
	unsigned int filesize = 54 + 3 * w * h + paddingSize * h;
	header[2] = (unsigned char)(filesize);
	header[3] = (unsigned char)(filesize >> 8);
	header[4] = (unsigned char)(filesize >> 16);
	header[5] = (unsigned char)(filesize >> 24);
	info[4] = (unsigned char)(w);
	info[5] = (unsigned char)(w >> 8);
	info[6] = (unsigned char)(w >> 16);
	info[7] = (unsigned char)(w >> 24);
	info[8] = (unsigned char)(h);
	info[9] = (unsigned char)(h >> 8);
	info[10] = (unsigned char)(h >> 16);
	info[11] = (unsigned char)(h >> 24);
	info[12] = (unsigned char)(1);
	info[14] = (unsigned char)(3 * 8); // 3 byte 

	for (auto i = 0; i < 14; i++) output << header[i];
	for (auto i = 0; i < 40; i++) output << info[i];

	if (G == nullptr) G = R;
	if (B == nullptr) B = R;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) output << B[(y * w) + x] << G[(y * w) + x] << R[(y * w) + x];
		for (int i = 0; i < paddingSize; i++) output << (unsigned char)(0);
	}
}

void testPrintArrayToBmp() {
	auto R = new unsigned char[512 * 480];
	auto G = new unsigned char[512 * 480];
	auto B = new unsigned char[512 * 480];

	memset(R, 0x00, 512 * 480);
	memset(G, 0x00, 512 * 480);
	memset(B, 0x00, 512 * 480);

	for (size_t x = 0; x < 512; x++) {
		for (size_t y = 0; y < 120; y++) R[(y * 512) + x] = (unsigned char)(x / 2);
		for (size_t y = 120; y < 240; y++) G[(y * 512) + x] = (unsigned char)(x / 2);
		for (size_t y = 240; y < 360; y++) B[(y * 512) + x] = (unsigned char)(x / 2);
		for (size_t y = 360; y < 480; y++) R[(y * 512) + x] = G[(y * 512) + x] = B[(y * 512) + x] = (unsigned char)(x / 2);
	}

	printArrayToBmp(512, 480, R, G, B, "test.bmp");

	delete[] R; delete[] G; delete[] B;
}

int loadCsvFile(std::ifstream* file, unsigned int numberOfLines, double* &a, unsigned int nDataA, std::function<double(double)> fA, double* &b, unsigned nDataB, std::function<double(double)> fB, int startColA, int startColB) {
	std::cerr << "Loading file..." << std::endl;
	try {
		auto nLine = numberOfLines;
		if (nLine == 0) nLine = (unsigned int)std::count(std::istreambuf_iterator<char>(*file), std::istreambuf_iterator<char>(), '\n');
		std::cerr << "Number of lines: " << nLine << std::endl;
		if (a == nullptr) a = new double[nLine * nDataA];
		if (b == nullptr && nDataB > 0) b = new double[nLine * nDataB];
		unsigned int indexA, indexB; indexA = indexB = 0;
		std::string line;
		file->clear();
		file->seekg(0);
		std::getline(*file, line); // Skip first line
		std::cerr << "First line is skipped" << std::endl;
		std::cerr << "Processing file..." << std::endl;
		unsigned int dataLine = 0;
		if (startColA < 0) startColA = 0;
		if (startColB < 0) startColB = startColA + nDataA;
		while (std::getline(*file, line) && dataLine < nLine) {
			dataLine++;
			if (dataLine % 100 == 0) std::cerr << '\r' << dataLine;
			std::string item;
			std::stringstream ss;
			ss.str(line);
			auto position = 0;
			unsigned int offsetA, offsetB;
			for (std::string item; std::getline(ss, item, ',');) {
				double value;
				value = std::stod(item); 
				offsetA = position - startColA;
				if (offsetA >= 0 && offsetA < nDataA) {
					if (fA != nullptr) value = fA(value);
					a[indexA + offsetA] = value;
				}
				offsetB = position - startColB;
				if (offsetB >= 0 && offsetB < nDataB) {
					if (fB != nullptr) value = fB(value);
					b[indexB + offsetB] = value;
				}
				position++;
			}
			indexA += nDataA;
			indexB += nDataB;
		}
		std::cerr << '\r' << "File is loaded." << std::endl;
		return dataLine;
	}
	catch (const std::exception &e) { problemAndExit("Failed to load file: " + std::string(e.what())); }
}

void expandCategories(unsigned int nData, unsigned int outputWidth, double* input, double* &output) {
	if (output == nullptr) output = new double[nData * outputWidth];
	std::fill_n(output, nData * outputWidth, 0.0);
	for (size_t i = 0; i < nData; i++) output[i * outputWidth + (unsigned int)(round(input[i]))] = 1.0;
}

void problemAndExit(std::string message) {
	std::cerr << message << std::endl;
	exit(1);
}

const char* charToBinaryString(char c) {
	switch (toupper(c)) {
	case '0': return "0000";
	case '1': return "0001";
	case '2': return "0010";
	case '3': return "0011";
	case '4': return "0100";
	case '5': return "0101";
	case '6': return "0110";
	case '7': return "0111";
	case '8': return "1000";
	case '9': return "1001";
	case 'A': return "1010";
	case 'B': return "1011";
	case 'C': return "1100";
	case 'D': return "1101";
	case 'E': return "1110";
	case 'F': return "1111";
	}
	return nullptr;
}

std::string hexStringToBinaryString(const std::string& hex) {
	std::stringstream ss;
	for (unsigned i = 0; i != hex.length(); ++i) ss << charToBinaryString(hex[i]);
	return ss.str();
}

std::string doubleToHexString(double d) {
	unsigned long long bits{ *reinterpret_cast<unsigned long long*>(&d) };
	std::bitset<sizeof(double) * 8> b(bits);
	auto binaryString{ b.to_string() };
	int value{ 0 };
	std::string res;
	for (auto i = 0; i < binaryString.length(); i++) {
		value = value * 2 + (binaryString.at(i) - '0');
		if (i > 0 && (i + 1) % 4 == 0) {
			if (value <= 9) res = res + (char)(value + '0');
			else res = res + (char)(value - 10 + 'A');
			value = 0;
		}
	}
	return res;
}

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
	if (from.empty()) return;
	size_t start_pos{ 0 };
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length();
	}
}

double hexStringToDouble(const std::string& hexString, bool bigEndian) {
	std::string msg(hexString);
	replaceAll(msg, " ");
	replaceAll(msg, "-");
	std::string hexStringCorrected{ msg };
	if (!bigEndian) {
		hexStringCorrected.clear();
		for (size_t i = 0; i < 8; i++) {
			hexStringCorrected += msg[14 - i * 2];
			hexStringCorrected += msg[15 - i * 2];
		}
	}
	auto binaryString{ hexStringToBinaryString(hexStringCorrected) };
	unsigned long long x{ 0 };
	for (size_t i = 0; i < 64; i++) x = (x << 1) + ((unsigned long long)binaryString[i] - '0');
	double d{ 0 };
	memcpy(&d, &x, 8);
	return d;
}

long long SystemCurrentTimeMillis() {
	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
	long long value = ms.count();
	return  ms.count();
}

std::vector<std::string>& split(std::string message, std::string delim) {
	std::regex rgx(delim);
	std::sregex_token_iterator iter(message.begin(), message.end(), rgx, -1);
	std::sregex_token_iterator end;
	std::vector<std::string>* res = new std::vector<std::string>();
	int i = 0;
	for (; iter != end; ++iter) {
		res->resize(i + 1);
		(*res)[i] = (*iter);
		i++;
	}
	return *res;
}