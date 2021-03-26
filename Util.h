#pragma once

#include <bitset>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream> 
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <functional>
#include <iomanip>
#include <tuple>

#ifdef _WIN32
    #include <windows.h> 
#else
    #include <limits.h>
    #include <math.h>
    #include <signal.h>
#endif

struct Xorshift64_state { uint64_t a = 32; };

void setSeed(uint64_t seed);
uint64_t rand64(struct Xorshift64_state *state);
double randUniform();
uint64_t randUniform64();

long long paresInt(std::string item);
double paresDouble(std::string item);
void getLine(std::stringstream& ss, std::string& item, char delimiter);

void printArrayToBmp(int w, int h, unsigned char* R, unsigned char* G, unsigned char* B, std::string aFileName);
void testPrintArrayToBmp();

int loadCsvFile(std::ifstream* file, unsigned int numberOfLines, double* &a, unsigned int nDataA, std::function<double(double)> fA,  double* &b, unsigned nDataB, std::function<double(double)> fB, int startColA = -1, int startColB = -1);
void expandCategories(unsigned int nData, unsigned int outputWidth, double* input, double* &output);

void problemAndExit(std::string message);
const char* charToBinaryString(char c);
std::string hexStringToBinaryString(const std::string& hex);
std::string doubleToHexString(double d);
void replaceAll(std::string& str, const std::string& from, const std::string& to = "");
double hexStringToDouble(const std::string& hexString, bool bigEndian = true);
long long SystemCurrentTimeMillis();
std::vector<std::string>& split(std::string message, std::string delim);