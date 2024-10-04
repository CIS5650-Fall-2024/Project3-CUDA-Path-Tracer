#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cudaUtilities.h"

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f



class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}

template<typename T>
__inline__ __device__ T Lerp(const T& a, const T& b, float t) {
    return (1.0f - t) * a + t * b;
}

template<typename T>
__inline__ __device__ T Clamp(const T& a, const T& edge0, const T& edge1) {
	return glm::clamp(a, edge0, edge1);
}

template<typename T>
__inline__ __device__ T Square(const T& a) {
	return a * a;
}

// Hash function to generate a random number in [0, 1]
__inline__ __device__ float hash01(uint32_t seed) {
    seed ^= seed >> 21;
    seed ^= seed << 35;
    seed ^= seed >> 4;
    seed *= 2685821657736338717ull;
    return (seed & 0xFFFFFF) / float(0xFFFFFF);
}