#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define ONE_OVER_PI       0.3183098861837906912164442019275156781077f
#define ONE_OVER_TWO_PI   0.1591549430918953357688837633725143620344f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f
#define INF_F             (std::numeric_limits<float>::infinity())

#define StochasticAntialiasing 0
#define DOF 0
#define BVH 1
#define NEE 0
#define RussianRoulette 0

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

class Timer {
public:
    inline void start() 
    {
        t0 = std::chrono::steady_clock::now();
    }

    inline void stop()
    {
        t1 = std::chrono::steady_clock::now();
    }

    inline double duration()
    {
        return (std::chrono::duration<double>(t1 - t0)).count();
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> t0;
    std::chrono::time_point<std::chrono::steady_clock> t1;
};
