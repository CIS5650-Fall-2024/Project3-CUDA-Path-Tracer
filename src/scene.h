#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <unordered_map>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void InitializeCameraAndRenderState();
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName);
    void transformToTarget(const glm::vec3& bboxCenter, const glm::vec3& bboxScale);
public:
    Scene();
    ~Scene();
    void LoadFromFile(string filename);
    void autoCentralize();
    bool sceneReady = false;

    Mesh mesh;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::unordered_map<std::string, uint32_t> MatNameToID;

    // other render options
    bool renderWithPathTracing = true;
    bool sortByMaterial = false;
    // other mesh options
    bool autoCentralizeObj = true;
};
