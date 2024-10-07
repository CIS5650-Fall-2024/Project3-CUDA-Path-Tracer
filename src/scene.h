#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void LoadFromOBJ(const std::string& fileName, Geom& geom);
    AABB calculateAABB(const Triangle& tri);
    int buildBVH(std::vector<BVHNode>& nodes, std::vector<Triangle>& triangles, int start, int end);
    bool loadTexture(const std::string& filename, TextureData& textureData);
    bool loadTexture_hdr(const std::string& filename, EnvData_hdr& EnvData_hdr);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
