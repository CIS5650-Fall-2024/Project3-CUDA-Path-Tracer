#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "bsdf.h"
#include "light.h"
#include "bvh.h"
#include "tiny_obj_loader.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Primitive> prims;
    std::vector<BVHNode> bvh;
    std::vector<BSDF> materials;
    std::vector<Light> lights;
    RenderState state;
};
