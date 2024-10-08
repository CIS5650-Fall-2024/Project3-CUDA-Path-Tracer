#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "stb_image.h"

using namespace std;
#define STB_IMAGE_IMPLEMENTATION

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Vertex> vertices;
    std::vector<Texture> textures;
    EnvTexture environmentMap;
    RenderState state;
};
