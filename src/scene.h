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
    void loadFromOBJ(const std::string& objName, int materialID);
    void loadFromTexture(const std::string& textureName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Vertex> vertices;
    std::vector<Texture> textures;
    std::vector<glm::vec4> texturePixels;
    RenderState state;
};
