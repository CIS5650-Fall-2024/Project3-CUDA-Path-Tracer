#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <fstream>
#include <cstring>
#include <unordered_map>

#include "glm/glm.hpp"
#include "json.hpp"

#include "utilities.h"
#include "sceneStructs.h"
#include "meshLoader.h"
#include "bvh.h"

using namespace std;
using json = nlohmann::json;

class Mesh {   
public:
    Mesh();
    ~Mesh();

    std::vector<Triangle> faces;
    std::vector<int> indices;
    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;

    // The glm::vec4* in each of this vars is deleted in pathrace.cu once they are loaded to the GPU
    std::vector<tuple<std::string, glm::vec4*, glm::ivec2>> albedoTextures;
    std::vector<tuple<std::string, glm::vec4*, glm::ivec2>> normalTextures;
    std::vector<tuple<std::string, glm::vec4*, glm::ivec2>> bumpTextures;
};

class Scene
{
private:
    ifstream fp_in;

    void loadMesh(const std::string &filepath, Mesh &mesh);
    void loadFromJSON(const std::string& jsonName);
    template <typename T>
    void getValueFromJson(const json &data, const std::string &key, T &value);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> lights;
    std::vector<Material> materials;
    std::vector<tuple<glm::vec4*, glm::ivec2>> albedoTextures;
    std::vector<tuple<glm::vec4*, glm::ivec2>> normalTextures;
    std::vector<tuple<glm::vec4*, glm::ivec2>> bumpTextures;
    RenderState state;
};
