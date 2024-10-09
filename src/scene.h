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
    void loadFromJSON(const std::string& jsonName);

    void expandBounds(int start, int end, glm::vec3& mins, glm::vec3& maxs);
    int getSubtreeSize(int nodeIndex);

    void loadFromOBJ(const std::string& objName, std::vector<glm::vec3>& verts, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs, std::vector<std::string>& matNames, std::unordered_map<std::string, uint32_t>& MatNameToID);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<glm::vec3> textures;
    std::vector<glm::vec3> env;
    int env_width;
    int env_height;
    RenderState state;

    std::vector<BVHNode> nodes;
    std::vector<int> indices;
    void constructBVHTree();
    void buildBVHTree(int start, int end);
};
