#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"

#define DEBUG 1

#define MAX_DEPTH 20

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    int loadGlTF(const string &fullPath, Geom &geom);
    int loadObj(const std::string& fullPath, Geom& geom);

    void buildBVH();
    int buildBVHRecursive(int start, int count, int depth);
    int flattenBVHTree(int nodeIndex, int* offset);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // mesh loading
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvh;
    std::vector<LinearBVHNode> linearBVH;
};
