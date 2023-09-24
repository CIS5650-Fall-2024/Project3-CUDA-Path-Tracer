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

class Scene {
private:
    string basePath;

    ifstream fp_in;
    int loadMaterial(string materialId);
    int loadGeom(string objectid);
    int loadCamera();

    int loadMesh(string filePath);
    int buildBvh(int startTri, int numTris);
    void bvhUpdateNodeBounds(BvhNode& node);
    void bvhSubdivide(BvhNode& node);

    unordered_map<string, int> meshIndices;

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::vector<Triangle> tris;
    std::vector<BvhNode> bvhNodes;
    std::vector<int> bvhTriIdx;
    RenderState state;
};
