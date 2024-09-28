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
    std::vector<Triangle> assembleMesh(std::string& inputfile, std::string& basestring, glm::mat4& transform, glm::mat4& inv_transpose_transform);
    
    unsigned int rootNodeIdx{ 0 };
    unsigned int nodesUsed{ 1 };
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;

    std::vector<Geom> meshes;
    std::vector<Triangle> mesh_triangles;
    int triangle_count;

    std::vector<Texture> textures;
    std::vector<Texture> bumpmaps;

    std::vector<BVHNode> bvhNodes;

    RenderState state;

    void constructBVH();
    void updateNodeBounds(BVHNode& node);
    void subdivide(BVHNode& node);
    float evaluateSAH(BVHNode& node, int axis, float pos);
};
