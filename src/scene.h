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
    int triangle_count{ 0 };

    std::vector<Texture> textures;
    std::vector<Texture> bumpmaps;
    std::unique_ptr<Texture> environmentmap;
    glm::vec2 environmentmap_dim{ 0 };

    int overall_tex_idx{ 0 };
    int overall_bump_idx{ 0 };
    std::vector<int> tex_starts{ 0 };
    std::vector<int> bump_starts{ 0 };

    std::vector<glm::vec2> tex_dims;
    std::vector<glm::vec2> bump_dims;

    std::vector<BVHNode> bvhNodes;

    RenderState state;

    void constructBVH();
    void updateNodeBounds(BVHNode& node);
    void subdivide(BVHNode& node);
    float evaluateSAH(BVHNode& node, int axis, float pos);
    float findBestSplitPlane(BVHNode& node, int& axis, float& splitPos);
};
