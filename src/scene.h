#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;






struct BVH_Node
{
    AABB aabb;       // The axis-aligned bounding box that encloses this node.
    int leftChild = -1;     // Index of the left child node.
    int rightChild = -1;    // Index of the right child node.

    int splitAxis = -1;     // The axis along which the node is split (0: x, 1: y, 2: z).
    int fisrtT_idx = -1; // the first triangle in the leaf node.
    int TCount = -1;       // Number of traingles contained in the node.
};


struct BVH_TriangleAABB_relation
{
    int triangle_id;
    AABB aabb;

    __host__ __device__ BVH_TriangleAABB_relation() : triangle_id(-1), aabb(AABB()) {}

    __host__ __device__ BVH_TriangleAABB_relation(int id, const AABB& aabb_in) : triangle_id(id), aabb(aabb_in) {}
};




static AABB combine2AABB(const AABB& aabb1, const AABB& aabb2) {
    return AABB(
        glm::min(aabb1.min, aabb2.min),  // Get the minimum point for the new bounding box
        glm::max(aabb1.max, aabb2.max)   // Get the maximum point for the new bounding box
    );
}




static AABB computeBoundingBox(const std::vector<Vertex>& vertices) {
    glm::vec3 min(FLT_MAX);
    glm::vec3 max(-FLT_MAX);

    for (const auto& vertex : vertices) {
        min = glm::min(min, vertex.pos);
        max = glm::max(max, vertex.pos);
    }

    return AABB(min, max);
}


static AABB transformAABB(const AABB& box, const glm::mat4& transform) {
    glm::vec3 worldMin(FLT_MAX);
    glm::vec3 worldMax(-FLT_MAX);

    // Transform all 8 corners of the AABB
    for (int x = 0; x <= 1; ++x) {
        for (int y = 0; y <= 1; ++y) {
            for (int z = 0; z <= 1; ++z) {
                glm::vec3 corner = box.min + glm::vec3(x, y, z) * (box.max - box.min);
                glm::vec3 worldCorner = glm::vec3(transform * glm::vec4(corner, 1.0f));
                worldMin = glm::min(worldMin, worldCorner);
                worldMax = glm::max(worldMax, worldCorner);
            }
        }
    }

    return AABB(worldMin, worldMax);
}




class Scene
{
private:
    ifstream fp_in;

    void loadFromJSON(const std::string& jsonName);
    bool loadShapeFromOBJ(const std::string& filename, std::vector<Vertex>& vertices, std::vector<Triangle>& triangles);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;


    std::vector<Triangle> BVH_triangles;
    std::vector<BVH_Node> BVH_Nodes;


    int constructBVH(int, int);
    void updateBVH(const std::vector<BVH_TriangleAABB_relation>&, std::vector<BVH_Node>&, int);

    int Scene::subdivide(std::vector<BVH_TriangleAABB_relation>&, std::vector<BVH_Node>&, int, int&, int&);
};





