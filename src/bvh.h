#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    __host__ __device__ AABB() : min(glm::vec3(FLT_MAX)), max(glm::vec3(-FLT_MAX)) {}
    __host__ __device__ AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}
    __host__ __device__ AABB(const glm::vec3& point1, const glm::vec3& point2, const glm::vec3& point3) {
        min = glm::min(point1, glm::min(point2, point3));
        max = glm::max(point1, glm::max(point2, point3));
    }

    __host__ __device__ void Union(const AABB& input_aabb) {
        min = glm::min(min, input_aabb.min);
        max = glm::max(max, input_aabb.max);
    }

    __host__ __device__ int LongestAxisIndex() const {
        glm::vec3 diagonal = (max - min);
        int maxAxis = (diagonal.x > diagonal.y) ? 0 : 1;
        return (diagonal[maxAxis] > diagonal.z) ? maxAxis : 2;
    }

    __host__ __device__ float SurfaceArea() const {
        glm::vec3 d = (max - min);
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

struct BVH_Node {
    AABB aabb;
    int leftChild = -1;
    int rightChild = -1;
    int splitAxis = -1;
    int firstT_idx = -1; // Index of the first triangle in the leaf node
    int TCount = -1;     // Number of triangles in this node
};

class BVH {
public:
    std::vector<BVH_Node> nodes;

    void buildBVH(std::vector<Triangle>& triangles);
private:
    void subdivide(int nodeIdx, std::vector<Triangle>& triangles, int start, int end);
    int findOptimalSplit(std::vector<Triangle>& triangles, int start, int end, int axis, const AABB& parentAABB);
};

AABB computeBoundingBox(const Geom& geom, const std::vector<Vertex>& vertices);
