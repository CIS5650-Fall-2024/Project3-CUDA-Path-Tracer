#pragma once

#include <cstring>  // For memcpy
#include <tuple>
#include <limits>
#include <unordered_map>

#include "glm/glm.hpp"

#include "sceneStructs.h"

class NodeList {
private:
    // Resize the Nodes array when the capacity is exceeded
    void resize(int newCapacity);

public:
    BVHNode* nodes;      // Pointer to dynamic array of Nodes
    int index;        // Current index to track number of nodes
    int capacity;     // Current capacity of the Nodes array

    // Constructor initializes array with initial capacity of 256 nodes
    NodeList();
    ~NodeList();

    // Add a Node to the list, resize if necessary
    int addNode(const BVHNode& node);

    // Return the current number of nodes
    const int nodeCount();
};

class BVH {
private:
    BVHTriangle* allBvhTriangles;

    float lerp(float a, float b, float t);

    // Calculate the size of the bounding box
    glm::vec3 computeNodeBoundsSize(const BVHNode& node);
    // Calculate the cost of a node to decide if it should be split
    float computeNodeCost(const glm::vec3& size, int numTriangles);
    float evalSplit(int splitAxis, float splitPos, int start, int count);
    SplitResult chooseSplit(const BVHNode& node, int start, int count);
    void splitNode(int parentIndex, const glm::vec3* verts, int triGlobalStart, int triNum, int depth = 0);

public:
    NodeList allNodes;
    Triangle* allTriangles;

    BVH();
    BVH(const glm::vec3* verts, const glm::vec3* normals, const int* indices, int indexCount);
    ~BVH();
};
