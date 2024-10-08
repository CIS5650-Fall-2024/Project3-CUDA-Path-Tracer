#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "glm/glm.hpp"
#include "sceneStructs.h"

//Doing memory innefficiently, but it will make everything simpler.
//We will store the triangles explicitly
struct Tri {

    glm::vec3 vertices[3];
    glm::vec3 midpoint;
    glm::vec3 color;
    glm::vec3 normal;
    float emittance;
    float reflectivity;
    float roughness;
};

struct FloatInt {
    float f;
    int i;
};

struct OctTreeRegion{
    Geom boundingBox;
    int start_tri_ind;
    int depth;
    int length;
};

__device__ FloatInt getTriIntersectionLinear(
    Ray r,
    Tri* triangles,
    int numTriangles,
    float t_min
);

__device__ FloatInt getTriIntersectionOctTree(
    Ray r,
    Tri* triangles,
    int numTriangles,
    OctTreeRegion* octRegions,
    int numOcts,
    float t_min
);

void readOBJ(const std::string& filename, std::vector<Tri>& triangles, glm::vec3 scale, glm::vec3 trans);

int objTest(std::vector<glm::vec3> vertices, std::vector<Tri> triangles);


//void formOctreeTrianglesArray(Tri* triangles, int numTriangles, OctTreeRegion* octrees, int numOcts, int currDepth, int maxDepth, int levelInd, int numOctreesOnLevel, int startTriInd);
void formOctreeTrianglesArray(Tri* triangles, int numTriangles, OctTreeRegion* octrees,
                              int currDepth, int maxDepth, int curr_node_index, int startTriInd);

