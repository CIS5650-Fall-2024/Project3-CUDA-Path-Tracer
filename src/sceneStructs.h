#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
};

struct Triangle {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    glm::vec3 centroid;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    Triangle* tris;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        //float exponent;
        //glm::vec3 color;
        bool isSpecular;
        glm::vec3 kd;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};


struct bbox {
    bbox() {
        bmin = glm::vec3{ 1e30f };
        bmax = glm::vec3{ -1e30f };
    }

    glm::vec3 bmin, bmax;

    __host__ __device__ void grow(glm::vec3 p) { 
        bmin = glm::vec3{ glm::min(bmin.x, p.x), glm::min(bmin.y, p.y), glm::min(bmin.z, p.z) };
        bmax = glm::vec3{ glm::max(bmax.x, p.x), glm::max(bmax.y, p.y), glm::max(bmax.z, p.z) };
    }
    __host__ __device__ float area()
    {
        glm::vec3 e = bmax - bmin; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

struct BVHNode {
    BVHNode() {
        aabb = bbox();
        leftFirst = 0;
        triCount = 0;
    }

    bbox aabb;          // 24 bytes - aabb can be defined with 6 floats
    unsigned int leftFirst, triCount;    // 8 bytes; total: 32 bytes
    __host__ __device__ bool isLeaf() { return triCount > 0; }
};
