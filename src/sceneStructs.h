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
    OBJ, // read obj file
    ERROR
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
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
};

// Mesh Data uses for storing all the data info when parsing obj file
struct Mesh_Data 
{
    glm::vec3 point;
    glm::vec3 normal;
    glm::vec2 coordinate;
    glm::vec3 tangent;
    int material;
};
struct BVH_Data
{
    std::vector<Mesh_Data> bvh_mesh_data;
    glm::vec3 center;
    float radius;
    int child_indices[2]{ -1, -1 };
};
struct BVH_Main_Data
{
    int index;
    int count;
    glm::vec3 center;
    float radius;
    int child_indices[2]{ -1, -1 };
};
struct Texture_Data 
{
    int width;
    int height;
    int index;
};


struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    int albedo = -1;
    int normal = -1;
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
  glm::vec3 tangent;
  glm::vec2 uvCoord;
  int materialId;
};

struct checkPathComplete {
    __host__ __device__ bool operator()(const PathSegment& pathSegment) {
        return pathSegment.remainingBounces <= 0;
    }
};

struct compareIntersections {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};