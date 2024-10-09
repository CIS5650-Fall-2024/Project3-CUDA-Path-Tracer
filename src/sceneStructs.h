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
    TRIANGLE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct TriangleData
{
    glm::vec3 verts[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
};

struct Geom
{
    enum GeomType type;
    int materialid;
    TriangleData triData;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct DiffuseMap
{
    int index = -1;
    int width, height, channel;
    int startIdx;
};

struct Material
{
    glm::vec3 color;
    DiffuseMap diffuseMap;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    struct
    {
        bool isMicrofacet = false;
        float roughness;
    } microfacet;
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
    float focalLength;
    float apertureRadius;
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
    bool hitLight;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  int materialId;
};

struct BVHNode
{
    glm::vec3 mins;
    glm::vec3 maxs;
    int leftChild;
    int rightChild;
    int numPrimitives;
    int startIndex;
    bool isLeaf() const
    {
        return numPrimitives > 0;
    }
    BVHNode() :mins(glm::vec3(1e30f)), maxs(glm::vec3(-1e30f)), leftChild(-1), rightChild(-1), numPrimitives(0), startIndex(-1) {};
};
