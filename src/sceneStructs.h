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
    MESH
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

    // mesh
    int meshStart;
    int meshCount;
    int bvhRootIndex = -1;
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
    float roughness;
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

    // depth of field
    float focalDistance;
    float aperture;
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
    bool insideObject;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
    glm::vec2 uv;
    bool outside;
    glm::vec3 intersectionPoint;
};

struct AABB {
    glm::vec3 minBounds = glm::vec3(FLT_MAX);
    glm::vec3 maxBounds = glm::vec3(-FLT_MAX);
    glm::vec3 centroid = glm::vec3(0.f);
};

struct Triangle {
    int v[3];
    int n[3];
    int uv[3];
    int materialId;

    AABB aabb;
};

struct BVHNode {
    AABB aabb;
    int startIndex = -1;
    int leftChildIndex = -1;
    int rightChildIndex = -1;
    int primitiveCount = -1;
    int axis = -1;
};

struct LinearBVHNode {
    AABB aabb;
    union {
        int primitivesOffset;
        int secondChildOffset;
    };
    int nPrimitives;
    int axis;
};