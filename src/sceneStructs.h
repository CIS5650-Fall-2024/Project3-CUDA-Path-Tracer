#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum MatType
{
    LIGHT = 0,
    DIFFUSE_REFL = 1,
    SPEC_REFL = 2,
    SPEC_TRANS = 3,
    SPEC_GLASS = 4,
    MICROFACET_REFL = 5,
    DIAMOND = 6,
    CERAMIC = 7,
    MATTEBLACK = 8
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

enum GeomType
{
    G_SPHERE = 0,
};

enum LightType
{
    AREALIGHT = 0,
    POINTLIGHT = 1,
};

enum ShapeType
{
    RECTANGLE = 0,
    SPHERE = 1,
};

struct AreaLight {
    glm::vec3 Le;
    float emittance;
    int ID;

    ShapeType shapeType;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Geom
{
    GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    glm::vec3 color;
    enum MatType type;
    float roughness;
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
    glm::vec3 L;
    glm::vec3 beta;
    int pixelIndex;
    int remainingBounces;
    bool lastHitWasSpecular;
};

// Functor for the removal condition
struct CheckRemainingBounces {
    __host__ __device__
        bool operator()(const PathSegment& p) {
        return p.remainingBounces > 0;
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray

struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 texCol;
  int materialId;
  int areaLightId;
};

struct getMatId {
    __host__ __device__
    int operator()(const ShadeableIntersection& s)
    {
        return s.materialId;
    }
};