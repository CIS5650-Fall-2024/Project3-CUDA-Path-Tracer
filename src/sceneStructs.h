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
    int diffuse_textureId;
    int normal_textureId;
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
    float lens_radius = 0.3f;
    float focal_dis = 5.0f;
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
  glm::vec2 surfaceUV;
  glm::vec3 tangent;
};

struct isActive {
    __host__ __device__
        bool operator()(const PathSegment& segment) {
        return segment.remainingBounces > 0;
    }
};

struct compareMatID {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};

struct Vertex {
    int materialid;
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
    glm::vec3 tangent;
};

struct Texture {
    int width;
    int height;
    int startIdx;
};