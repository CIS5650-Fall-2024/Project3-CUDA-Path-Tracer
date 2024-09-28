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

    // triangle attributes
    glm::vec3 trianglePos[3];  
    glm::vec3 triangleNor[3]; 
    glm::vec2 triangleTex[3]; 
};

struct Material
{
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    struct
    {
      int albedo; 
      int normal; 
    } textureIdx;

    glm::vec3 color;
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
    bool isFinished; 
    bool isTerminated; 
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  
  // for textures
  glm::vec2 texSample; 

  // for normal maps
  glm::vec3 tangent; 
  glm::vec3 bitangent;

  int materialId;

  __host__ __device__ bool operator<(const ShadeableIntersection& other) const
  {
    return materialId < other.materialId;
  }
};
