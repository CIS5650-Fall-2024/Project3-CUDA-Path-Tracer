#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "gltfLoader.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

enum MatType
{
    DIFFUSE,
    LIGHT,
    SPECULAR,
    DIELECTRIC,
    GGX,
    SKIN,
    INVALID
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

template <typename T>
__host__ __device__ void cuda_swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

struct AABB {
  glm::vec3 min;
  glm::vec3 max;

  AABB() : min(glm::vec3(FLT_MAX)), max(glm::vec3(-FLT_MAX)) {}

  __host__ __device__ bool intersect(const Ray& ray, float& tMin, float& tMax) const {
    for (int i = 0; i < 3; ++i) {
      float invD = 1.0f / ray.direction[i];
      float t0 = (min[i] - ray.origin[i]) * invD;
      float t1 = (max[i] - ray.origin[i]) * invD;
      if (invD < 0.0f) cuda_swap(t0, t1);
      tMin = t0 > tMin ? t0 : tMin;
      tMax = t1 < tMax ? t1 : tMax;
      if (tMax <= tMin) return false;
    }
    return true;
  }
};

struct BVHNode {
  AABB bounds;
  int left;   // Index of left child (internal node)
  int right;  // Index of right child (internal node)
  int start, end; // Range of triangle indices (leaf node)
  bool isLeaf;    // Whether this node is a leaf
};

struct Geom
{
    enum GeomType type;
    int materialid;
    MatType matType;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int offset;
    int count;
    int indexOffset;
    int indexCount;
    BVHNode* meshBVH;         // Pointer to the BVH nodes for the triangles in this mesh
    int meshBVHCount;
    int* triangleIndices;     // Pointer to triangle indices for the mesh BVH
    int bvhRoot;              // Root node index in meshBVH
    int triangleCount;        // Number of triangles in the mesh
    int *triangleIndicesGPU;
    int *meshBVHGPU;
};

struct Material
{
    MatType type;
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
    float subsurfaceScattering;
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
    float aperture;
    float focusDistance;
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
    int materialType;
    int materialId;
    int intersectionIdx;
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