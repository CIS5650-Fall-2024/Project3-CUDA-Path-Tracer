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

struct AABB {
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
    glm::vec3 centroid = glm::vec3(0.f);
};

struct BVHNode {
    AABB aabb;
    int left = -1;
    int right = -1;
    int meshidx = -1;
};

// Triangle mesh
struct MeshTriangle {
    int v[3];
    int vn[3];
    int vt[3];
    int materialid = -1;
    AABB aabb;
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
    //For BVH
    int bvhrootidx;
    int meshidx;
    int meshcnt;
    // For motion blur
    glm::vec3 velocity;
};

struct Texture {
    int width;
    int height;
    int channels;
    int texturePathIndex;
    size_t dataSize;
};

enum class ShadingType {
    Specular,
    Diffuse,
    Refract,
    TexturePBR,
    Emitting,
    SubsurfaceScatter,
    Unknown
};

struct Material
{
    ShadingType shadingType{ ShadingType::Unknown };

    glm::vec3 color{ 0.0f, 0.0f, 0.0f };
    float emittance{ -1.0f };
    float specularRoughness{ -1.0f };
    float indexOfRefraction{ -1.0f };
    
    // Texture IDs
    int baseColorTextureId{ -1 };
    int roughnessMetallicTextureId{ -1 };
    int normalTextureId{ -1 };
    int emissiveTextureId{ -1 };
    int procedualTextureID{ -1 };

    // Sub-surface scattering
    glm::vec3 sigma_a = glm::vec3(0.f);
    glm::vec3 sigma_s = glm::vec3(0.f);
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
    float lensRadius = -1;
    float focalDistance = -1;
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
  float metallic;
  float roughness;
  int materialId;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  bool outside;
};
