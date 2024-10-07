#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

struct AABB {
    glm::vec3 AABBmin;
    glm::vec3 AABBmax;
};

struct BVHNode {
    
    AABB bound;
    int left;
    int right;
    int start;
    int end;
    bool isLeaf;
};

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
struct Triangle {
    glm::vec3 v0, v1, v2;
    glm::vec3 normal;

    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;
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

    BVHNode* bvhNodes;
    int numBVHNodes;
    Triangle* triangles;
    int numTriangles;
};

struct TextureData {
    int width;
    int height;
    int channels;
    unsigned char* h_data; // Host data
};

struct EnvData_hdr {
    int width;
    int height;
    int channels;
    float* h_data; // Host data
     
};


struct Texture {
    cudaTextureObject_t texObj;
    cudaArray_t cuArray;
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

    //Host Side
    TextureData albedoMapData;
    TextureData normalMapData;
    EnvData_hdr envMapData;

    // Device-side CUDA texture objects
    Texture albedoMapTex;
    Texture normalMapTex;

    Texture envMap;
    bool isEnvironment;
    float env_intensity;
    

    

   
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
    float focalDistance;
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

  glm::vec2 uv;
};
