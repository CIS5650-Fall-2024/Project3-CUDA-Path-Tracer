#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "bbox.h"
#include "flags.h"

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
};

enum MatType
{
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE,
    NOMAT
};

enum TextureType
{
    CONSTANT,
    CHECKER,
    IMAGE
};

struct ImageTextureInfo
{
    int index;
    int width;
    int height;
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
    ImageTextureInfo bumpmapTextureInfo;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uv[3];
    int numVertices;

    BBox bbox() {
        BBox bbox;
        for (int i = 0; i < numVertices; i++) {
            bbox.enclose(vertices[i]);
        }
        bbox.transform(transform);
        return bbox;
    }
};

struct Material
{
    enum MatType type;
    enum TextureType texType;
    glm::vec3 color;
    float checkerScale;
    ImageTextureInfo imageTextureInfo;
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
    float focalLength;
    float apertureSize;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    unsigned int sampleWidth;
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
  glm::vec2 texCoord;
  int materialId; // materialId == -1 means no intersection
};
