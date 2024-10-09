#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
static constexpr int MAX_BVH_DEPTH = 16;
static constexpr int MAX_TRIANGLES_PER_LEAF = 4;

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
    int attributeIndex[3];
    glm::vec3 center;
};

struct Mesh {
	int vertStartIndex;
    int trianglesStartIndex;
	int numTriangles = 0;
	int baseColorUvIndex;
	int normalUvIndex;
	int emissiveUvIndex;
	int bvhRootIndex;
    glm::vec3 boundingBoxMin;
    glm::vec3 boundingBoxMax;
};

struct Geom
{
    enum GeomType type;
    int materialid = -1;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int meshId;
};

struct Material
{
    glm::vec3 color;
    glm::vec3 emissiveFactor;
    float metallic;
	float roughness;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
	int baseColorTextureId = -1;
	int normalTextureId = -1;
	int emissiveTextureId = -1;
};

struct Texture {
	int width;
	int height;
	int numComponents;
    int size;
	std::vector<glm::vec4> data; // we can use a vector here, because this data structure doesn't go to the GPU.
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
	glm::vec3 color = glm::vec3(1.0f);
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
  glm::vec3 surfaceTangent;
  glm::vec3 surfaceBitangent;
  int materialId;
  glm::vec2 baseColorUvs;
  glm::vec2 normalUvs;
  glm::vec2 emissiveUvs;
};

struct BvhNode {
	glm::vec3 min = glm::vec3(FLT_MAX);
	glm::vec3 max = glm::vec3(-FLT_MAX);
	int leftChild = -1;
	int rightChild = -1;
	int trianglesStartIdx = 0;
	int numTriangles = 0;
};