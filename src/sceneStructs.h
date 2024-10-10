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

enum MatType {
    DIFFUSE, 
    MIRROR, 
    DIELECTRIC, 
    MICROFACET,
    TEXTURE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 points[3];
    glm::vec3 planeNormal;
    glm::vec3 normals[3];
    glm::vec2 uvs[3];

    Triangle()
        : points{ glm::vec3(), glm::vec3(), glm::vec3() },
        planeNormal(glm::vec3()),
        normals{ glm::vec3(), glm::vec3(), glm::vec3() },
        uvs{ glm::vec2(), glm::vec2(), glm::vec2() } {}

    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
        : points{ p1, p2, p3 },
        planeNormal(glm::normalize(glm::cross(p2 - p1, p3 - p2))),
        normals{ planeNormal, planeNormal, planeNormal },
        uvs{ glm::vec2(), glm::vec2(), glm::vec2() } {}

    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3) 
        : points{ p1, p2, p3 },
        planeNormal(glm::normalize(glm::cross(p2 - p1, p3 - p2))),
        normals{ n1, n2, n3 },
        uvs{ glm::vec2(), glm::vec2(), glm::vec2() } {}
};


/****** For BVH ******/
struct BoundingBox {
    glm::vec3 Min;
    glm::vec3 Max;
    bool hasPoint = false;

    BoundingBox() : Min(glm::vec3()), Max(glm::vec3()) {};

    // Calculate Centre (similar to the C# property)
    glm::vec3 Centre() const {
        return (Min + Max) / 2.0f;
    }

    // Calculate Size (similar to the C# property)
    glm::vec3 Size() const {
        return Max - Min;
    }

    // Grow the bounding box to include a new point defined by min and max
    void resize(const glm::vec3& min, const glm::vec3& max) {
        if (hasPoint) {
            Min.x = std::min(min.x, Min.x);
            Min.y = std::min(min.y, Min.y);
            Min.z = std::min(min.z, Min.z);
            Max.x = std::max(max.x, Max.x);
            Max.y = std::max(max.y, Max.y);
            Max.z = std::max(max.z, Max.z);
        }
        else {
            hasPoint = true;
            Min = min;
            Max = max;
        }
    }
};

struct BVHTriangle {
    glm::vec3 center;
    glm::vec3 minCoors;
    glm::vec3 maxCoors;
    int index;

    BVHTriangle()
        : center(glm::vec3()), minCoors(glm::vec3()), maxCoors(glm::vec3()), index(0) {}

    // Constructor
    BVHTriangle(const glm::vec3& centre, const glm::vec3& min, const glm::vec3& max, int index)
        : center(centre), minCoors(min), maxCoors(max), index(index) {}
};

// Assuming Node struct exists
struct BVHNode {
    glm::vec3 minCoors;
    glm::vec3 maxCoors;
    int startIdx;
    int numOfTriangles;
    
    BVHNode()
        : minCoors(glm::vec3()), maxCoors(glm::vec3()), startIdx(0), numOfTriangles(0) {}

    BVHNode(const BoundingBox& bounds)
        : minCoors(bounds.Min), maxCoors(bounds.Max), startIdx(-1), numOfTriangles(-1) {}

    // Constructor with BoundingBox and triangle/child data
    BVHNode(const BoundingBox& bounds, int startIndex, int triCount)
        : minCoors(bounds.Min), maxCoors(bounds.Max), startIdx(startIndex), numOfTriangles(triCount) {}
};

struct SplitResult {
    int axis;
    float pos;
    float cost;

    SplitResult(int axis_, float pos_, float cost_) : axis(axis_), pos(pos_), cost(cost_) {}
};

struct TriangleHitInfo {
    bool didHit;
    float dst;
    float3 hitPoint;
    float3 normal;
    int triIndex;
};
/*****************************************************************************************************************************/

struct Geom
{
    enum GeomType type;
    // int materialid;
    struct {
        int materialid;
        int albedoTextureID;
        int normalTextureID;
        int bumpTextureID;
    } material;
    int numTriangles = 0;
    int numBvhNodes = 0;

    Triangle* triangles; // Host-side pointer
    Triangle* devTriangles; // Device-side pointer
    // Triangle* bvhTriangles; // Host-side pointer
    // Triangle* devBvhTriangles; // Device-side pointer
    // BVHNode* bvhNodes; // Host-side pointer
    // BVHNode* devBvhNodes; // Device-side pointer

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    int type;
    glm::vec3 color;
    glm::vec3 specularColor;
    float roughness;
    float emittance;
    float indexOfRefraction;
};

/****** For Texture Loading ******/
struct Texture{
    glm::ivec2 size;
    glm::vec4* dev_data;
};

struct TextureValues {
    glm::vec4 albedo;
    glm::vec4 normal;
    glm::vec4 bump;
};
/*****************************************************************************************************************************/

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
    float lensRadius;
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
    bool hasHitLight;
    float eta; // Used for Ruassin roulette to determine how likely this ray survives
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
    struct 
    {
        int materialId;
        int albedoTextureID;
        int normalTextureID;
        int bumpTextureID;
    } materials;
};
