#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    GLTF_MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

struct AABB
{
    glm::vec3 min{ glm::vec3(FLT_MAX) };
    glm::vec3 max{ glm::vec3(FLT_MIN) };

    static AABB combine(const AABB& bounds1, const AABB& bounds2)
    {
        AABB unionBounds;
        unionBounds.min = glm::min(bounds1.min, bounds2.min);
        unionBounds.max = glm::max(bounds1.max, bounds2.max);
        return unionBounds;
    }

    /// <summary>
    /// Adds this point to the bounds
    /// </summary>
    /// <param name="p"></param>
    void include(glm::vec3 p)
    {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    glm::vec3 getDiagonal() const
    {
        return max - min;
    }

    int getLongestSplitAxis()
    {
        glm::vec3 d = getDiagonal();
        if (d.x > d.y && d.x > d.z)
        {
            return X_AXIS;
        }
        else if (d.y > d.z)
        {
            return Y_AXIS;
        }
        else
        {
            return Z_AXIS;
        }
    }
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor;
};

struct Triangle
{
    Vertex v0, v1, v2;
    bool hasNormals{ false };   // when normals are not defined we approximate them on cuda side
    AABB aabb;
    glm::vec3 centroid;
    union {
        int a;
        int b;
    };

    void computeAabbAndCentroid()
    {
        aabb.min = glm::min(glm::min(glm::min(aabb.min, v2.pos), v1.pos), v0.pos);
        aabb.max = glm::max(glm::max(glm::max(aabb.max, v2.pos), v1.pos), v0.pos);
        centroid = (v0.pos + v1.pos + v2.pos) * 0.333333f;
    }
};

struct SceneMesh
{
    const unsigned short* indices;
    bool hasIndices{ false };
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    bool hasNormals{ false };

    int startTriIdx{ -1 };
    int endTriIdx{ -1 };

    AABB aabb;

    SceneMesh() :
        indices(), hasIndices(false), positions(std::vector<glm::vec3>()), normals(std::vector<glm::vec3>())
    {}
};

/// <summary>
/// Used for the case where there's multiple meshes in a GLTF scene
/// </summary>
struct SceneMeshGroup
{
    bool valid{ false };
    int startTriIdx{ -1 };
    int endTriIdx{ -1 };
    int startMeshIdx{ -1 };
    int endMeshIdx{ -1 };
    int startBvhNodeIdx{ -1 };
    AABB aabb;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int startTriIdx;
    int endTriIdx;
    int startBvhNodeIdx;

    AABB aabb;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;

    // Thin lens camera
    float apertureSize;
    float focalLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    glm::vec3 accum_throughput;     // throughput, we'll only use it if we eventually hit a light source
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct Intersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};