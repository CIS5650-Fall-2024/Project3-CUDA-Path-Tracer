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

struct Triangle
{
    glm::vec3 verts[3];
    glm::vec2 uvs[3];
    glm::vec3 normals[3];
    //tangent
    glm::vec3 tangent;
    //bitangent
    glm::vec3 bitangent;
    glm::vec3 centroid;
    //transfromed vertices
    glm::vec3 transVerts[3];
};

struct Texture
{
	unsigned char* data;
	int width;
	int height;
	int channels;
};

//BVH
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct BVHNode
{
	AABB aabb;
	int left = -1;
	int right = -1;
    // For leaf nodes
    int triIndexStart;
    int triIndexEnd;
    bool isLeaf = false;
   // bool isLeaf() { return (triIndexEnd - triIndexStart) <= 1; }
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
    int triIndexStart;
    int triIndexEnd;
    //Add for texture
    int textureid = -1 ;
    int hasTexture = 0 ;
    //Add for normal
    int normalid = -1;
    int hasNormal = 0;

    int rootNodeIdx = -1;
    AABB aabb;
#if 0
    glm::vec3 getCentroid(const std::vector<Triangle>& triangles) const
    {
        if (type == SPHERE || type == CUBE) {
            return translation; // For a sphere or cube, centroid is the center (translation)
        }
        else if (type == MESH) {
            glm::vec3 centroid(0.0f);
            int count = 0;
            // For a mesh, calculate centroid from all its triangles
            for (int i = triIndexStart; i < triIndexEnd; ++i) {
                const Triangle& tri = triangles[i];
                centroid += (tri.verts[0] + tri.verts[1] + tri.verts[2]) / 3.0f;
                count++;
            }
            if (count > 0) {
                centroid /= count;
            }
            return centroid;
        }
        return glm::vec3(0.0f); // Default value if none of the conditions are met
    }
#endif
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

    //Add for texture

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
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool outside;
  int textureid = -1;
  glm::vec2 uv;
  int normalid = -1;
  //tangent
  glm::vec3 tangent;
  //bitangent
  glm::vec3 bitangent;
};
