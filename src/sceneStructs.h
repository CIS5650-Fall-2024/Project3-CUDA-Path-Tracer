#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
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
    float roughness;
};


struct Light
{
    int geom_id;
    float intensity;
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
    glm::vec3 defocus_disk_up;       // Defocus disk vertical radius
    glm::vec3 defocus_disk_right;       // Defocus disk horizontal radius
    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 3.4;    // Distance from camera lookfrom point to plane of perfect focus
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
    glm::vec3 directColor;
    glm::vec3 indirectColor;
    int pixelIndex;
    int remainingBounces;
    glm::vec3 throughput;
    bool endPath;
    int path_index;
    float IOR;
    

    
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 intersect_point;
  int materialId;
};
