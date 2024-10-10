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
    CUSTOM_MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 v0, v1, v2; // Vertices
    glm::vec3 n0, n1, n2; // Normals
    glm::vec2 uv0, uv1, uv2; // UV coordinates
};

struct Geom
{
    GeomType type;
    int materialid;

    // Transformation matrices
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // For custom meshes
    int triangleStartIndex;
    int triangleCount;

    // Bounding box in object space
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
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

    // Texture indices (-1 if not used)
    int baseColorTextureIndex;
    int normalTextureIndex;

    bool isProcedural;
    glm::vec3 proceduralColor1;
    glm::vec3 proceduralColor2;
    float proceduralScale;
};

struct Texture {
    int width;
    int height;
    int components; // Number of color channels (e.g., 3 for RGB, 4 for RGBA)
    std::vector<unsigned char> imageData; // Image pixel data
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


    float lensRadius;    // Aperture size
    float focalDistance; // Focal length
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

struct ShadeableIntersection {
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
    int triangleIndex;         // Index of the intersected triangle (-1 if not a triangle)
    glm::vec3 barycentricCoords; // Barycentric coordinates at the intersection point
    GeomType hitGeomType;      // Type of geometry intersected
    int geomIndex;             // Index of the geometry that was hit
};
