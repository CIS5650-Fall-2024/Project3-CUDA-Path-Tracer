#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

struct Mesh
{
    int material_id;
    glm::vec3 *vertices;
    int num_vertices;
    int *indices;
    int num_indices;
    Geom bounding_volume;

    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    void compute_bounding_box() {
        this->bounding_volume.type = CUBE;
        this->bounding_volume.materialid = 0;

        glm::vec3 min{std::numeric_limits<float>::infinity()};
        glm::vec3 max{-std::numeric_limits<float>::infinity()};

    for (auto i = 0; i < num_vertices; i++) {
        const auto &vertex = vertices[i];
        min = glm::min(min, vertex);
        max = glm::max(max, vertex);
    }

    this->bounding_volume.translation = (min + max) / 2.0f;
    this->bounding_volume.scale = max - min;

    this->bounding_volume.transform = this->transform * glm::translate(glm::mat4(), this->bounding_volume.translation) * glm::scale(glm::mat4(1.0f), this->bounding_volume.scale);

    this->bounding_volume.inverseTransform = glm::inverse(this->bounding_volume.transform);
    this->bounding_volume.invTranspose = glm::inverseTranspose(this->bounding_volume.transform);

    }
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

    struct {
        float translucency;
        float absorption;
        float thickness;
    } subsurface;
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

    float lens_radius = 0.2f;
    float focal_distance = 4.0f;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

    bool anti_aliasing = false;
    bool sort_by_material = false;
    bool depth_of_field = false;
    bool stream_compaction = false;
    bool boundary_volume_culling = false;
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
};
