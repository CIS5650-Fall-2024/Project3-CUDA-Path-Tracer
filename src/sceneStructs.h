#pragma once

#include <string>
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

enum TextureType
{
    TEXTURE_2D,
    CUBE_MAP,
    PROCEDURAL,
    SKYBOX
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle
{
    glm::vec3 points[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3]; 
	glm::vec3 planeNormal;
    int index_in_mesh;
	int materialid;

	Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int idx)
	{
		points[0] = p1;
		points[1] = p2;
		points[2] = p3;
        index_in_mesh = idx;

	}
	Triangle()
	{
		points[0] = glm::vec3(0.0f);
		points[1] = glm::vec3(0.0f);
		points[2] = glm::vec3(0.0f);
		index_in_mesh = -1;
	}
};

struct Mesh
{
	glm::vec3* vertices;
    glm::vec3* normals;
    glm::vec2* uvs;
	int* indices;
	int num_vertices;
	int num_normals;
	int num_uvs;
	int num_indices;
	int num_triangles;
	Triangle* triangles;

	Mesh()
	{
		vertices = nullptr;
		normals = nullptr;
		uvs = nullptr;
		indices = nullptr;
		triangles = nullptr;
		num_vertices = 0;
		num_normals = 0;
		num_uvs = 0;
		num_indices = 0;
        num_triangles = 0;

	}
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
	Mesh* mesh;
};

struct Texture
{
    glm::vec3* data;
    int width;
    int height;
    enum TextureType type;

    Texture(int w, int h) : width(w), height(h)
    {
        cudaMalloc(&data, width * height * sizeof(glm::vec3));
    }

	Texture() : width(0), height(0), data(nullptr), type(TEXTURE_2D) {}

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

    // textures
	int hasDiffuseTexture;
	int hasNormalTexture;
    Texture* diffuseTexture = nullptr;
    Texture* normalTexture = nullptr;

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

    float lensRadius = 0.08;
	float focalDistance = 10;
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
  //glm::mat3* TBN;
  glm::vec3 bitangent;
  glm::vec3 tangent;
};
