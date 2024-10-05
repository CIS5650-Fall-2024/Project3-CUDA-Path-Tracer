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

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 norm;
	glm::vec2 uv;
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
	glm::vec3 mesh_aabb_max;
	glm::vec3 mesh_aabb_min;
	int vertex_offset;
	int vertex_count;
	int texture_offset;
	int texture_count;
	int norm_texture_offset;
	int norm_texture_count;
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
	float lensRadius;
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
	int textureOffset;
	int textureCount;
	int normTextureOffset;
	int normTextureCount;
};

struct Texture
{
	int height;
	int width;
	glm::vec3 color;
};
