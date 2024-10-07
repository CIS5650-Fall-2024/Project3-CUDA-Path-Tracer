#pragma once
#include "utilities.h"
#include "sceneStructs.h"

extern Triangle* dev_triangles;
extern Geom* dev_geoms;
extern int dev_numGeoms;
extern Material* dev_materials;
extern int* dev_triTransforms;
extern Light* dev_lights;

void initSceneCuda(Geom* geoms, Material* materials, Triangle* triangles, Light* lights, int numGeoms, int numMaterials, int numTriangles, int numLights);
//__global__ void updateTriangleTransformIndex(Geom* dev_geoms, int* dev_triTransforms, int numGeoms);
//__global__ void updateTriangleTransform(Geom* dev_geoms, Triangle* dev_triangles, int* dev_triTransforms, int numGeoms, int numTriangles);

//void updateTrianglesTransform(Geom* geom, Triangle* triangles, int numGeoms, int numTriangles);
void freeSceneCuda();

void printGeoms();

void checkCUDAError(const char* msg);

__inline__ __device__ glm::vec3 slerp(glm::vec3 a, glm::vec3 b, float t)
{
	float dotProduct = glm::dot(a, b);
	dotProduct = glm::clamp(dotProduct, -1.0f, 1.0f);
	float theta = acosf(dotProduct) * t;
	glm::vec3 relativeVec = b - a * dotProduct;
	relativeVec = glm::normalize(relativeVec);
	return a * cosf(theta) + relativeVec * sinf(theta);
}