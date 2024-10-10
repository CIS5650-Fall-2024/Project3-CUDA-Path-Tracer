#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <thrust/random.h>
#include "sceneStructs.h"
#include "utilities.h"

// Universal Light class
class Light {
public:
	// Universal Light constructor
	Light(enum LightSourceType type, glm::vec3 color, 
		  glm::vec3 pos, glm::vec3 dir, glm::vec3 dimX, glm::vec3 dimY, 
		  float angle, bool delta);

	// Light sample shadow ray
	__device__ glm::vec3 sampleL(const glm::vec3& interP, 
								 glm::vec3& wi, float* pdf, 
								 thrust::default_random_engine& rng) const;

	// Check if delta light
	__device__ bool isDelta() const;

private:
	// Light source type
	LightSourceType type;

	// Light source radiance
	glm::vec3 radiance;

	// Light source position (if any)
	glm::vec3 pos;

	// Light source facing direction
	glm::vec3 dir;

	// Light source X dimension (if any)
	glm::vec3 dimX;

	// Light source Y dimension (if any)
	glm::vec3 dimY;

	// Light source area (if any)
	float area;

	// Light source angle (if any)
	float angle;

	// Light source if delta light
	bool delta;
};