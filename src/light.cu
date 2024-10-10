#include "light.h"

// Universal Light constructor
Light::Light(enum LightSourceType type, glm::vec3 color, 
	         glm::vec3 pos, glm::vec3 dir, glm::vec3 dimX, glm::vec3 dimY, 
	         float angle, bool delta)
	: type(type), radiance(color), pos(pos), 
	  dir(glm::normalize(dir)), dimX(dimX), dimY(dimY), area(glm::length(dimX) * glm::length(dimY)), 
	  angle(angle), delta(delta) {}

// Sample a shadow ray direction based on the light source type and compute radiance
__device__ glm::vec3 Light::sampleL(const glm::vec3& interP, 
									glm::vec3& wi, float* pdf, 
									thrust::default_random_engine& rng) const {
	// variables to use
	glm::vec3 d;
	float sqDist;
	float dist;
	
	switch (type) {

	case AREALIGHT:
		thrust::uniform_real_distribution<float> s(-0.5, 0.5);
		// Sample point on the surface area
		d = pos + s(rng) * dimX + s(rng) * dimY - interP;
		sqDist = glm::dot(d, d);
		dist = sqrt(sqDist);
		float cosTheta = glm::dot(-d / dist, dir);
		// Update shadow ray
		wi = d / dist;
		*pdf = sqDist / (area * cosTheta);
		return cosTheta > 0 ? radiance * cosTheta * (1.0f / sqDist) : glm::vec3(0.0f);

	case DIRECTIONALLIGHT:
		// Directional light's direction and intensity is consistent and light source is considered located at infinity
		wi = -dir;
		*pdf = 1.0f;
		return radiance;

	case POINTLIGHT:
		d = pos - interP;
		sqDist = glm::dot(d, d);
		dist = sqrt(sqDist);
		// Update shadow ray
		wi = d / dist;
		*pdf = 1.0f;
		return radiance * (1.0f / sqDist);

	case SPOTLIGHT:
		d = pos - interP;
		sqDist = glm::dot(d, d);
		dist = sqrt(sqDist);
		float intersectAngle = acos(glm::dot(dir, -d / dist));
		// Update shadow ray
		wi = d / dist;
		*pdf = 1.0f;
		return intersectAngle < angle ? radiance * (1.0f / sqDist) : glm::vec3(0.0f);

	default:
		wi = -dir;
		*pdf = 1.0f;
		return radiance;
	}
}

// Check if delta light, which only needs one sample
__device__ bool Light::isDelta() const {
	return delta;
}