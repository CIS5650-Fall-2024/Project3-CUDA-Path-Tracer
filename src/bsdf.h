#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <thrust/random.h>
#include "sceneStructs.h"
#include "utilities.h"

// Math helper
__device__ float cosThetaUnit(const glm::vec3& w);

__device__ float absCosThetaUnit(const glm::vec3& w);

// Build local coordinate space
__device__ void makeCoordTrans(glm::mat3& o2w, const glm::vec3& n);

// Reflect helper
__device__ glm::vec3 reflect(const glm::vec3& wo);

// Refract helper
__device__ bool refract(const glm::vec3& wo, glm::vec3& wi, float eta);

__device__ float schlickFresnel(float cosThetaI, float etaI, float etaT);

// Sampler helper
__device__ glm::vec3 cosineWeightedHemisphereSampler3D(float* pdf, thrust::default_random_engine& rng);

__device__ glm::vec3 phoneSampleHemisphere(float shininess, float* pdf, thrust::default_random_engine& rng);

// Universal BSDF class
class BSDF {
public:
	// Universal BSDF constructor
	BSDF(enum MaterialType type, 
		 glm::vec3 albeto, glm::vec3 specularColor, glm::vec3 transmittanceColor, glm::vec3 emissiveColor,
		 float kd, float ks, float shininess, float ior, bool delta);

	// BSDF evaluation
	__device__ glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi);

	// BSDF sampling and evaluation
	__device__ glm::vec3 sampleF(const glm::vec3& wo, 
								 glm::vec3& wi, float* pdf, 
								 thrust::default_random_engine& rng);

	// Properties
	__device__ MaterialType getType() const;

	__device__ glm::vec3 getEmission() const;

	__device__ bool isDelta() const;

private:
	// Material Type
	MaterialType type;

	// Surface diffuse albedo
	glm::vec3 albeto;

	// Surface specular color
	glm::vec3 specularColor;

	// Surface transmittance color
	glm::vec3 transmittanceColor;

	// Surface emissive color
	glm::vec3 emissiveColor;

	// Diffuse coefficient
	float kd;

	// Specular coefficient
	float ks;

	// Shininess exponent for specular reflection
	float shininess;

	// Index of Refraction
	float ior;

	// Whether it's delta bsdf
	bool delta;
};
