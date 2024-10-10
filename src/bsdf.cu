#include "BSDF.h"

// Math helper
__device__ float cosThetaUnit(const glm::vec3& w) {
	return w.z;
}

__device__ float absCosThetaUnit(const glm::vec3& w) {
	return fabs(w.z);
}

// Build local coordinate system with intersection surface normal as local z-axis
__device__ void makeCoordTrans(glm::mat3& o2w, const glm::vec3& n) {
	// normal as local z
	glm::vec3 z = glm::vec3(n.x, n.y, n.z);
	glm::vec3 h = z;

	// h as not parallel to z
	if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) h.x = 1.0f;
	else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) h.y = 1.0f;
	else h.z = 1.0f;

	// construct x-axis and y-axis
	z = glm::normalize(z);
	glm::vec3 y = glm::normalize(glm::cross(h, z));
	glm::vec3 x = glm::normalize(glm::cross(z, y));

	// fill in o2w
	o2w[0] = x;
	o2w[1] = y;
	o2w[2] = z;
}

// Get reflection wi with wo in local coordinate system
__device__ glm::vec3 reflect(const glm::vec3& wo) {
	return glm::vec3(-wo.x, -wo.y, wo.z);
}

__device__ float schlickFresnel(float cosThetaI, float etaI, float etaT) {
	float r0 = powf((etaI - etaT), 2) / powf((etaI + etaT), 2);
	return r0 + (1.0f - r0) * powf((1.0f - cosThetaI), 5.0f);
}

// Get refract wi with wi in local coordinate system using Snell's Law, check whether it's total internal reflection
__device__ bool refract(const glm::vec3& wo, glm::vec3& wi, float ior) {
	float eta, check;
	if (wo.z > 0) 
	{
		eta = 1.0f / ior;
		check = 1 - eta * eta * (1.0 - wo.z * wo.z);
		if (check < 0) 
		{
			return false;
		}
		wi.x = -eta * wo.x;
		wi.y = -eta * wo.y;
		wi.z = -sqrtf(check);
		return true;
	}
	else 
	{
		eta = ior;
		check = 1 - eta * eta * (1.0 - wo.z * wo.z);
		if (check < 0) 
		{
			return false;
		}
		else
		{
			wi.x = -eta * wo.x;
			wi.y = -eta * wo.y;
			wi.z = sqrtf(check);
			return true;
		}
	}
}

// Sample a cosine-weighted random unit direction in a hemisphere in local coordinate system
__device__ glm::vec3 cosineWeightedHemisphereSampler3D(float* pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float xi1 = u01(rng);
	float xi2 = u01(rng);

	float r = sqrt(xi1);
	float theta = 2.0f * PI * xi2;
	*pdf = sqrt(1 - xi1) * ONE_OVER_PI;
	return glm::vec3(r * cos(theta), r * sin(theta), sqrt(1 - xi1));
}

// Sample a halfway unit vector in a hemisphere
__device__ glm::vec3 phoneSampleHemisphere(float shininess, float* pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float xi1 = u01(rng);
	float xi2 = u01(rng);

	float cosTheta = powf(xi1, 1.0f / (shininess + 1.0f));
	float sinTheta = sqrt(fmax(0.0f, 1.0f - cosTheta * cosTheta));
	float phi = 2.0f * PI * xi2;
	*pdf = (shininess + 2.0f) * ONE_OVER_TWO_PI * powf(glm::max(cosTheta, 0.0f), shininess);
	return glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// Universal BSDF constructor
BSDF::BSDF(enum MaterialType type,
		   glm::vec3 albeto, glm::vec3 specularColor, glm::vec3 transmittanceColor, glm::vec3 emissiveColor,
		   float kd, float ks, float shininess, float ior, bool delta)
	: type(type), 
	  albeto(albeto), specularColor(specularColor), transmittanceColor(transmittanceColor), emissiveColor(emissiveColor),
	  kd(kd), ks(ks), shininess(shininess), ior(ior), delta(delta){}

// BSDF evaluation based on material type using wi and wo
__device__ glm::vec3 BSDF::f(const glm::vec3& wo, const glm::vec3& wi) {
	switch (type) {
	case MIRROR:
		// Perfect reflection
		return glm::vec3(0.0f);

	case GLASS:
		// reflect-refract using Schlick's approximation
		return glm::vec3(0.0f);

	case GLOSSY:
		// Blinn-Phone
		glm::vec3 diffuse = kd * albeto * ONE_OVER_PI;
		glm::vec3 h = glm::normalize(wi + wo);
		float specAngle = glm::max(h.z, 0.0f);
		glm::vec3 specular = ks * specularColor * (shininess + 2.0f) * ONE_OVER_TWO_PI * powf(specAngle, shininess);
		return diffuse + specular;

	case EMISSION:
		// Light
		return glm::vec3(0.0f);

	default:
		// Default Diffuse
		return albeto * ONE_OVER_PI;
	}
}

// BSDF sampling for wi using wo in local coordinate system and evaulation based on material type using wi and wo
__device__ glm::vec3 BSDF::sampleF(const glm::vec3& wo, 
								   glm::vec3& wi, float* pdf, 
								   thrust::default_random_engine& rng) {
	// Diffuse by default
	switch (type) {
	case MIRROR:
		// Perfect reflection
		wi = reflect(wo);
		*pdf = 1;
		return specularColor / absCosThetaUnit(wi);

	case GLASS:
		if (!refract(wo, wi, ior)) 
		{
			wi = reflect(wo);
			*pdf = 1.f;
			return specularColor / absCosThetaUnit(wi);
		}
		else
		{
			float r = schlickFresnel(abs(wo.z), 1.0f, ior);
			thrust::uniform_real_distribution<float> u011(0, 1);
			if (u011(rng) < r) 
			{
				wi = reflect(wo);
				*pdf = r;
				return r * specularColor / absCosThetaUnit(wi);
			}
			else
			{
				*pdf = 1 - r;
				float eta;
				if (wo.z > 0) 
				{
					eta = 1.0f / ior;
				}
				else 
				{
					eta = ior;
				}
				return (1 - r) * transmittanceColor / absCosThetaUnit(wi) / powf(eta, 2);
			}
		}

	case GLOSSY:
		// Blinn-Phone
		thrust::uniform_real_distribution<float> u01(0, 1);
		float probSpecular = ks / (kd + ks);
		float dice = u01(rng);

		if (dice > probSpecular) {
			wi = cosineWeightedHemisphereSampler3D(pdf, rng);
			return f(wo, wi);
		}
		else 
		{
			glm::vec3 h = phoneSampleHemisphere(shininess, pdf, rng);
			wi = 2.0f * glm::dot(h, wo) * h - wo;
			return f(wo, wi);
		}

	case EMISSION:
		// Light
		wi = cosineWeightedHemisphereSampler3D(pdf, rng);
		return glm::vec3(0.0f);

	default:
		// Default Diffuse
		wi = cosineWeightedHemisphereSampler3D(pdf, rng);
		return albeto * ONE_OVER_PI;
	}
}

// Get material type
__device__ MaterialType BSDF::getType() const {
	return type;
}

// Get emission
__device__ glm::vec3 BSDF::getEmission() const {
	return emissiveColor;
}

// Check if delta bsdf
__device__ bool BSDF::isDelta() const
{
	return delta;
}