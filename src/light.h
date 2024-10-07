#pragma once
#include "utilities.h"
#include <thrust/random.h>
#include "sceneStructs.h"
#include "bvh.h"


__inline__ __device__ glm::vec3 DirectSampleAreaLight(
	int idx,
	const glm::vec3& view_point,
	const glm::vec3& view_nor,
	int num_lights,
	glm::vec3& wiW,
	float& pdf,
	thrust::default_random_engine& rng,
	LinearBVHNode* dev_nodes,
	Triangle* dev_triangles,
	const Light& light)
{
	Ray shadowRay;

	// sample a point on the light
	glm::vec3 xi = glm::vec3(thrust::uniform_real_distribution<float>(-1.0f, 1.0f)(rng), thrust::uniform_real_distribution<float>(-1.0f, 1.0f)(rng), 0.0f);
	glm::vec4 vpl = light.inverseTransform * glm::vec4(view_point, 1.0f);
	glm::vec3 wi = xi - glm::vec3(vpl);

	// compute cosTheta and distance
	float cosTheta = AbsDot(glm::vec3(0, 0, 1), normalize(-wi));
	wiW = glm::vec3(light.transform * glm::vec4(wi, 0.));
	float r = length(wiW);

	// compute pdf da
	pdf = r * r / (cosTheta * light.area  + 0.001);
	wiW = normalize(wiW);

	// check if there is block
	ShadeableIntersection isect;
	Ray ray{ view_point, wiW };
	if (BVHIntersect(ray, dev_nodes, dev_triangles, &isect) && isect.lightId == idx);
	{
		//return glm::vec3(cosTheta);
		return (float)num_lights * light.emission;
	}
	return glm::vec3(0.0f, 0.f, 0.f);
}

__inline__ __device__ glm::vec3 getEnvironmentalRadiance(const glm::vec3& direction, cudaTextureObject_t envMap) {
	float theta = acosf(direction.y);         // θ
	float phi = atan2f(direction.z, direction.x); // φ
	if (phi < 0) phi += 2.0f * PI;

	float u = phi / (2.0f * PI);            // [0, 1]
	float v = theta / PI;                   // [0, 1]
	if (envMap == NULL) return glm::vec3(0.0f); // return black if no envMap (for debugging purposes
	float4 texel = tex2D<float4>(envMap, u, v);
	return glm::vec3(texel.x, texel.y, texel.z);
}

__inline__ __device__ glm::vec3 Sample_Li(
    const glm::vec3& view_point,
	const glm::vec3& nor,
    glm::vec3& wiW,
    float& pdf,
    int randomLightIdx,
    int N_LIGHTS,
    cudaTextureObject_t envMap,
    thrust::default_random_engine& rng,
	LinearBVHNode* dev_nodes,
	Triangle* dev_triangles,
	Light* dev_lights,
	const glm::mat3& ltw,
	const glm::mat3& wtl)
{
    // Choose a random light from among all of the
    // light sources in the scene, including the environment light
    int num_lights = N_LIGHTS;

    // choose a random light
	if (envMap != NULL && randomLightIdx == num_lights - 1)
	{
		// sample the environment map
		glm::vec3 wi = glm::vec3(thrust::uniform_real_distribution<float>(-1.0f, 1.0f)(rng), thrust::uniform_real_distribution<float>(-1.0f, 1.0f)(rng), 1.0);
		wiW = normalize(ltw * normalize(wi));
		pdf = 1.0f / (2.0f * PI);
		if (BVHIntersect(Ray{ view_point, wiW }, dev_nodes, dev_triangles)) return glm::vec3(0.0f);
		return getEnvironmentalRadiance(wiW, envMap) * (float)num_lights;
	}

	Light light = dev_lights[randomLightIdx];
	if (light.lightType == AREALIGHT)
	{
		return DirectSampleAreaLight(randomLightIdx, view_point, nor, N_LIGHTS, wiW, pdf, rng, dev_nodes, dev_triangles, light);
	}
    // choose an area light
	return DirectSampleAreaLight(randomLightIdx, view_point, nor, N_LIGHTS, wiW, pdf, rng, dev_nodes, dev_triangles, light);
}

__inline__ __device__ glm::vec3 Evaluate_Li(
	const glm::vec3& wiW,
	const glm::vec3& view_point,
	float& pdf,
	int randomLightIdx,
	int N_LIGHTS,
	cudaTextureObject_t envMap,
	LinearBVHNode* dev_nodes,
	Triangle* dev_triangles,
	Light* dev_lights)
{
	if (envMap != NULL && randomLightIdx == N_LIGHTS - 1)
	{
		pdf = 1.0f / (2.0f * PI);
		return getEnvironmentalRadiance(wiW, envMap) * (float)N_LIGHTS;
	}

	Light light = dev_lights[randomLightIdx];
	ShadeableIntersection isect;
	if (!BVHIntersect(Ray{ view_point, wiW }, dev_nodes, dev_triangles, &isect) || isect.lightId == (uint8_t)(-1))
	{
		return glm::vec3(0.0f, 0.f, 0.f);
	}
	if (light.lightType == AREALIGHT)
	{
		float r = isect.t;
		pdf = r * r / (AbsDot(isect.surfaceNormal, wiW) * light.area + 0.001);
		return (float)N_LIGHTS * light.emission;
	}
	return glm::vec3(0.0f);
}