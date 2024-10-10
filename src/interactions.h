#pragma once

#include "intersections.h"
#include <glm/glm.hpp>
#include <thrust/random.h>
// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);


__host__ __device__ float fresnelDielectric(float cosThetaI, float ior);

__host__ __device__ glm::vec3 sample_f_glass(
    glm::vec3 &normal, glm::vec3 rayDir, float IOR, glm::vec3 color,
    glm::vec3 &wiW, thrust::default_random_engine& rng);

__host__ __device__ glm::vec3 sample_f_specular_reflection(
    glm::vec3 normal, glm::vec3 rayDir, glm::vec3 color, glm::vec3 &wiW);

__host__ __device__ glm::vec3 sample_f_specular_transmission(
    glm::vec3 &normal, glm::vec3 rayDir, float IOR, glm::vec3 color,
	glm::vec3 &wiW);

__host__ __device__ glm::vec3 sample_f_specular_plastic(
	glm::vec3 normal, glm::vec3 rayDir, glm::vec3 color, float roughness,
	glm::vec3 &wiW, thrust::default_random_engine& rng, float &pdf);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    glm::vec3 textureColor);
