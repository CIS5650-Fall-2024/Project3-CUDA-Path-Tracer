#pragma once

#include "intersections.h"
#include <glm/glm.hpp>
#include <thrust/random.h>

__device__ glm::vec3 sampleTexture(const Texture& texture, const glm::vec2 uv);

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng);

__host__ __device__ glm::vec2 calculateRandomPointOnDisk(
    thrust::default_random_engine &rng);

// Represents a radiance or BSDF sample
struct Sample
{
    glm::vec3 incomingDirection;
    glm::vec3 value;
    float pdf;
    bool delta;
};

// Samples direct illumination from a light
__host__ __device__ Sample sampleLight(
    glm::vec3 viewPoint,
    const Geom &geom,
    const Material *materials,
    thrust::default_random_engine &rng);

__device__ Sample sampleBsdf(
    const Material &material,
    glm::vec2 uv,
    glm::vec3 normal,
    glm::vec3 outgoingDirection,
    thrust::default_random_engine &rng);

__host__ __device__ glm::vec3 getBsdf(
    const Material &material,
    glm::vec3 normal,
    glm::vec3 incomingDirection,
    glm::vec3 outgoingDirection
);

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
__device__ void scatterRay(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    glm::vec2 uv,
    thrust::default_random_engine &rng);
