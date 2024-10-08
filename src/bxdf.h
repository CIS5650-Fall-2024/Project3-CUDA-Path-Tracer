#pragma once

#include <glm/glm.hpp>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include "sceneStructs.h"

__host__ __device__ float DielectricFresnel(
    float cosThetaI,
    float etaI, 
    float etaT
);

__host__ __device__ void SpecularBRDF(
    PathSegment& pathSegment,
    const Material& material,
    glm::vec3 intersect,
    glm::vec3 normal
);

__host__ __device__ void SpecularBTDF(
    PathSegment& pathSegment,
    const Material& material,
    glm::vec3 intersect,
    glm::vec3 normal
);

__host__ __device__ void DielectricBxDF(
    PathSegment& pathSegment,
    const Material& material,
    glm::vec3 intersect,
    glm::vec3 normal,
    thrust::default_random_engine& rng
);

__host__ __device__ glm::vec3 GGXBRDF(
    const glm::vec3 &intersection,
    const glm::vec3 &normal,
    const glm::vec3 &incomingRay,
    const glm::vec3 &reflectedRay,
    const Material &material);

__host__ __device__ glm::vec3 sampleGGXNormal(
    const glm::vec3 &N, 
    float roughness, 
    thrust::default_random_engine &rng
);