#pragma once

#include <glm/glm.hpp>
#include <thrust/random.h>

#include "utilities.h"

__host__ __device__ glm::vec2 squareToUniformDisk(const glm::vec2 &sample);
__host__ __device__ glm::vec3 squareToCosineHemisphere(const glm::vec2 &sample, const glm::vec3 &normal);
__host__ __device__ glm::vec3 squareToBeckmann(const glm::vec2 &sample, const float roughness);
__host__ __device__ float squareToBeckmannPdf(const glm::vec3 &m, const float roughness);