#pragma once
//#define JITTER 0.5
#define USE_BVH
//#define DEBUG_NORMAL 0 // 1 : clamped, 0 : unclamped
//#define DEBUG_THROUGHPUT
//#define DEBUG_RADIANCE
//#define DEBUG_WORLD_POS
//#define DEBUG_UV
//#define DEBUG_BVH
//////////////////////////////////////////////////////////////////
/////////////////////Post Process Directives//////////////////////
//////////////////////////////////////////////////////////////////
 
//#define POSTPROCESS
#define TONE_MAPPING_ACES 1
#define TONE_MAPPING_REINHARD 1

__device__ __inline__ glm::vec3 ACESFilm(glm::vec3 x)
{
	return glm::clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}
__device__ __inline__ glm::vec3 Reinhard(glm::vec3 x)
{
	return x / (x + glm::vec3(1.0f));
}

__device__ __inline__ float hash13(glm::vec3 v) {
    const glm::vec3 seed = glm::vec3(12.9898, 78.233, 37.719);
    float dotProduct = glm::dot(v, seed);
    float hash = glm::fract(sin(dotProduct) * 43758.5453f);

    return hash;
}