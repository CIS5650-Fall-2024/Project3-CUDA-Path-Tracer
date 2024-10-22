#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <thrust/random.h>

#include "sceneStructs.h"
#include "utilities.h"

#include "glTFLoader.h"
#include "intersections.h"
#include "interactions.h"

__device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf);

__device__ float Pdf_Li(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, int num_Lights, const glm::vec3& view_point, const glm::vec3& normal, const glm::vec3& wiW,
    const int chosenLightIdx);

__device__ glm::vec3 Sample_Li(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, int num_Lights, const glm::vec3& view_point, const glm::vec3& normal, glm::vec3& wiW, float& pdf,
    int& chosenLightIdx, int& chosenLightID, LightType& chosenLightType,
    thrust::default_random_engine& rng);

__device__ glm::vec3 DirectSampleAreaLight(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, int idx, int num_Lights, const glm::vec3& view_point, const glm::vec3& normal, glm::vec3& wiW, float& pdf,
    thrust::default_random_engine& rng);

__device__ glm::vec3 MISDirectLi(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, cudaTextureObject_t* texObjs,
    int num_Lights,
    const glm::vec3& woWOut, const glm::vec3& view_point, const glm::vec3& normal,
    const Material& m,
    const glm::vec3 texCol,
    const bool useTexCol,
    const bool BVHEmpty,
    thrust::default_random_engine& rng);