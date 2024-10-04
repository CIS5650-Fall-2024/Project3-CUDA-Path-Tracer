#pragma once

#include <vector>
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "scene.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_MATERIAL_ID 1
#define BVH 1
#define SSAA 1
#define OIDN 1
#define RR 0

#define PI_OVER_TWO       1.5707963267948966192313216916397514420986f
#define PI_OVER_FOUR      0.7853981633974483096156608458198757210493f

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void LoadTextureData(Scene* scene, std::string filename, cudaTextureObject_t& texObj);
void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
__host__ __device__ glm::vec2 RingsProcedualTexture(const glm::vec2& u);
__device__ glm::vec3 checkerboard(glm::vec2 uv);
__device__ glm::vec3 palettes(glm::vec2 uv);
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths);

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
#if BVH
    BVHNode* bvh ,
#endif
    MeshTriangle* meshes
    , Texture* textures
    , glm::vec3* vertices
    , glm::vec3* normals
    , glm::vec2* texcoords);

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture* textures,
    cudaTextureObject_t albedoTexture,
    cudaTextureObject_t normalTexture,
    cudaTextureObject_t metallicTexture,
    cudaTextureObject_t emissiveTexture);

struct isRayAlive;