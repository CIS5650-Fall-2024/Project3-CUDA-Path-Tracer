#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#include "glTFLoader.h"

using uint = unsigned int;

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + t * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

__host__ __device__ float rectangleIntersectionTest(
    AreaLight light,
    Ray r,
    float radiusU,
    float radiusV,
    const glm::vec3& pos,
    const glm::vec3& normal,
    glm::vec2& UV);

//float rectangleIntersect(vec3 pos, vec3 normal,
//    float radiusU, float radiusV,
//    vec3 rayOrigin, vec3 rayDirection,
//    out vec2 out_uv, mat4 invT);
// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);


__host__ __device__ float triangleIntersectionTest(
    Ray r,
    const MeshTriangle& tri,
    glm::vec3& intersectionPoint,
    glm::vec3& normal);

__device__ void computeBarycentricWeights(const glm::vec3& p, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, glm::vec3& weights);

__device__ bool intersectAABB(const Ray& r, const AABB& aabb);

__device__ bool AreaLightIntersect(ShadeableIntersection& intr, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    AreaLight* areaLights,
    int areaLightIdx);

__device__ bool AllLightIntersectTest(ShadeableIntersection& intr, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    AreaLight* areaLights,
    int num_areaLights);

__device__ bool DirectLightIntersectTest(ShadeableIntersection& intr, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    AreaLight* areaLights,
    int num_areaLights);

//__device__ bool DirectLightBVHIntersect(Ray r,
//    MeshTriangle* triangles, BVHNode* bvhNodes);

__device__ void BVHIntersect(Ray r, ShadeableIntersection& intersection,
    MeshTriangle* triangles, BVHNode* bvhNodes, cudaTextureObject_t* texObjs);

__device__ void SceneIntersect(ShadeableIntersection& isect, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    cudaTextureObject_t* texObjs,
    AreaLight* areaLights,
    int num_areaLights,
    bool BVHEmpty);