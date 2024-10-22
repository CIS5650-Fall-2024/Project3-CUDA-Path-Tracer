#pragma once

#include "intersections.h"
#include <glm/glm.hpp>
#include <thrust/random.h>

const __device__ __constant__ float INV_PI = 0.31830988618379067f;

const __device__ __constant__ float PI_OVER_FOUR = 0.78539816339744831f;

const __device__ __constant__ float PI_OVER_TWO = 1.57079632679489662f;

//Utility Functions
inline __device__ float CosTheta(const glm::vec3& w) { return w.z; }
inline __device__ float AbsCosTheta(const glm::vec3& w) { return glm::abs(w.z); }
inline __device__ glm::vec3 Faceforward(const glm::vec3& n, const glm::vec3& v) {
    return (dot(n, v) < 0.f) ? -n : n;
}
inline __device__ bool SameHemisphere(glm::vec3 w, glm::vec3 wp) {
    return w.z * wp.z > 0;
}
inline __device__ float Cos2Theta(glm::vec3 w) { return w.z * w.z; }
inline __device__ float Sin2Theta(glm::vec3 w) {
    return glm::max(0.f, 1.f - Cos2Theta(w));
}
inline __device__ float Tan2Theta(glm::vec3 w) {
    return Sin2Theta(w) / Cos2Theta(w);
}
inline __device__ float SinTheta(glm::vec3 w) { return sqrt(Sin2Theta(w)); }
inline __device__ float CosPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}
inline __device__ float SinPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}
inline __device__ float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }
inline __device__ float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }
inline __device__ float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }

//BSDF Functions
inline __device__ void coordinateSystem(const glm::vec3 v1, glm::vec3& v2, glm::vec3& v3) {
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

/**
* LocalToWorld returns a mat3 that transforms hemisphere local space (tangent spacce) vectors back to world space
* (0,0,1) is the up vector/normal vector in tangent space
**/
inline __device__ glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    //Obtain tan and bitangent using coordinateSystem(...)
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

/**
* WorldToLocal returns a mat3 that transforms world space vectors to normal aligned hemisphere space vectors
* (0,0,1) is the up vector/normal vector in tangent space
**/
inline __device__ glm::mat3 WorldToLocal(glm::vec3 nor) {
    return transpose(LocalToWorld(nor));
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////SAMPLE WARPING///////////////////////////////////

/**
* Map random uv on a unit square ([0,1],[0,1]) -> to a point on a unit circle with origin (0,0)!
*/

__device__ void squareToDiskConcentric(const glm::vec2 xi, glm::vec3& wi);
//Malley's method
__device__ void squareToHemisphereCosine(const glm::vec2 xi, glm::vec3& wi);

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

///////////////////////////SAMPLE WARPING END/////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//DIFFUSE//
__device__ void f_diffuse(
    glm::vec3& f,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol);

__device__ void pdf_diffuse(
    float& pdf, const glm::vec3& wi);

__device__ void sample_f_diffuse(
    PathSegment& pathSegment,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);


//DIFFUSE//


//SPECULAR//

// GLASS!

__device__ void sample_f_diamond(
    PathSegment& pathSegment,
    const glm::vec3& woOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);

__device__ void sample_f_glass(
    PathSegment& pathSegment,
    const glm::vec3& woOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);

__device__ float FresnelDielectricEval(float cosThetaI);

__device__ void sample_f_diamond_refl(
    PathSegment& pathSegment,
    const glm::vec3& woOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);

__device__ void sample_f_specular_refl(
    PathSegment& pathSegment,
    const glm::vec3& woOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);

__device__ void sample_f_specular_trans(
    PathSegment& pathSegment,
    const glm::vec3& woOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);

//SPECULAR//

//MICROFACET//
__device__ float Lambda(glm::vec3 w, float roughness);
__device__ float TrowbridgeReitzG(glm::vec3 wo, glm::vec3 wi, float roughness);
__device__ float TrowbridgeReitzD(glm::vec3 wh, float roughness);

__device__ float TrowbridgeReitzPdf(glm::vec3 wh, float roughness);

__device__ glm::vec3 sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness);

__device__ glm::vec3 f_microfacet_refl(glm::vec3 col, glm::vec3 woOut, glm::vec3 wi, float roughness);

__device__ void sample_f_microfacet_refl(
    PathSegment& pathSegment,
    const glm::vec3& woOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);


__device__ void sample_f_ceramic_refl(
    PathSegment& pathSegment,
    const glm::vec3& woOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    bool useTexCol,
    thrust::default_random_engine& rng);
//MICROFACET//

/**
* Given an incoming w_o, and an intersection, evaluate the BSDF to find:
*   f(), pdf() and wiW
**/
__device__ void sample_f(
    PathSegment& pathSegment,
    const glm::vec3& woWOut,
    float& pdf,
    glm::vec3 &f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    const bool useTexCol,
    thrust::default_random_engine& rng);

__device__ void f(
    const glm::vec3& woWOut,
    const glm::vec3& wiWOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    const bool useTexCol,
    thrust::default_random_engine& rng);


__device__ void pdf(
    const glm::vec3& woWOut,
    const glm::vec3& wiWOut,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    const glm::vec3 texCol,
    const bool useTexCol,
    thrust::default_random_engine& rng);

