#pragma once

#include <glm/glm.hpp>
#include <thrust/swap.h>

#include "warp.h"

/** Helper Functions */
/** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the angle between the normal and v */
__host__ __device__ float cosTheta(const glm::vec3 &v);
__host__ __device__ float tanTheta(const glm::vec3 &v);
__host__ __device__ void coordinateSystem(const glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3);
__host__ __device__ glm::mat3 LocalToWorld(const glm::vec3 &nor);
__host__ __device__ glm::mat3 WorldToLocal(const glm::vec3 &nor);
__host__ __device__ float fresnel(float cosThetaI, float extIOR, float intIOR);
__host__ __device__ float computeG(const glm::vec3 &v, const glm::vec3 &wh, const float roughness);
/*****************************************************************************/


/** PDFs */
__host__ __device__ float pdfDiffuse(const glm::vec3 &woL, const glm::vec3 &wiL);
__host__ __device__ float pdfMirror();
__host__ __device__ float pdfDielectric();
__host__ __device__ float pdfMicrofacet(const float m_ks, const float roughness, const glm::vec3 &woL, const glm::vec3 &wiL, const glm::vec3 &whL);
/*****************************************************************************/

/** Eval */
__host__ __device__ glm::vec3 evalDiffuse(const glm::vec3 &albedo, const glm::vec3 &woL, const glm::vec3 &wiL);
__host__ __device__ glm::vec3 evalMirror();
__host__ __device__ glm::vec3 evalDielectric();
__host__ __device__ glm::vec3 evalMicrofacet(const glm::vec3 &woL, const glm::vec3 &wiL, const glm::vec3 &whL, const float roughness, const float m_extIOR, const float m_intIOR, const glm::vec3 &m_kd, const float m_ks);
/*****************************************************************************/

/** Bounce Directions and Return Colours */
__host__ __device__ glm::vec3 sampleDiffuse(const glm::vec3 &albedo, const glm::vec3 &normal, const glm::vec2 &sample2D, glm::vec3 &wiW, float &eta);
__host__ __device__ glm::vec3 sampleMirror(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::vec3 &woW, glm::vec3 &wiW, const glm::vec3 &specColour, float &eta);
__host__ __device__ glm::vec3 sampleDielectric(const glm::vec3 normal, glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const float sample1D, const float m_extIOR, const float m_intIOR, const glm::vec3 specColour, glm::vec3 &wiW, float &eta);
__host__ __device__ glm::vec3 sampleMicrofacet(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const glm::vec3 &m_kd, const float m_ks, const float roughness, const float m_extIOR, const float m_intIOR, const glm::vec2 sample2D, glm::vec3 &wiW, float &pdf, float &eta);
/*****************************************************************************/