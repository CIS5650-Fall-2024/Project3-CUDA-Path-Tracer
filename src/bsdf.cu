#include "bsdf.h"

/** Helper Functions */
/** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the angle between the normal and v */
__host__ __device__ float cosTheta(const glm::vec3 &v) {
    return v.z;
}

__host__ __device__ float tanTheta(const glm::vec3 &v) {
    float temp = 1 - v.z * v.z;

    if (temp <= 0.0f) {
        return 0.0f;
    }
        
    return sqrt(temp) / v.z;
}

__host__ __device__ void coordinateSystem(const glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3) {
    if (abs(v1.x) > abs(v1.y)) {
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }
    else {
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
            
    v3 = cross(v1, v2);
}

__host__ __device__ glm::mat3 LocalToWorld(const glm::vec3 &nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__host__ __device__ glm::mat3 WorldToLocal(const glm::vec3 &nor) {
    return transpose(LocalToWorld(nor));
}

__host__ __device__ float fresnel(float cosThetaI, float extIOR, float intIOR) {
    float etaI = extIOR, etaT = intIOR;

    if (extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
       at the inside of the object */
    if (cosThetaI < 0.0f) {
        thrust::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
          sinThetaTSqr = eta*eta * (1-cosThetaI*cosThetaI);

    if (sinThetaTSqr > 1.0f)
        return 1.0f;  /* Total internal reflection! */

    float cosThetaT = sqrt(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosThetaI - etaT * cosThetaT)
             / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT)
             / (etaT * cosThetaI + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) / 2.0f;
}

__host__ __device__ float computeG(const glm::vec3 &v, const glm::vec3 &wh, const float roughness) {
    float v_dot_wh = glm::dot(v, wh); 
    float c = v_dot_wh / cosTheta(v);

    if (c <= 0) {
        return 0.0f;
    }

    float b = 1.0f / (roughness * tanTheta(v));
    if (b < 1.6) {
        float b_sqr = b * b;
        float nom = 3.535 * b + 2.181 * b_sqr;
        float denom = (1 + 2.276 * b + 2.577 * b_sqr);
        return nom / denom;
    }
        
    return 1.0f;
}
/*****************************************************************************/

/** PDFs */
__host__ __device__ float pdfDiffuse(const glm::vec3 &woL, const glm::vec3 &wiL){
    float cosThetaWiL = cosTheta(wiL);

    if (cosTheta(woL) <= 0 || cosThetaWiL <= 0) {
        return 0.0f;
    }

    return M_1_PIf * cosThetaWiL;
}

__host__ __device__ float pdfMirror(){
    return 0.0f;
}

__host__ __device__ float pdfDielectric(){
    return 0.0f;
}

__host__ __device__ float pdfMicrofacet(const float m_ks, const float roughness, const glm::vec3 &woL, const glm::vec3 &wiL, const glm::vec3 &whL) {
    float cosThetaWiL = cosTheta(wiL);
    if (cosTheta(woL) <= 0 || cosThetaWiL <= 0) {
        return .0f;
    }

    float D = squareToBeckmannPdf(whL, roughness);
    float Jh = 1.0f / (4 * glm::dot(whL, wiL));

    return m_ks * D * Jh + (1 - m_ks) * cosThetaWiL / M_PIf;
}
/*****************************************************************************/

/** Eval */
__host__ __device__ glm::vec3 evalDiffuse(const glm::vec3 &albedo, const glm::vec3 &woL, const glm::vec3 &wiL) {
    if (cosTheta(woL) <= 0 || cosTheta(wiL) <= 0) {
        return glm::vec3(0.0f);
    }

    return albedo * M_1_PIf;
}

__host__ __device__ glm::vec3 evalMirror() {
    return glm::vec3(0.0f);
}

__host__ __device__ glm::vec3 evalDielectric() {
    return glm::vec3(0.0f);
}

__host__ __device__ glm::vec3 evalMicrofacet(const glm::vec3 &woL, const glm::vec3 &wiL, const glm::vec3 &whL, const float roughness, const float m_extIOR, const float m_intIOR, const glm::vec3 &m_kd, const float m_ks) {
    float cosThetaWiL = cosTheta(wiL);
    float cosThetaWoL = cosTheta(woL);
    
    if (cosThetaWoL <= 0 || cosThetaWiL <= 0) {
        return glm::vec3(0.0f);
    }

    float wh_dot_woL = glm::dot(whL, woL);

    float D = squareToBeckmannPdf(whL, roughness);
    float F = fresnel(wh_dot_woL, m_extIOR, m_intIOR);
    float G = computeG(woL, whL, roughness) * computeG(wiL, whL, roughness);

    return m_kd / M_PIf + m_ks * D * F * G / (4 * cosThetaWoL * cosThetaWiL * cosTheta(whL));
}

/** Bounce Directions and Return Colours */
__host__ __device__ glm::vec3 sampleDiffuse(const glm::vec3 &albedo, const glm::vec3 &normal, const glm::vec2 &sample2D, glm::vec3 &wiW, float &eta) {
    wiW = glm::normalize(squareToCosineHemisphere(sample2D, normal));
    eta = 1.0f;

    return albedo;
}

__host__ __device__ glm::vec3 sampleMirror(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::vec3 &woW, glm::vec3 &wiW, const glm::vec3 &specColour, float &eta) {
    glm::vec3 woL = glm::normalize(worldToLocal * woW);

    if (cosTheta(woL) <= 0.0f) {
        wiW = glm::vec3(0.0f);
    }

    // Note that glm::reflect equation is woW - 2 * glm::dot(woW, normal) * normal
    // Instead of normally what we would have: 2 * glm::dot(woW, normal) * normal - woW
    wiW = glm::normalize(glm::reflect(-woW, normal));
    eta = 1.0f;

    return specColour;
}

__host__ __device__ glm::vec3 sampleDielectric(const glm::vec3 normal, glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const float sample1D, const float m_extIOR, const float m_intIOR, const glm::vec3 specColour, glm::vec3 &wiW, float &eta) {
    glm::vec3 woL = glm::normalize(worldToLocal * woW);
    glm::vec3 normalLocal = glm::vec3(0.0f, 0.0f, 1.0f);
    float cosThetaWoL = cosTheta(woL);
    float eta1 = m_extIOR;
    float eta2 = m_intIOR;
    float F = fresnel(cosTheta(woL), m_extIOR, m_intIOR);

    if (cosThetaWoL < 0.0f) {
        // Ray is hitting from the inside
        thrust::swap(eta1, eta2);
        cosThetaWoL = -cosThetaWoL;
        normalLocal = -normalLocal;
    }

    float indexRatio = eta1 / eta2;
    float indexRatio_sq = indexRatio * indexRatio;
    float cosThetaWoL_sq = cosThetaWoL * cosThetaWoL;
    float weightN = glm::max(0.0f, 1 - indexRatio_sq * (1 - cosThetaWoL_sq));
    
    if (sample1D <= F || weightN < 0.0f) {
        glm::vec3 c = sampleMirror(normal, worldToLocal, woW, wiW, specColour, eta);
        eta = eta1; // Put this line after the sampleMirror function call becasue sampleMirror changes eta to 1.0f
        return c;
    }
    else {
        wiW = glm::normalize(localToWorld * (-indexRatio * (woL - cosThetaWoL * normalLocal) - normalLocal * sqrt(weightN)));
        eta = eta2;

        return specColour / indexRatio_sq;
    }
}

__host__ __device__ glm::vec3 sampleMicrofacet(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const glm::vec3 &m_kd, const float m_ks, const float roughness, const float m_extIOR, const float m_intIOR, const glm::vec2 sample2D, glm::vec3 &wiW, float &pdf, float &eta) {
    glm::vec3 woL = glm::normalize(worldToLocal * woW);
    glm::vec3 wiL;
    glm::vec2 sample;

    if (sample2D.x < m_ks) {
        sample = glm::vec2((sample2D.x - m_ks) / (1 - m_ks), sample2D.y);
        wiW = glm::normalize(squareToCosineHemisphere(sample, normal));
        wiL = glm::normalize(worldToLocal * wiW);
    }
    else {
        sample = glm::vec2(sample2D.x / m_ks, sample2D.y);
        glm::vec3 n = squareToBeckmann(sample, roughness);
        wiL = glm::reflect(-woL, n);
        wiW = glm::normalize(localToWorld * wiL);
    }

    eta = 1.0f;

    float cosTheta_wiL = cosTheta(wiL);
    if (cosTheta_wiL <= 0.0f || cosTheta(woL) <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 whL = glm::normalize(wiL + woL);
    pdf = pdfMicrofacet(m_ks, roughness, woL, wiL, whL);

    return evalMicrofacet(woL, wiL, whL, roughness, m_extIOR, m_intIOR, m_kd, m_ks) * cosTheta_wiL / pdf;
}
/*****************************************************************************/