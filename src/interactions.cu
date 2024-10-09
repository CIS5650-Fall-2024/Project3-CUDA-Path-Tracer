#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void sampleDiffuse(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    float prob)
{
   pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
   pathSegment.color *= m.color / prob;
}

__host__ __device__ void sampleRefl(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    float prob)
{
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.color *= m.specular.color / prob;
}

__host__ __device__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, glm::normalize(wi));
    float sin2ThetaI = fmaxf(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (!(1.0f - sin2ThetaT)) return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    wt = eta * (wi - n * cosThetaI) - n * cosThetaT;
    return true;
}

__host__ __device__ void sampleRefract(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    float prob)
{
    float etaA = 1.f;
    float etaB = m.indexOfRefraction;
    bool entering = (glm::dot(pathSegment.ray.direction, normal) <= 0);
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;

    glm::vec3 N = entering ? normal : -normal;

    glm::vec3 refractedDir = glm::reflect(pathSegment.ray.direction, N);

    Refract(pathSegment.ray.direction, N, etaI / etaT, refractedDir);
    pathSegment.ray.direction = refractedDir;
    pathSegment.color *= m.specular.color / prob;
}

__host__ __device__ float fresnel(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m) 
{
    float cosTheta = abs(glm::dot(pathSegment.ray.direction, normal));
    float R0 = (1.f - m.indexOfRefraction) / (1.f + m.indexOfRefraction);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1.f - cosTheta, 5.f);
}

__host__ __device__ glm::vec3 sampleGGXNormal(
    const glm::vec3& normal,
    float roughness,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float u1 = u01(rng);
    float u2 = u01(rng);

    float a = roughness * roughness;
    float phi = 2.0f * PI * u1;
    float cosTheta = glm::sqrt((1.f - u2) / (1.0f + (a * a - 1.f) * u2));
    float sinTheta = glm::sqrt(1.f - cosTheta * cosTheta);

    glm::vec3 h = glm::vec3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);

    //To world space
    glm::vec3 up = glm::abs(normal.z) < 0.999f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    glm::vec3 tangentX = glm::normalize(glm::cross(up, normal));
    glm::vec3 tangentY = glm::cross(normal, tangentX);

    return glm::normalize(h.x * tangentX + h.y * tangentY + h.z * normal);
}

__host__ __device__ float ggxDistribution(
    const glm::vec3& normal,
    const glm::vec3& halfVector,
    float roughness)
{
    float a = roughness * roughness;
    float NdotH = glm::max(glm::dot(normal, halfVector), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a - 1.0f) + 1.0f;
    return (a * a) / (PI * denom * denom);
}

__host__ __device__ float lambda(const float& cosTheta, float a)
{
    if (cosTheta < EPSILON) return 0.f;
    float tan2Theta = powf(cosTheta, 4.f) * a;
    return (-1 + sqrt(1.f + tan2Theta)) / 2.f;
}

__host__ __device__ float smithGeometry(const float& cosThetaO, const float& cosThetaI, float roughness)
{
    float a = roughness * roughness;
    return 1.f / (1.f + lambda(cosThetaO, a) + lambda(cosThetaI, a));
}

__host__ __device__ glm::vec3 fresnelSchlick(
    float cosTheta,
    const glm::vec3& F0)
{
    return F0 + (glm::vec3(1.0f) - F0) * glm::pow(1.0f - cosTheta, 5.0f);
}

__host__ __device__ float ggxPDF(
    const glm::vec3& normal,
    const glm::vec3& viewDirection,
    const glm::vec3& halfVector,
    float roughness)
{
    float D = ggxDistribution(normal, halfVector, roughness);

    float VdotH = glm::max(glm::dot(viewDirection, halfVector), EPSILON);
    float NdotH = glm::max(glm::dot(normal, halfVector), 0.0f);

    return (D * NdotH) / (4.0f * VdotH);
}

__host__ __device__ void MicrofacetReflection(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    const float& roughness)
{
    float alpha = roughness * roughness;
    glm::vec3 wh = sampleGGXNormal(normal, roughness, rng);

    glm::vec3 reflectedDirection = glm::reflect(pathSegment.ray.direction, wh);

    //if roughness if very small, treat it as perfectly specular
    if (roughness < EPSILON)
    {
        pathSegment.color *= m.specular.color;
    }
    else
    {
        float cosThetaI = glm::max(glm::dot(normal, -pathSegment.ray.direction), EPSILON);
        float cosThetaO = glm::max(glm::dot(normal, reflectedDirection), EPSILON);

        glm::vec3 F = fresnelSchlick(glm::dot(reflectedDirection, wh), m.color);
        float D = ggxDistribution(normal, wh, roughness);
        float G = smithGeometry(cosThetaO, cosThetaI, alpha);

        glm::vec3 spec = (m.specular.color * F * D * G / (4.f * cosThetaI * cosThetaO)) / ggxPDF(normal, -pathSegment.ray.direction, wh, roughness);
        glm::vec3 diffuse = m.color * (glm::vec3(1.f) - F) / PI;
        pathSegment.color *= (diffuse + spec);
    }
    pathSegment.ray.direction = reflectedDirection;
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    
    float probDiffuse = 0.f;

    float totalIntensity = glm::length(m.color) + glm::length(m.specular.color);

    if (totalIntensity > 0.f) {
        probDiffuse = glm::length(m.color) / totalIntensity;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);

    if (m.microfacet.isMicrofacet)
    {
        MicrofacetReflection(pathSegment, normal, m, rng, m.microfacet.roughness);
    }
    else if (rand < probDiffuse) {
        //diffuse shading
        sampleDiffuse(pathSegment, normal, m, rng, probDiffuse);
    }
    else {
        //glass-like material
        if (m.hasReflective > 0.f && m.hasRefractive > 0.f) {
            float f = fresnel(pathSegment, normal, m);
            if (rand < f) {
                sampleRefl(pathSegment, normal, m, 1.f);
            }
            else {
                sampleRefract(pathSegment, normal, m, 1.f);
            }
        }
        else if (m.hasReflective > 0.f) {
            //reflection
            //divide color by 1 - probDiffuse
            sampleRefl(pathSegment, normal, m, 1.f - probDiffuse);
        }
        else if (m.hasRefractive > 0.f) {
            //refraction
            //divide color by 1 - probDiffsue
            sampleRefract(pathSegment, normal, m, 1.f - probDiffuse);
        }
        else {
            //diffuse shading
            //divide color by 1 - probDiffuse
            //this probably shouldn't ever happen if our material is valid
            //because this means that m.specular.color != 0 but hasReflective & hasRefractive == 0
            if (probDiffuse != 0.f) {
                sampleDiffuse(pathSegment, normal, m, rng, 1.f - probDiffuse);
            }
        }
    }
   
    //printf("Color (%f, %f, %f)\n", pathSegment.color.x, pathSegment.color.y, pathSegment.color.z);
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
}
