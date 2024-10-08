#include "interactions.h"

#include <algorithm>

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






__host__ __device__ float fresnelDielectric(float cosThetaI, float ior) {
    // Clamp cosThetaI to the range [-1, 1]
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // Determine the indices of refraction for entering or exiting the material
    float etaI = 1.0f;   // IOR of air (assumed to be 1.0)
    float etaT = ior;    // IOR of the material (glass, water, etc.)

    // Check if we are inside the material (cosThetaI < 0 means ray is inside the material)
    if (cosThetaI < 0.0f) {
        cosThetaI = -cosThetaI;
        std::swap(etaI, etaT);
    }

    // Calculate the ratio of indices of refraction
    float eta = etaI / etaT;

    // Compute the cosine of the transmitted angle using Snell's law
    float sinThetaT2 = eta * eta * (1.0f - cosThetaI * cosThetaI);

    // Total internal reflection
    if (sinThetaT2 > 1.0f) {
        return 1.0f;  // All light is reflected
    }

    float cosThetaT = glm::sqrt(1.0f - sinThetaT2);

    // Compute the Fresnel reflectance using Schlick's approximation
    float rParallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    // Return the average of the parallel and perpendicular reflectance
    return (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f;
}





__host__ __device__ glm::vec3 sample_f_diffuse(glm::vec3 normal, glm::vec3 incident, glm::vec3 albedo, thrust::default_random_engine& rng, glm::vec3& woW)
{
    woW = calculateRandomDirectionInHemisphere(normal, rng);

    // calculate PDF for cosine-weighted hemisphere sampling
    //float pdf = glm::dot(woW, normal) / PI;

    //// Lambertian BRDF
    //glm::vec3 out_color = albedo / PI;
    //out_color *= glm::dot(woW, normal) / pdf;

    glm::vec3 out_color = albedo;

    return out_color;
}





__host__ __device__ glm::vec3 sample_f_reflection(glm::vec3 normal, glm::vec3 incident, glm::vec3 color, glm::vec3& woW)
{
    woW = glm::reflect(incident, normal);
    return color;
}




__host__ __device__ glm::vec3 sample_f_transmission(glm::vec3 normal, glm::vec3 incident, float ior, glm::vec3 specular_color, glm::vec3& woW)
{

    glm::vec3 out_color;
    glm::vec3 reflection = glm::reflect(incident, normal);

    // Determine if we're entering or exiting the material
    float cosThetaI = glm::dot(incident, normal);
    float eta = (cosThetaI < 0) ? (1.0f / ior) : ior;  // Calculate the ratio of indices
    glm::vec3 refractNormal = (cosThetaI < 0) ? normal : -normal;  // Adjust normal


    // Recalculate cosThetaI for the possibly flipped normal
    cosThetaI = glm::dot(incident, refractNormal);
    float sinThetaTSquared = eta * eta * (1.0f - cosThetaI * cosThetaI);

    // Calculate refracted direction
    woW = glm::refract(incident, refractNormal, eta);

    // Check for total internal reflection
    if (glm::length(woW) == 0)   // sinThetaTSquared > 1.0f doesn't work
    {
        // Total internal reflection: only reflect

        woW = glm::reflect(incident, refractNormal);
        out_color = glm::vec3(0.f);
    }
    else
    {
        out_color = specular_color;
    }

    return out_color;
}





__host__ __device__ glm::vec3 sample_f_glass(glm::vec3 normal, glm::vec3 incident, float ior, glm::vec3 color, thrust::default_random_engine& rng, glm::vec3& woW)
{

    float cosThetaI = glm::dot(incident, normal);
    float fresnelReflectance = fresnelDielectric(cosThetaI, ior);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float reflect_prob = u01(rng);

    if (reflect_prob > 0.5) //reflection 
    {
        glm::vec3 bsdf = sample_f_reflection(normal, incident, color, woW);
        return fresnelReflectance * bsdf * 2.f;

    }
    else
    {
        glm::vec3 bsdf = sample_f_transmission(normal, incident, ior, color, woW);
        return 2.0f * bsdf * (1.0f - fresnelReflectance);
    }
}





__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.


    glm::vec3 bsdf;
    glm::vec3 woW;
    float pdf;

    glm::vec3 incident = pathSegment.ray.direction;
    glm::vec3 color = m.color;
    glm::vec3 specular_color = m.specular.color;
    float ior = m.indexOfRefraction;



    if (m.hasReflective > 0 && m.hasRefractive > 0)
    {
        pdf = 1.f;

        bsdf = sample_f_glass(normal, incident, ior, color, rng, woW);

    }
    else if (m.hasReflective > 0) //pure reflection
    {
        bsdf = sample_f_reflection(normal, incident, color, woW);

    }
    else if (m.hasRefractive > 0)
    {

        //bsdf = sample_f_transmission(normal, incident, ior, specular_color, woW);

        bsdf = sample_f_transmission(normal, incident, ior, color, woW);

    }
    else //pure diffuse
    {
        bsdf = sample_f_diffuse(normal, incident, color, rng, woW);
    }

    pathSegment.ray.direction = glm::normalize(woW);
    pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
    pathSegment.color *= bsdf;

}






