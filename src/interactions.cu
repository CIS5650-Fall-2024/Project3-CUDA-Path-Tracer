#include "interactions.h"
#include "device_launch_parameters.h"

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

__host__ __device__ glm::vec3 diffuseBsdf(glm::vec3& wi, const glm::vec3& normal, const Material& m, thrust::default_random_engine& rng) {
    wi = calculateRandomDirectionInHemisphere(normal, rng);
    return m.color;
}

__host__ __device__ glm::vec3 specularReflectiveBSDF(glm::vec3& wi, const glm::vec3& normal, const glm::vec3& wo, const Material& m) {
    wi = glm::reflect(wo, normal);
    return m.color;
}

//__host__ __device__ void scatterRay(
//    PathSegment & pathSegment,
//    glm::vec3 intersect,
//    glm::vec3 normal,
//    const Material &m,
//    thrust::default_random_engine &rng)
//{
//    // Initialize parameters
//    glm::vec3 wo = pathSegment.ray.direction;
//    glm::vec3 wi;
//    glm::vec3 spectrum;
//
//    // BSDF Evaluation based on materials
//    if (m.hasReflective == 1.0f) {
//        spectrum = specularReflectiveBSDF(wi, normal, wo, m);
//    }
//    else {
//        spectrum = diffuseBsdf(wi, normal, m, rng);
//    }
//
//    // Update pathsegment
//    pathSegment.color *= spectrum;
//    pathSegment.ray.direction = glm::normalize(wi);
//    pathSegment.ray.origin = intersect + normal * EPSILON_F;
//    pathSegment.remainingBounces -= 1;
//}
