#include "interactions.h"
#include "bxdf.h"

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

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& material,
    thrust::default_random_engine& rng
) {
    if (material.type == SKIN)
    {
        glm::vec3 diffuseDir = calculateRandomDirectionInHemisphere(normal, rng);
        glm::vec3 subsurfaceNormal = normal + material.subsurfaceScattering * calculateRandomDirectionInHemisphere(normal, rng);
        glm::vec3 finalDirection = glm::normalize(glm::mix(diffuseDir, subsurfaceNormal, material.subsurfaceScattering));
        pathSegment.ray.direction = finalDirection;
        pathSegment.color *= material.color;
        pathSegment.ray.origin = intersect + 0.1f * finalDirection;
    }
    else if (material.type == GGX)
    {
        glm::vec3 H = sampleGGXNormal(normal, material.roughness, rng);

        glm::vec3 incomingRay = pathSegment.ray.direction;
        glm::vec3 reflectedRay = glm::reflect(incomingRay, H);

        glm::vec3 brdfValue = GGXBRDF(intersect, normal, incomingRay, reflectedRay, material);

        pathSegment.ray.direction = reflectedRay;
        pathSegment.color *= brdfValue;
        pathSegment.ray.origin = intersect + 0.1f * reflectedRay;
    }
    else if (material.type == SPECULAR)
    {
    // if (material.hasReflective == 1.0f) {
        SpecularBRDF(pathSegment, material, intersect, normal);
    }
    // else if (material.hasRefractive > 0.0f) {
    else if (material.type == DIELECTRIC)
    {
        DielectricBxDF(pathSegment, material, intersect, normal, rng);
    }
    else if (material.type == LIGHT)
    {
        pathSegment.color *= (material.color * material.emittance);
        pathSegment.remainingBounces = 0;
    }
    else  if (material.type == DIFFUSE)
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= material.color;
        pathSegment.ray.origin = intersect + 0.1f * normal;
    }
}
