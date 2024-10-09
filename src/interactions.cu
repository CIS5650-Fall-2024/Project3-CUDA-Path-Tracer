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

__device__ float schlick(float cosine, float refIdx) {
    float r0 = (1 - refIdx) / (1 + refIdx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 scatterDirection;

    if (m.metallic > 0.0f) {
        glm::vec3 reflected = glm::reflect(pathSegment.ray.direction, normal);
        glm::vec3 diffuse = calculateRandomDirectionInHemisphere(normal, rng);
        scatterDirection = glm::normalize((1.0f - m.metallic) * diffuse + m.metallic * reflected);
    }
    else if (m.hasRefractive) {
        float ior = m.indexOfRefraction;
        float eta = 1.f / ior;
        float cosi = glm::dot(pathSegment.ray.direction, normal);
        if (cosi > 0) {
            eta = ior;
        }
        else {
            cosi = -cosi;
        }

        float k = 1.f - eta * eta * (1.f - cosi * cosi);
        if (k < 0.f) {
			// Total internal reflection
            scatterDirection = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            glm::vec3 refracted = glm::normalize(eta * pathSegment.ray.direction + (eta * cosi - sqrtf(k)) * normal);
            float reflectProb = schlick(cosi, eta);
            thrust::uniform_real_distribution<float> u01(0, 1);
			scatterDirection = (u01(rng) < reflectProb) ? glm::reflect(pathSegment.ray.direction, normal) : refracted;
        }
    }
    else {
        scatterDirection = calculateRandomDirectionInHemisphere(normal, rng);
    }

    if (m.roughness > 0.0f) {
        scatterDirection = glm::normalize(scatterDirection + m.roughness * calculateRandomDirectionInHemisphere(normal, rng));
    }

    pathSegment.ray.origin = intersect + scatterDirection * 0.001f;
    pathSegment.ray.direction = scatterDirection;
    pathSegment.remainingBounces--;
}
