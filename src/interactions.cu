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

__host__ __device__ float reflectance(float cosine, float refraction_index) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * glm::pow((1.0 - cosine), 5.0);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec2 uv,
    bool outside,
    const Material &m,
    glm::vec3 materialColor,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 newDirection;
    glm::vec3 currDir = normalize(pathSegment.ray.direction);
    glm::vec3 n = normalize(normal);

    pathSegment.ray.origin = intersect + 0.01f * n;

    if (m.hasReflective) {
        glm::vec3 reflectedDir = glm::reflect(currDir, n);
        // Perfect specular reflection
        newDirection = reflectedDir;
        pathSegment.color *= materialColor;
    }
    else if (m.hasRefractive) {
        float rand_f = u01(rng);
        
        float cos_theta = glm::min(1.f, dot(-currDir, n));
        float schlick_factor = reflectance(cos_theta, m.indexOfRefraction);
        if (schlick_factor >= rand_f) {
            newDirection = normalize(glm::reflect(currDir, n));
            pathSegment.ray.origin = intersect + 0.01f * newDirection;
            pathSegment.color *= materialColor;
        }
        else {
            float eta = m.indexOfRefraction;

            if (outside) {
                eta = 1.0 / m.indexOfRefraction;
            }
            glm::vec3 refracted = glm::refract(currDir, n, eta);
            newDirection = normalize(refracted);
            pathSegment.ray.origin = intersect + 0.01f * newDirection;
            pathSegment.color *= materialColor;
        }
        
    }
    else {
        newDirection = calculateRandomDirectionInHemisphere(n, rng);
        pathSegment.color *= materialColor;
    }
    pathSegment.ray.direction = normalize(newDirection);

    pathSegment.remainingBounces--;
}
