#include "interactions.h"

// Function to calculate a random direction in a hemisphere around the normal
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal
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

    // Generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    // Calculate the random direction
    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Modify scatterRay function
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // Set up RNG
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Russian Roulette Termination Probability
    float continueProbability = glm::min(glm::max(pathSegment.color.r, glm::max(pathSegment.color.g, pathSegment.color.b)), 1.0f);

    if (pathSegment.remainingBounces <= 0 || u01(rng) > continueProbability)
    {
        // Terminate the path
        pathSegment.remainingBounces = 0;
        pathSegment.color = glm::vec3(0.0f);
        return;
    }

    // Adjust pathSegment.color to account for the probability
    pathSegment.color /= continueProbability;

    if (m.hasReflective > 0.0f)
    {
        // Reflective material (same as before)
        glm::vec3 reflectedDirection = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.ray.direction = glm::normalize(reflectedDirection);
        pathSegment.color *= m.specular.color;
    }
    else if (m.hasRefractive > 0.0f)
    {
        // do once the semester is over
    }
    else
    {
        // Diffuse material (same as before)
        glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        float cosTheta = glm::max(0.0f, glm::dot(normal, newDirection));
        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.ray.direction = glm::normalize(newDirection);
        pathSegment.color *= m.color * cosTheta;
    }

    // Decrement remaining bounces
    pathSegment.remainingBounces--;
}