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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    
    // Scatter ray according to the material's properties
    if (m.hasReflective == 1.0f) {
        // A basic implementation of pure-diffuse (Lambertian) shading
        glm::vec3 newRayDirection = glm::reflect(pathSegment.ray.direction, normal);

        // Update the path segment with new ray data
        pathSegment.ray.origin = intersect + normal * 0.001f;  // Small offset to prevent self-intersection
        pathSegment.ray.direction = glm::normalize(newRayDirection);

        // Multiply the color of the path segment by the material's diffuse color
        pathSegment.color *= m.color;

        // Decrease the remaining bounces for this path segment
        pathSegment.remainingBounces--;

        // check if remainingBounces is 0
        if (pathSegment.remainingBounces == 0)
        {
            pathSegment.color *= 0.0f;
        }
    }
    else if (m.hasRefractive == 1.0f) {
        // Implement refractive material logic if needed in the future.
        glm::vec3 newRayDirection = glm::normalize(pathSegment.ray.direction);
        glm::vec3 newRayNormal = glm::normalize(normal);

        thrust::uniform_real_distribution<float> u01(0, 1);
        float randomNum = u01(rng);

        // compute the Fresnel factor using the Schlick's approximation
        const float cos = glm::dot(newRayNormal, newRayDirection);
        const float n_1 = 1.0f;
        const float n_2 = m.indexOfRefraction;
        const float r_0 = glm::pow((n_1 - n_2) / (n_1 + n_2), 2.0f);
        const float factor = r_0 + (1.0f - r_0) * glm::pow(1.0f + cos, 5.0f);

        if (randomNum > factor)
        {
            // compute the refraction ratio based on the material's index of refraction
            float ratio = 1.0f / m.indexOfRefraction;

            // determine whether the ray exiting from the surface
            if (cos >= 0.0f)
            {
                // update the normal
                newRayNormal = -newRayNormal;
                // update the refraction ratio
                ratio = m.indexOfRefraction;
            }

            // compute the refracted ray direction
            pathSegment.ray.direction = glm::refract(newRayDirection, newRayNormal, ratio);

            // set the new ray's origin
            pathSegment.ray.origin += pathSegment.ray.direction * 0.01f;
        }
        else
        {
            // reflect the ray's direction when the Fresnel factor is big
            pathSegment.ray.direction = glm::reflect( pathSegment.ray.direction, newRayNormal);
        }

        // accumulate the output color
        pathSegment.color *= m.color;

    }
    else {
        // A basic implementation of pure-diffuse (Lambertian) shading
        glm::vec3 newRayDirection = calculateRandomDirectionInHemisphere(normal, rng);

        // Update the path segment with new ray data
        pathSegment.ray.origin = intersect + normal * 0.001f;  // Small offset to prevent self-intersection
        pathSegment.ray.direction = glm::normalize(newRayDirection);

        // Multiply the color of the path segment by the material's diffuse color
        pathSegment.color *= m.color;

        // Decrease the remaining bounces for this path segment
        pathSegment.remainingBounces--;

        // check if remainingBounces is 0
        if (pathSegment.remainingBounces == 0)
        {
            pathSegment.color *= 0.0f;
        }
    }
}
