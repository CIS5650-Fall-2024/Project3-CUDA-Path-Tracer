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

__host__ __device__ void scatter_ray(
    PathSegment &path_segment,
    const ShadeableIntersection &intersection,
    const Material &m,
    thrust::default_random_engine &rng)
{
    path_segment.ray.origin = path_segment.ray.origin + path_segment.ray.direction * (intersection.t - EPSILON);

    const auto specular_direction = glm::normalize(glm::reflect(path_segment.ray.direction, intersection.surfaceNormal));
    const auto diffuse_direction = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng));

    // Perfectly Specular
    if (m.hasReflective == 1.0) {
        path_segment.ray.direction = specular_direction;
        path_segment.color *= m.specular.color;
    } else if (m.hasReflective > 0.0) {
        path_segment.ray.direction = glm::mix(diffuse_direction, specular_direction, m.hasReflective);
        path_segment.color *= glm::mix(m.color, m.specular.color, m.hasReflective);
    } else if (m.hasRefractive == 1.0) {
        const auto n1 = 1.0f;
        const auto n2 = m.indexOfRefraction;
        const auto R_0 = glm::pow((n1 - n2) / (n1 + n2), 2.0f);
        const auto cos_theta = glm::dot(intersection.surfaceNormal, path_segment.ray.direction);
        const auto fresnel_factor = R_0 + (1.0f - R_0) * glm::pow(1.0f + cos_theta, 5.0f);

        thrust::uniform_real_distribution<float> u01{0, 1};
        if (u01(rng) > fresnel_factor) { // refract
            if (cos_theta >= 0.0f) {
                path_segment.ray.direction = glm::refract(path_segment.ray.direction, -intersection.surfaceNormal, n2 / n1);
            } else {
                path_segment.ray.direction = glm::refract(path_segment.ray.direction, intersection.surfaceNormal, n1 / n2);
            }
            path_segment.ray.origin += path_segment.ray.direction * 0.01f;
        } else {
            path_segment.ray.direction = specular_direction;
        }
        path_segment.color *= m.specular.color;
    } else if (m.subsurface.translucency > 0.0f) {
        thrust::uniform_real_distribution<float> u01{0, 1};
        if (u01(rng) < m.subsurface.translucency) {
            
            for (int i = 0; i < 10; i++) {
                const auto scatter_distance = -glm::log(u01(rng)) / m.subsurface.thickness;
                path_segment.ray.origin += path_segment.ray.direction * scatter_distance;
                if (u01(rng) < 0.5) {
                // Internal scatter: continue scattering inside the material
                    path_segment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(-path_segment.ray.direction, rng));
                } else {
                    // Attempt to exit: scatter towards the surface normal
                    path_segment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(path_segment.ray.direction, rng));
                }
                path_segment.color *= glm::exp(-m.subsurface.absorption * scatter_distance);
            }

        } else {
            path_segment.ray.direction = diffuse_direction;
            path_segment.color *= m.color;
        }
    } else {
        path_segment.ray.direction = diffuse_direction;
        path_segment.color *= m.color;
    }
}
