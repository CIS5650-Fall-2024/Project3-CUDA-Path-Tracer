#include "warp.h"

__host__ __device__ glm::vec2 squareToUniformDisk(const glm::vec2 &sample) {
    float r = sqrt(sample.x);
    float theta = 2 * M_PIf * sample.y;

    return glm::vec2(r * cos(theta), r * sin(theta));
}

__host__ __device__ glm::vec3 squareToCosineHemisphere(const glm::vec2 &sample, const glm::vec3 &normal)
{
    float up = sqrt(sample.x); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = sample.y * TWO_PI;

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

__host__ __device__ glm::vec3 squareToBeckmann(const glm::vec2 &sample, const float roughness) {
    float phi = sample.x * 2 * M_PIf;
    float theta = atan(roughness * sqrt(-log(1 - sample.y)));

    float sin_theta = sin(theta);
    
    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);
    float z = cos(theta);

    return glm::vec3(x, y, z);
}

__host__ __device__ float squareToBeckmannPdf(const glm::vec3 &m, const float roughness) {
    float cos_theta = m.z;

    if (cos_theta <= 0) {
        return .0f;
    }

    float theta = acos(cos_theta);
    float sin_theta = sin(theta);
    float alpha_sq = roughness * roughness;

    float x_sq = m.x * m.x;
    float y_sq = m.y * m.y;

    // Avoid computing theta by using sin²(theta) = 1 - cos²(theta)
    float cos_theta_sq = cos_theta * cos_theta;
    float sin_theta_sq = 1.0f - cos_theta_sq;
    // sin²(theta) / cos²(theta) = tan²(theta), which is used here
    float tan_theta_sq = sin_theta_sq / cos_theta_sq;
    float exponent = -tan_theta_sq / alpha_sq;

    float cos_theta_cube = cos_theta * cos_theta * cos_theta;
    float denominator = M_PIf * alpha_sq * cos_theta_cube;

    return x_sq + y_sq < 1 ? exp(exponent) / denominator : .0f;
}

