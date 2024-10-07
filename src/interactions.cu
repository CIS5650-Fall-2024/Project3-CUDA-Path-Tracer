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

__host__ __device__ void kernBasicScatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    if (pathSegment.remainingBounces <= 0) return;
    
    glm::vec3 out_color(1.0);
    glm::vec3 Li;
    glm::vec3 Lo;

    Li = glm::normalize(pathSegment.ray.direction);
    if (m.hasReflective) {

        Lo = glm::reflect(Li, normal);
    }
    else {
        Lo = calculateRandomDirectionInHemisphere(normal, rng);
        
    }
    out_color = m.color;
    
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = glm::normalize(Lo);
    pathSegment.color *= out_color;
}

__host__ __device__ glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, glm::vec3 reflect, glm::vec3 tangent,
    thrust::default_random_engine& rng,
    float roughness) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    float x1 = u01(rng);
    float x2 = u01(rng);

    float theta = atan(roughness * sqrt(x1) / sqrt(1 - x1));
    float phi = 2 * PI * x2;

    glm::vec3 dir;
    dir.x = cos(phi) * sin(theta);
    dir.y = sin(phi) * sin(theta);
    dir.z = cos(theta);

    glm::vec3 r = glm::normalize(reflect);
    glm::mat3 m;
    m[2] = r;
    m[0] = glm::normalize(glm::vec3(0, r.z, -r.y));
    m[1] = glm::cross(m[2], m[1]);
    dir = glm::normalize(m * dir);
    glm::vec3 t(tangent);
    glm::vec3 b = glm::cross(normal, t);
    dir = t * dir.x + b * dir.y + normal * dir.z;

    return dir;
}

__host__ __device__ void scatterRay(
    ShadeableIntersection& i,
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 intersect_color,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO:
    if (pathSegment.remainingBounces <= 0) return;

    glm::vec3 out_color(1.0);
    glm::vec3 Li;
    glm::vec3 Lo;

    Li = glm::normalize(pathSegment.ray.direction);
    // perfect mirror
    if (m.hasReflective == 1.0f) {
        
        Lo = glm::reflect(Li, normal);
        out_color *= intersect_color;
    }
    // imperfect specular
    else if (m.hasReflective > 0.0f) {
        glm::vec3 refl = glm::reflect(Li, normal);
        Lo = calculateImperfectSpecularDirection(normal, refl, i.tangent, rng, m.hasReflective);
        out_color *= intersect_color;
    }
    // refraction
    else if (m.hasRefractive > 0.0f) {
        thrust::uniform_real_distribution<float> d01(0.0f, 1.0f);
        float rn = d01(rng);
        float cosine = -glm::dot(pathSegment.ray.direction, normal);
        float n = m.indexOfRefraction;
        float r = glm::pow((1.0f - n) / (1.0f + n), 2.0f);
        float f = (r + (1.0f - r) * glm::pow(1.0f - cosine, 5.0f));
        if (rn > f) {
            float ra = 1.0f / m.indexOfRefraction;
            if (glm::dot(pathSegment.ray.direction, normal) >= 0.0f) {
                normal = -normal;
                ra = m.indexOfRefraction;
            }
            Lo = glm::refract(pathSegment.ray.direction, normal, ra);
        }
    }
    // diffuse
    else {
        Lo = calculateRandomDirectionInHemisphere(normal, rng);
        out_color *= intersect_color;

    }
    if (m.hasRefractive > 0.0f) {
        pathSegment.ray.origin = intersect + Lo * 0.1f;
    }
    else {
        pathSegment.ray.origin = intersect;
    }
    pathSegment.ray.direction = glm::normalize(Lo);
    pathSegment.color *= out_color;
    
}
