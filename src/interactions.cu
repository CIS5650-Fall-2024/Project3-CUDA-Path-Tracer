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

__host__ __device__ glm::vec3 sampleHemisphereAroundDirection(
        const glm::vec3& direction,
        float roughness,
        thrust::default_random_engine& rng)
{
    // Map roughness to cone angle (in radians)
    float coneAngle = roughness * (PI / 2.0f);

    // Uniformly sample the hemisphere around the direction within the cone angle
    thrust::uniform_real_distribution<float> u01(0, 1);
    float u = u01(rng);
    float v = u01(rng);

    // Compute spherical coordinates
    float theta = acos(1.0f - u + u * cos(coneAngle));
    float phi = 2.0f * PI * v;

    // Convert spherical coordinates to Cartesian coordinates
    float sinTheta = sin(theta);
    glm::vec3 sampleDir = glm::vec3(
            sinTheta * cos(phi),
            sinTheta * sin(phi),
            cos(theta)
    );

    // Create an orthonormal basis around the reflection direction
    glm::vec3 w = glm::normalize(direction);
    glm::vec3 uVec = glm::normalize(glm::cross((abs(w.x) > 0.1f ? glm::vec3(0,1,0) : glm::vec3(1,0,0)), w));
    glm::vec3 vVec = glm::cross(w, uVec);

    // Transform sampleDir to align with the reflection direction
    glm::vec3 perturbedDir = glm::normalize(sampleDir.x * uVec + sampleDir.y * vVec + sampleDir.z * w);

    return perturbedDir;
}

__host__ __device__ void handleReflection(
        PathSegment& pathSegment,
        const glm::vec3& intersect,
        const glm::vec3& normal,
        const Material& m,
        thrust::default_random_engine& rng,
        float probabilityReflect)
{
    glm::vec3 perfectReflection  = glm::reflect(pathSegment.ray.direction, normal);

    // Sample a direction around the perfect reflection based on roughness
    glm::vec3 newRayDirection;
    if (m.roughness == 0.0f) {
        // Perfect mirror reflection
        newRayDirection = perfectReflection;
    } else if (m.roughness == 1.0f) {
        // Perfect diffuse reflection
        newRayDirection = calculateRandomDirectionInHemisphere(normal, rng);
    } else {
        // Sample around the reflection direction
        newRayDirection = sampleHemisphereAroundDirection(perfectReflection, m.roughness, rng);
    }

    // Update the ray's origin and direction for reflection
    pathSegment.ray.origin = intersect + normal * 0.001f;
    pathSegment.ray.direction = glm::normalize(newRayDirection);

    pathSegment.color *= m.specular.color / probabilityReflect;
}

__host__ __device__ void handleDiffuse(
        PathSegment& pathSegment,
        const glm::vec3& intersect,
        const glm::vec3& normal,
        const Material& m,
        thrust::default_random_engine& rng,
        float probabilityDiffuse)
{
    glm::vec3 newRayDirection = calculateRandomDirectionInHemisphere(normal, rng);

    // Update the ray's origin and direction for diffuse scattering
    pathSegment.ray.origin = intersect + normal * 0.001f;
    pathSegment.ray.direction = glm::normalize(newRayDirection);

    pathSegment.color *= m.color / probabilityDiffuse;
}

__host__ __device__ void handleRefraction(
        PathSegment& pathSegment,
        const glm::vec3& intersect,
        const glm::vec3& normal,
        const Material& m,
        thrust::default_random_engine& rng)
{
    glm::vec3 refractNormal = normal;
    float eta = pathSegment.insideObject ? m.indexOfRefraction : 1.0f / m.indexOfRefraction;
    float cosThetaI = glm::dot(refractNormal, -pathSegment.ray.direction);

    // If the ray is inside the object
    if (cosThetaI < 0.0f) {
        refractNormal = -refractNormal;
        cosThetaI = -cosThetaI;
    }

    // Compute sin²θ_i for Fresnel equations using Snell's law
    float sin2ThetaI = fmax(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = sin2ThetaI / (eta * eta);

    float reflectance = 1.0f;

    // Check for total internal reflection
    if (sin2ThetaT <= 1.0f) {
        float cosThetaT = sqrtf(fmax(0.0f, 1.0f - sin2ThetaT));

        // Fresnel equations
        float r_parl = ((eta * cosThetaI) - cosThetaT) / ((eta * cosThetaI) + cosThetaT);
        float r_perp = (cosThetaI - (eta * cosThetaT)) / (cosThetaI + (eta * cosThetaT));
        reflectance = (r_parl * r_parl + r_perp * r_perp) / 2.0f;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float randomValue = u01(rng);
    glm::vec3 newRayDirection;

    // Choose between reflection and refraction based on Fresnel reflection coefficient
    if (randomValue < reflectance) {
        // Reflect the ray using glm::reflect
        newRayDirection = glm::reflect(pathSegment.ray.direction, refractNormal);
        pathSegment.color *= m.specular.color / reflectance;
    } else {
        // Refract the ray using glm::refract (Snell's law)
        newRayDirection = glm::refract(pathSegment.ray.direction, refractNormal, eta);

        // Adjust color with transmission color and divide by the probability of refraction
        float transmittance = 1.0f - reflectance;
        pathSegment.color *= m.color / transmittance;

        // Toggle whether we're inside or outside the refractive object
        pathSegment.insideObject = !pathSegment.insideObject;
    }

    // Update the ray's origin and direction
    pathSegment.ray.origin = intersect + newRayDirection * 0.001f;
    pathSegment.ray.direction = glm::normalize(newRayDirection);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float randomValue = u01(rng);

    // Reflectivity, diffuse, and refractive weighting based on material properties
    float reflectivity = glm::length(m.specular.color);  // Reflective component
    float diffuseWeight = glm::length(m.color);          // Diffuse component
    float refractivity = m.hasRefractive > 0.0f ? 1.0f : 0.0f;  // Refractive component
    float sumWeights = reflectivity + diffuseWeight + refractivity;

    if (sumWeights == 0.0f) {
        sumWeights = 1.0f;
    }

    // Normalize the probabilities
    float probabilityReflect = reflectivity / sumWeights;
    float probabilityRefract = refractivity / sumWeights;
    float probabilityDiffuse = diffuseWeight / sumWeights;

    glm::vec3 newRayDirection;

    if (randomValue < probabilityRefract && m.hasRefractive > 0.0f) {
        handleRefraction(pathSegment, intersect, normal, m, rng);
    } else if (randomValue < (probabilityRefract + probabilityReflect) && m.hasReflective > 0.0f) {
        handleReflection(pathSegment, intersect, normal, m, rng, probabilityReflect);
    } else {
        handleDiffuse(pathSegment, intersect, normal, m, rng, probabilityDiffuse);
    }
}
