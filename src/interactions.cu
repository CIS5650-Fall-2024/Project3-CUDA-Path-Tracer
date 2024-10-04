 #include "interactions.h"

//Help from Janet Wang
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphereStratified(
    glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    int sqrtVal = 20;
    float invSqrtVal = 1.f / sqrtVal;
    int numSamples = sqrtVal * sqrtVal;

    int i = glm::min((int)(u01(rng) * numSamples), numSamples - 1);
    int y = i / sqrtVal;
    int x = i % sqrtVal;

    float x_stratified = (x + u01(rng) - 0.5) * invSqrtVal;
    float y_stratified = (y + u01(rng) - 0.5) * invSqrtVal;

    float up = sqrt(x_stratified); 
    float over = sqrt(1 - up * up);
    float around = y_stratified * TWO_PI;

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphereCosWeighed(
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

__host__ __device__
float schlick(float cos, float reflectIndex) {
    float r0 = powf((1.f - reflectIndex) / (1.f + reflectIndex), 2.f);
    return r0 + (1.f - r0) * powf((1.f - cos), 5.f);
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 intersect = intersection.t * pathSegment.ray.direction + pathSegment.ray.origin;
    glm::vec3 normal = intersection.surfaceNormal;
    bool outside = intersection.outside;

    switch (m.shadingType) {
    case ShadingType::Specular:
        specularBSDF(pathSegment, intersect, normal, m, rng);
        break;
    case ShadingType::Diffuse:
        diffuseBSDF(pathSegment, intersect, normal, m, rng);
        break;
    case ShadingType::Refract:
        schlickBTDF(pathSegment, intersect, normal, m, rng);
        break;
    case ShadingType::TexturePBR:
        pbrBSDF(pathSegment, intersect, normal, m, rng, intersection);
        break;
    case ShadingType::SubsurfaceScatter:
        scatterBSSRDF(pathSegment, intersect, normal, m, rng, outside);
        break;
    default:
        // Default case, if none of the above conditions are met
        // Diffuse Black for unknown
        diffuseBSDF(pathSegment, intersect, normal, m, rng);
        pathSegment.color *= glm::vec3(0.f);
        break;
    }

}

__host__ __device__
void diffuseBSDF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
#if COSSAMPLING
    pathSegment.ray.direction = calculateRandomDirectionInHemisphereCosWeighed(normal, rng);
#else
    pathSegment.ray.direction = calculateRandomDirectionInHemisphereStratified(normal, rng);
#endif
    pathSegment.ray.origin = intersect + EPSILON * normal;
    pathSegment.color *= m.color;
}

__host__ __device__
void specularBSDF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.ray.origin = intersect + EPSILON * normal;
    pathSegment.color *= m.color;
}

__host__ __device__
void schlickBTDF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    //Refract
    glm::vec3 inDirection = pathSegment.ray.direction;
    //Dot is positive if normal & inDirection face the same dir -> the ray is inside the object getting out
    bool insideObj = glm::dot(inDirection, normal) > 0.0f;

    //glm::refract (followed raytracing.github.io trick for hollow glass sphere effect by reversing normals)
    float eta = insideObj ? m.indexOfRefraction : (1.0f / m.indexOfRefraction);
    glm::vec3 outwardNormal = insideObj ? -1.0f * normal : normal;
    glm::vec3 finalDir = glm::refract(glm::normalize(inDirection), glm::normalize(outwardNormal), eta);

    //Check for TIR (if magnitude of refracted ray is very small)
    if (glm::length(finalDir) < 0.01f) {
        pathSegment.color *= 0.0f;
        finalDir = glm::reflect(inDirection, normal);
    }

    //Use schlicks to calculate reflective probability (also followed raytracing.github.io)
    float cosine = glm::dot(inDirection, normal);
    float reflectProb = schlick(cosine, m.indexOfRefraction);
    float sampleFloat = u01(rng);

    pathSegment.ray.direction = reflectProb < sampleFloat ? glm::reflect(inDirection, normal) : finalDir;
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
    pathSegment.color *= m.color;

}

void scatterBSSRDF(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, thrust::default_random_engine& rng, bool outside)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random_num = u01(rng);
    glm::vec3 accumulatedColor(1.0f);

    for (int i = 0; i < 10; ++i) {
        if (outside) {
            glm::vec3 insideDirection = calculateRandomDirectionInHemisphereCosWeighed(-normal, rng);
            pathSegment.ray.direction = insideDirection;
            pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;
            outside = false;
        }
        else {
            glm::vec3 insideDirection = calculateRandomDirectionInHemisphereCosWeighed(pathSegment.ray.direction, rng);

            // Attenuation based on path length
            float dist = glm::length(intersect - pathSegment.ray.origin);
            glm::vec3 transmittance = exp(-dist * m.sigma_a);

            accumulatedColor *= transmittance;

            // Check for ray exit based on scattering coefficient
            float probabilityOfScattering = glm::length(m.sigma_s) / (glm::length(m.sigma_s) + glm::length(m.sigma_a));
            if (random_num > probabilityOfScattering) {
                pathSegment.ray.direction = calculateRandomDirectionInHemisphereCosWeighed(pathSegment.ray.direction, rng);
                pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;
                outside = true;
                break; // Exit the loop if the ray leaves the material
            }
            else {
                pathSegment.ray.direction = insideDirection;
                pathSegment.ray.origin = intersect + insideDirection * EPSILON;
            }
        }
    }

    pathSegment.color *= accumulatedColor;
}

__host__ __device__
void pbrBSDF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    const ShadeableIntersection& intersection) {

    float metallic = intersection.metallic;
    float roughness = intersection.roughness;

    glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);
    glm::vec3 randomDir = calculateRandomDirectionInHemisphereCosWeighed(normal, rng);
    glm::vec3 finalDir = glm::mix(reflectDir, randomDir, roughness);

    float reflectiveness = metallic + (1.0f - metallic) * schlick(glm::dot(normal, -pathSegment.ray.direction), 0.04f);

    if (thrust::uniform_real_distribution<float>(0, 1)(rng) < reflectiveness) {
        pathSegment.ray.direction = finalDir;
    }
    else {
        pathSegment.ray.direction = randomDir;
    }

    pathSegment.ray.origin = intersect + EPSILON * normal;
    pathSegment.color *= m.color; 
}