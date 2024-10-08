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

__host__ __device__  glm::vec3 samplePointOnLight(const Geom& geom, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (geom.type == SPHERE)
    {
        float theta = 2 * PI * u01(rng);
        float phi = acos(2 * u01(rng) - 1);
        float r = geom.scale.x; // Assuming uniform scale for sphere

        glm::vec3 localPoint(
            r * sin(phi) * cos(theta),
            r * sin(phi) * sin(theta),
            r * cos(phi)
        );

        return geom.translation + glm::vec3(geom.transform * glm::vec4(localPoint, 1.0f));
    }
    else if (geom.type == CUBE)
    {
        // Randomly choose one of the 6 faces
        int face = int(u01(rng) * 6);
        float u = u01(rng);
        float v = u01(rng);

        glm::vec3 localPoint;
        switch (face)
        {
        case 0: localPoint = glm::vec3(-0.5f, u - 0.5f, v - 0.5f); break; // Left
        case 1: localPoint = glm::vec3(0.5f, u - 0.5f, v - 0.5f); break; // Right
        case 2: localPoint = glm::vec3(u - 0.5f, -0.5f, v - 0.5f); break; // Bottom
        case 3: localPoint = glm::vec3(u - 0.5f, 0.5f, v - 0.5f); break; // Top
        case 4: localPoint = glm::vec3(u - 0.5f, v - 0.5f, -0.5f); break; // Back
        case 5: localPoint = glm::vec3(u - 0.5f, v - 0.5f, 0.5f); break; // Front
        }

        return geom.translation + glm::vec3(geom.transform * glm::vec4(localPoint, 1.0f));
    }

    // Default case (shouldn't happen)
    return geom.translation;
}


__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    Geom* geoms,
    int geoms_size,
    Material* materials)
{
    if (pathSegment.remainingBounces == 0)
        return;

    normal = glm::normalize(normal);
    glm::vec3 viewDir = -pathSegment.ray.direction;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);

    glm::vec3 newDirection;
    glm::vec3 indirectLight(0.0f);
    glm::vec3 directLight(0.0f);

    if (m.plasticSpecular > 0.0f)
    {
        float F0 = pow((m.indexOfRefraction - 1) / (m.indexOfRefraction + 1), 2);
        float cosTheta = glm::max(glm::dot(normal, viewDir), 0.0f);
        float fresnel = F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);

        if (rand < fresnel)
        {
            // Specular reflection
            newDirection = glm::reflect(-viewDir, normal);

            if (m.roughness > 0.0f)
            {
                glm::vec3 randomDir = calculateRandomDirectionInHemisphere(normal, rng);
                newDirection = glm::normalize(glm::mix(newDirection, randomDir, m.roughness));
            }

            pathSegment.color *= m.specular.color;
        }
        else
        {
            // Diffuse reflection
            newDirection = calculateRandomDirectionInHemisphere(normal, rng);
            pathSegment.color *= m.color;
        }
    }
    else if (m.hasReflective > 0.0f)
    {
        // Metallic reflection
        newDirection = glm::reflect(-viewDir, normal);

        if (m.roughness > 0.0f)
        {
            glm::vec3 randomDir = calculateRandomDirectionInHemisphere(normal, rng);
            newDirection = glm::normalize(glm::mix(newDirection, randomDir, m.roughness));
        }

        pathSegment.color *= glm::mix(m.specular.color, m.color * glm::dot(newDirection, normal), m.metallic);
    }
    else if (m.hasRefractive > 0.0f)
    {
        // Refractive surface
        float n1 = 1.0f; // Assume air as the surrounding medium
        float n2 = m.indexOfRefraction;
        float eta = n1 / n2;
        float cosThetaI = glm::dot(-pathSegment.ray.direction, normal);
        float k = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

        if (k >= 0.0f)
        {
            newDirection = eta * pathSegment.ray.direction + (eta * cosThetaI - sqrtf(k)) * normal;
            pathSegment.color *= m.color;
        }
        else
        {
            newDirection = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
        }
    }
    else
    {
        // Diffuse reflection
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    indirectLight = pathSegment.color;

#if DIRECT_LIGHTING
    // Compute direct lighting
    for (int i = 0; i < geoms_size; ++i)
    {
        if (materials[geoms[i].materialid].emittance > 0.0f)
        {
            glm::vec3 lightPoint = samplePointOnLight(geoms[i], rng);
            glm::vec3 lightDir = glm::normalize(lightPoint - intersect);
            float lightDistance = glm::length(lightPoint - intersect);

            Ray shadowRay;
            shadowRay.origin = intersect + normal * 0.001f; // Offset to avoid self-intersection
            shadowRay.direction = lightDir;
            bool occluded = false;

            for (int j = 0; j < geoms_size; ++j)
            {
                if (j != i) // Don't check intersection with the light itself
                {
                    float t;
                    glm::vec3 tmpIntersect, tmpNormal;
                    bool tmpOutside;
                    if (geoms[j].type == CUBE)
                        t = boxIntersectionTest(geoms[j], shadowRay, tmpIntersect, tmpNormal, tmpOutside);
                    else if (geoms[j].type == SPHERE)
                        t = sphereIntersectionTest(geoms[j], shadowRay, tmpIntersect, tmpNormal, tmpOutside);

                    if (t > 0 && t < lightDistance)
                    {
                        occluded = true;
                        break;
                    }
                }
            }

            if (!occluded)
            {
                float cosTheta = glm::max(glm::dot(normal, lightDir), 0.0f);
                glm::vec3 lightColor = materials[geoms[i].materialid].color;
                float lightIntensity = materials[geoms[i].materialid].emittance;

                glm::vec3 brdf = m.color / PI;

                directLight += (brdf * lightColor * lightIntensity * cosTheta) / (lightDistance * lightDistance);
            }
        }
    }
#endif
    pathSegment.color = indirectLight + directLight;
    pathSegment.ray.origin = intersect + newDirection * 0.001f;
    pathSegment.ray.direction = glm::normalize(newDirection);
    pathSegment.remainingBounces--;
}
