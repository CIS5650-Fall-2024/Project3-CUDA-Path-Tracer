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
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    float light_intensity = 1.0f;
    glm::vec3 intersect_val = glm::normalize(intersect);
    normal = glm::normalize(normal);
    if (pathSegment.remainingBounces <= 0)
        return;

    glm::vec3 incident_vector = glm::normalize(pathSegment.ray.direction);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float randomValue = u01(rng);

    if(m.hasRefractive > 0.0f) //refract
    {
        float cosTheta_i = glm::dot(incident_vector, normal);
        float eta = (cosTheta_i > 0.f) ? (1 / m.indexOfRefraction) : (m.indexOfRefraction);
        cosTheta_i = glm::abs(cosTheta_i);
        float sin2Theta_i = glm::max(0.0f, (1 - (cosTheta_i*cosTheta_i)));
        float sin2Theta_t = sin2Theta_i / (eta*eta);
        float cosTheta_t = sqrt(1 - sin2Theta_t);
        float Rparl = ((eta * cosTheta_i) - ( cosTheta_t)) /
                    ((eta * cosTheta_i) + ( cosTheta_t));
        float Rperp = ((cosTheta_i) - (eta * cosTheta_t)) /
                    (( cosTheta_i) + (eta * cosTheta_t));

        float frensel =  ((Rparl * Rparl + Rperp * Rperp) / 2);
        //Refract or reflect
      
        if(sin2Theta_t >= 1.0f || randomValue < frensel)//tir
        {
            pathSegment.ray.direction = glm::normalize(glm::reflect(incident_vector, normal));
            pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
            pathSegment.throughput *= m.color;

        }
        else
        {

            pathSegment.ray.direction = glm::normalize(eta * incident_vector + (eta * cosTheta_i - cosTheta_t) * -normal);
            pathSegment.ray.origin = intersect + EPSILON*pathSegment.ray.direction;  
            pathSegment.throughput *= m.color;

        }
    }
    else if(randomValue < m.hasReflective ) //reflect
    {   
        
        pathSegment.ray.direction = glm::normalize(glm::reflect(incident_vector, normal));
        pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
        pathSegment.throughput *= m.specular.color;
        pathSegment.throughput *=  glm::max(glm::dot(pathSegment.ray.direction, normal), 0.0f);
    }
    else //diffuse
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
        pathSegment.throughput *= m.color / PI;
        pathSegment.throughput *=  glm::max(glm::dot(pathSegment.ray.direction, normal), 0.0f);
    }
    pathSegment.remainingBounces--;
}
