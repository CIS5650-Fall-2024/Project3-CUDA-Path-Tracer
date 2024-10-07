#include "interactions.h"

#define TEST_DIFFUSE 0
#define USE_DIFFUSE_TEXTURE 1
#define USE_NORMAL_TEXTURE 1
#define TEST_NORMAL_MAP 0

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

__host__ __device__ glm::vec3 textureSample(const Texture* texture, glm::vec2 uv) {
    int x = (int)(uv.x * texture->width);
    int y = (int)((1.0f - uv.y) * texture->height);
    int index = 0;
    index = y * texture->width + x;
    glm::vec3 texColor = texture->data[index];
    return texColor;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 tangent,
    glm::vec3 bitangent,
    glm::vec2 uv,
    //glm::mat3 TBN,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Set the new ray origin and direction to the intersection point
	glm::vec3 newOrigin = intersect + EPSILON * normal;   
    glm::vec3 newDirection = normalize(calculateRandomDirectionInHemisphere(normal, rng));

    float pdf = 1.0;

#if !TEST_DIFFUSE
    // Diffuse shading (Lambertian reflection)
    if (!m.hasReflective && !m.hasRefractive && !m.hasDiffuseTexture) {
        // Simple diffuse scattering
        pathSegment.ray.direction = newDirection;

		float cosTheta = glm::abs(glm::dot(normal, newDirection));
		pdf = cosTheta / PI;

        // throughput
        if (pdf > EPSILON) {
            pathSegment.color *= m.color;
        }
    }
    else if (m.hasReflective) {
        // Reflective materials: Specular reflection
        newDirection = glm::reflect(pathSegment.ray.direction, normal);
		// calculate specular reflection color
        
        // Adjust the color for reflective materials using the specular component
        pathSegment.color *= m.color;
    }
    else if (m.hasRefractive) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rand = u01(rng);
        float cosTheta = glm::dot(normal, pathSegment.ray.direction);
        float eta = (cosTheta > 0) ? (m.indexOfRefraction / 1.0f) : (1.0f / m.indexOfRefraction);
        glm::vec3 refractDirection = glm::normalize( glm::refract(pathSegment.ray.direction, normal, eta));

        // Adjust normal direction and cosTheta for refraction calculations
        if (cosTheta < 0) {
            cosTheta = -cosTheta; //entering the medium
        }
        else {
            normal = -normal; // Flip the normal
        }

        // Calculate Fresnel reflectance using Schlick's approximation
        float R0 = pow((1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction), 2);
        float reflectance = R0 + (1 - R0) * pow(1 - cosTheta, 5);

        // Check if the refraction results in total internal reflection
        if (glm::length(refractDirection) == 0) {
            reflectance = 1.0; 
        }

        if (rand < reflectance) {
            // Reflect
            newDirection = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            // Refract
            newDirection = refractDirection;
        }
		newOrigin = intersect - EPSILON * normal;
        pathSegment.color *= m.color;      
    }

    // texture
#if USE_DIFFUSE_TEXTURE
    if (m.hasDiffuseTexture) {
        glm::vec3 textureColor = textureSample(m.diffuseTexture, uv);
        // gamma correction to texture color
        textureColor = glm::pow(textureColor, glm::vec3(1.0f / 2.2f));
        pathSegment.color *= textureColor;
    }
#endif
#if USE_NORMAL_TEXTURE
    if (m.hasNormalTexture) {
        glm::vec3 sampledNormal = textureSample(m.normalTexture, uv);
        sampledNormal = 2.0f * sampledNormal - glm::vec3(1.0f); // [0,1] to [-1, 1]
		glm::mat3 TBN = glm::mat3(tangent, bitangent, normal);
        normal = normalize(TBN * sampledNormal);
        newOrigin = intersect;
        newDirection = normalize(calculateRandomDirectionInHemisphere(normal, rng));
#if TEST_NORMAL_MAP
        pathSegment.color = normal;
#endif
    }
#endif

#else   
    //test diffuse
	

	float cosTheta = glm::dot(normal, newDirection);
	pdf = cosTheta * INV_PI;

	// Multiply the path color by the material color (throughput)
	if (pdf > EPSILON) {
		pathSegment.color *= m.color; // why not divide by PI?
	}
    else {
		pathSegment.color = glm::vec3(0.0f);
        return;
    }

#endif

	// UPDATE the path segment with the new ray origin and direction
    pathSegment.ray.origin = newOrigin;
    pathSegment.ray.direction = newDirection;
    pathSegment.remainingBounces--;
}

__host__ __device__ void getEnvironmentMapColor(
    PathSegment& pathSegment,
    const Texture& enviromentMap,
    thrust::default_random_engine& rng) {

	glm::vec3 rayDir = -pathSegment.ray.direction;
    rayDir = glm::normalize(rayDir);
    float u = 0.5f + (atan2(rayDir.z, rayDir.x) / (2.0f * PI));
    float v = 0.5f - (asin(rayDir.y) / PI);
    glm::vec2 uv(u, v);

	// sample the environment map
	glm::vec3 color = textureSample(&enviromentMap, uv);
	pathSegment.color *= color;

}
