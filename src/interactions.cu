#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal,
	thrust::default_random_engine& rng)
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
	PathSegment& pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material& m,
	thrust::default_random_engine& rng)
{
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	// though intersection calculation will also normalize, still normalize here for now
	pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
	pathSegment.ray.origin = intersect;
	pathSegment.color *= m.color;

	//if is transparent, then use BTDF
	if (m.hasRefractive) {
		//refractive part
		//Schlick’s approximation
		float cos_theta = -glm::dot(pathSegment.ray.direction, normal);

		float n_i = 1.0f;
		float n_o = m.indexOfRefraction;

		float ratio = cos_theta < 0 ? n_o / n_i : n_i / n_o;
		float R_0 = glm::pow((n_i - n_o) / (n_i + n_o), 2);

		auto R_theta = R_0 + (1.0f - R_0) * glm::pow(1.0f - cos_theta, 5);

		thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
		float rand_R = u01(rng);

		glm::vec3 refract_direction_with_shilick;
		if (rand_R > R_theta) {
			refract_direction_with_shilick = glm::refract(pathSegment.ray.direction, cos_theta < 0 ? -normal : normal, ratio);
			//this is very very important!
			pathSegment.ray.origin -= normal * 0.001f;
		}
		else {
			refract_direction_with_shilick = glm::reflect(pathSegment.ray.direction, normal);
			pathSegment.ray.origin += normal * 0.001f;
		}
		pathSegment.ray.direction = refract_direction_with_shilick;
	}
	//if not transparent but specular then use BRDF
	else if (m.hasReflective) {
		auto reflect_direction = glm::reflect(pathSegment.ray.direction, normal);
		auto diffuse_direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.direction = glm::normalize(glm::mix(diffuse_direction, reflect_direction, m.hasReflective));
		pathSegment.ray.origin += normal * 0.001f;
	}
	//if neither reflective nor specular, fully diffuse
	else {
		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.origin += normal * 0.001f;
	}

	pathSegment.remainingBounces--;
}