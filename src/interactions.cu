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

__host__ __device__ float fresnelDielectric(float cosThetaI, float IOR) {
    float etaI = 1.f;
    float etaT = IOR;
    etaT = etaT < EPSILON ? 1.55f : etaT;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    if (cosThetaI > 0.f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
    }
    cosThetaI = glm::abs(cosThetaI);

    // Computer cosThetaT using Snell's law
    float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1.0f) {
        return 1.f;
    }

    // Compute Fresnel reflectance using light polarization eqns, see PBRT 8.2.1
    float cosThetaT = glm::sqrt(glm::max(0.f,
        1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) * 0.5f; // coefficient
}

__host__ __device__ glm::vec3 sample_f_specular_reflection(
    glm::vec3 normal, glm::vec3 rayDir, glm::vec3 color, glm::vec3& wiW) {
    wiW = glm::reflect(rayDir, normal);
    return color;
}

__host__ __device__ glm::vec3 sample_f_specular_transmission(
	glm::vec3 normal, glm::vec3 rayDir, float IOR, glm::vec3 color, glm::vec3 &wiW) {
	float etaA = 1.f; // IOR of air
    float etaB = IOR; // IOR of material

    // Determine if we're entering or exiting the material
    bool entering = glm::dot(rayDir, normal) < 0.0f;
    float eta = entering ? 1.0f / IOR : IOR;

    normal = entering ? normal : -normal;
    wiW = glm::refract(rayDir, normal, eta);
    
	// Total internal reflection
    if (glm::length(wiW) < EPSILON) {
        return glm::vec3(0.0f);
    }
    return color;
}

__host__ __device__ glm::vec3 sample_f_glass(
    glm::vec3 normal, glm::vec3 rayDir, float IOR, glm::vec3 color,
    glm::vec3& wiW, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);
	float fresnel = fresnelDielectric(glm::dot(normal, rayDir), IOR);

    if (random < 0.5f) {
        // Reflection
		wiW = glm::reflect(rayDir, normal);
		return fresnel * color * 2.f;
    }
    else {
        // Refraction
        glm::vec3 T = sample_f_specular_transmission(normal, rayDir, IOR, color, wiW);
        float absDot = glm::abs(glm::dot(normal, wiW));
        if (absDot == 0.0f) {
            T = color;
        }
        else {
            T = color / absDot;
        }
        return 2.0f * T * (1.0f - fresnel);
    }
}

__host__ __device__ glm::vec3 sample_f_specular_plastic(
    glm::vec3 normal, glm::vec3 rayDir, Material,
    glm::vec3 &wiW, thrust::default_random_engine& rng) {

    wiW = glm::vec3(0, 1, 0);
	return glm::vec3(1.0f, 0.0f, 0.0f);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 wiW;
    glm::vec3 bsdf;
	float pdf;
	bool ignore_pdf = true;

    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.hasReflective && m.hasRefractive) {
		// Transparent and reflective material like glass
		bsdf = sample_f_glass(normal, pathSegment.ray.direction, m.indexOfRefraction, m.color, wiW, rng);
		pdf = 1.0f;
		ignore_pdf = false;
    }
    else if (m.hasPlastic) {
        if (m.roughness == 0.f) {
            // Reflective material that has color, like smooth plastic
            bsdf = sample_f_specular_plastic(normal, pathSegment.ray.direction, m, wiW, rng);
		}
		else {
			// Reflective material that has color, like rough plastic
			//bsdf = sample_f_ggx(normal, pathSegment.ray.direction, m, wiW, rng);
		}
		pdf = 1.0f;
		ignore_pdf = false;
    }
	else if (m.hasReflective) {
        // Acts like a mirror, no diffuse component
		bsdf = sample_f_specular_reflection(normal, pathSegment.ray.direction, m.color, wiW);
	}
	else if (m.hasRefractive) {
		// Transparent material that only transmits
		bsdf = sample_f_specular_transmission(normal, pathSegment.ray.direction, m.indexOfRefraction, m.color, wiW);
		//bsdf = glm::vec3(1.0f, 0.f, 0.f);
	}
	else {
		bsdf = m.color / PI;
		wiW = calculateRandomDirectionInHemisphere(normal, rng);
		pdf = glm::cos(glm::acos(glm::dot(wiW, normal))) / PI;
		ignore_pdf = false;
	}

	pathSegment.color *= ignore_pdf ? bsdf : bsdf * abs(glm::dot(wiW, normal)) / pdf;
	pathSegment.ray.direction = glm::normalize(wiW);
	pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
	pathSegment.remainingBounces--;
}
