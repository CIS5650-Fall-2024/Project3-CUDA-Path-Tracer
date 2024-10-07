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

#if 1
__host__ __device__ float FresnelDielectricEval(float cosThetaI, float etaI, float etaT, bool outside) {

    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
    bool entering = cosThetaI > 0.f;

    if (!entering) {
    //if (!outside) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
    }
    cosThetaI = abs(cosThetaI);

    float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

__host__ __device__ glm::vec3 refraction(const glm::vec3& uv, const glm::vec3& normal, float eta) {
    auto cosTheta = glm::min(glm::dot(-uv, normal), 1.0f);
    glm::vec3 r_out_perp = eta * (uv + cosTheta * normal);
    glm::vec3 r_out_parallel = normal * -glm::sqrt(glm::abs(1.0f - (glm::length(r_out_perp) * glm::length(r_out_perp))));
    return r_out_perp + r_out_parallel;
}

//DIFFUSE_REFL
__host__ __device__ glm::vec3 Sample_f_diffuse(
    glm::vec3& wi,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    wi = calculateRandomDirectionInHemisphere(normal, rng);
    return m.color;
}

//DIFFUSE_TRAN
__host__ __device__ glm::vec3 Sample_f_diffuse_trans(
    glm::vec3& wi,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    float& pdf) {
    wi = calculateRandomDirectionInHemisphere(normal, rng);
    wi.z = -wi.z;
    float absDot = glm::abs(glm::dot(wi, normal));
    pdf = absDot == 0 ? 0 : absDot * INV_PI;
    return m.color * INV_PI;
}

//SPEC_REFL
//pdf = 1.0f;
__host__ __device__ glm::vec3 Sample_f_specular_refl(
    glm::vec3& wi,
    glm::vec3& wo,
    glm::vec3 normal,
    const Material& m) {
    wi = glm::reflect(wo, normal);
    return m.specular.color;
}

//SPEC_TRAN
//pdf = 1.0f;
__host__ __device__ glm::vec3 Sample_f_specular_trans(
    glm::vec3& wi,
    glm::vec3& wo,
    glm::vec3 normal,
    const Material& m,
    bool outside) {

    float etaI = 1.0f;
    float etaT = m.indexOfRefraction; // 1.55f
    float cosThetaI = glm::clamp(glm::dot(wo, normal), -1.f, 1.f);
    
    glm::vec3 bsdf = glm::vec3(0.f);
    float eta;
    if (cosThetaI < 0) {
    //if (!outside){
        eta = etaI / etaT;
    }
    else {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        eta = etaI / etaT;
        normal = -normal;
    }
    float cos_theta = glm::min(glm::dot(wo, normal), 1.0f);
    float sin_theta = glm::sqrt(glm::max(0.f, 1.f - cos_theta * cos_theta));

    //wi = glm::refract(wo, normal, eta);
    wi = refraction(wo, normal, eta);

    if (eta * sin_theta > 1.0f) {
        wi = glm::reflect(wo, normal);
        return m.color;
    }

    float absDot = glm::abs(glm::dot(wi, normal));
    if (absDot == 0.0f) {
        bsdf = m.specular.color;
    }
    else {
        bsdf = m.specular.color / absDot;
    }
    return bsdf;
}

//SPEC_GLASS
__host__ __device__ glm::vec3 Sample_f_glass(
    glm::vec3& wi,
    glm::vec3& wo,
    float& pdf,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    bool outside,
    glm::vec3& offset) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 bsdf = glm::vec3(0.f);
    if (u01(rng) < 0.5f) {
        wi = glm::reflect(wo, normal);
        float absDot = glm::abs(glm::dot(wi, normal));
        if (absDot == 0) {
            bsdf = m.color;
        }
        else {
            bsdf = m.color / absDot;
        }
        float fresnel = FresnelDielectricEval(glm::dot(wi, normal), 1.0f, m.indexOfRefraction, outside);
        bsdf *= fresnel;
        pdf = 1.0f;
        offset = 0.001f * normal;
    }
    else {
        bsdf = Sample_f_specular_trans(wi, wo, normal, m, outside);

        float fresnel = FresnelDielectricEval(glm::dot(wi, normal), 1.0f, m.indexOfRefraction, outside);
        pdf = 1.0f;
        bsdf *= (1.0f - fresnel);
        offset = -0.001f * normal;
    }
    return bsdf *= 2.0f;
}


#endif


# if 0
// Faster Schlick Fresnel approximation than FresnelDielectricEval
__host__ __device__ float schlickFresnel(const glm::vec3& V, const glm::vec3& N, const float ior)
{
    float cosTheta = abs(glm::dot(V, N));
    float R0 = (1 - ior) / (1 + ior);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1.f - cosTheta, 5.0f);
}
# endif

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    bool outside,
    glm::vec3& texCol, bool hasTexture)
{
    glm::vec3 wi(0.f);
    glm::vec3 bsdf(0.f);
    float pdf = 1.0f;
    thrust::uniform_real_distribution<float> u01(0, 1);
    if (m.hasReflective > 0.0f && m.hasRefractive > 0.0f) {
        glm::vec3 offset;
        bsdf = Sample_f_glass(wi, pathSegment.ray.direction, pdf, normal, m, rng, outside, offset);
        float absDot = glm::abs(glm::dot(wi, normal));
        pathSegment.color *= bsdf * absDot / pdf;
        pathSegment.ray.origin = intersect + offset;
    }
    else if (m.hasRefractive > 0.0f) {
        bsdf = Sample_f_specular_trans(wi, pathSegment.ray.direction, normal, m, outside);
        pathSegment.color *= bsdf;
        pathSegment.ray.origin = intersect - 0.001f * normal;
    }
    else if (m.hasReflective > 0.0f) {
        bsdf = Sample_f_specular_refl(wi, pathSegment.ray.direction, normal, m);
        pathSegment.color *= bsdf;
        pathSegment.ray.origin = intersect + 0.001f * normal;
    }
    else {
        bsdf = Sample_f_diffuse(wi, normal, m, rng);
        if(hasTexture) {
			pathSegment.color *= texCol;
		}
		else {
			pathSegment.color *= bsdf;
		}
        pathSegment.ray.origin = intersect;
    }

    // Update ray direction and origin
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.remainingBounces--;
}
