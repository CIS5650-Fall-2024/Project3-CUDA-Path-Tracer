#include "interactions.h"
//#include "pbr.h"
#include "disneybsdf.h"
#include "light.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

	// The random generated direction is cosine weighted by sqrt the random number
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

	// the final direction is a combination of a linear combination of the two perpendicular directions and the normal
    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}



__device__ void scatterRay(
    PathSegment& pathSegment,
	const ShadeableIntersection& intersection,
    const glm::vec3& intersect,
    const Material &m,
    thrust::default_random_engine &rng,
    int num_lights,
    LinearBVHNode* dev_nodes,
    Triangle* dev_triangles,
    Light* dev_lights,
    cudaTextureObject_t envMap)
{
	glm::vec3 normal = intersection.surfaceNormal;
	glm::vec2 uv = intersection.uv;

    glm::vec3 wo = -pathSegment.ray.direction;
	glm::vec3 wi = glm::vec3(0.0f);
	glm::vec3 col = glm::vec3(1.0f);
	Material mat = m;

    // TODO: implement PBR model
    thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
	glm::mat3 ltw = LocalToWorld(normal);
	glm::mat3 wtl = glm::transpose(ltw);
    
	float pdf = 0.f;
	
	glm::vec3 bsdf = Sample_disneyBSDF(m, wo, xi, wi, ltw, wtl, pdf, rng);
	if (pdf <= 0) {
		pathSegment.remainingBounces = 0;
		return;
	}
	col = bsdf * AbsDot(wi, normal) / pdf;
    pathSegment.remainingBounces--;

#ifdef DEBUG_NORMAL
    col = glm::vec3(1.f);
    pathSegment.color = DEBUG_NORMAL ? (normal + 1.0f) / 2.0f : normal;
	pathSegment.remainingBounces = 0;
#elif defined(DEBUG_WORLD_POS)
	col = glm::vec3(1.f);
    pathSegment.color = glm::clamp(intersect, glm::vec3(0), glm::vec3(1.0f));
	pathSegment.remainingBounces = 0;
#elif defined(DEBUG_UV)
	col = glm::vec3(1.f);
	pathSegment.color = glm::vec3(uv, 0);
	pathSegment.remainingBounces = 0;
#endif
    //pathSegment.color = glm::vec3(col);
    //col = glm::vec3(1.0);
    //pathSegment.remainingBounces = 0;

	pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.throughput *= col;
}

__device__ void MIS(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    const glm::vec3& intersect,
    const Material& m,
    thrust::default_random_engine& rng,
    int num_lights,
    LinearBVHNode* dev_nodes,
    Triangle* dev_triangles,
    Light* dev_lights,
    cudaTextureObject_t envMap,
    int depth)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 normal = intersection.surfaceNormal;
    glm::vec2 uv = intersection.uv;

    glm::vec3 wo = -pathSegment.ray.direction;
    glm::vec3 wi = glm::vec3(0.0f);
    glm::vec3 col = glm::vec3(1.0f);
    Material mat = m;

    //normal = glm::dot(normal, wo) > 0 ? normal : -normal;
    // disney bsdf
    glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
    glm::mat3 ltw = LocalToWorld(normal);
    glm::mat3 wtl = glm::transpose(ltw);

    glm::vec3 wi_disney = glm::vec3(0.f);
    float pdf_disney = 0.f;

	glm::vec3 wol(wtl * wo);
    bool isRefract(false), isReflect(false), isInternal(glm::dot(normal, wo) < 0);
	BSDF_setUp(m, wi_disney, wol, rng, isRefract, isReflect);

    //wi_disney = calculateRandomDirectionInHemisphere(normal, rng);
    glm::vec3 Li_disney = Evaluate_disneyBSDF(m, wi_disney, wol, pdf_disney, isRefract, isReflect);


	if (pdf_disney <= 1e-6) {
		pathSegment.remainingBounces = 0;
		return;
	}
    

    glm::vec3 currAccum = pathSegment.accumLight;

    if (num_lights > 0)
    {
        // direct lighting
        int light_id = 0;
        float pdf_direct = 0.f;
        glm::vec3 wi_direct = glm::vec3(0.f);
        glm::vec3 Li_direct = Sample_Li(intersect, normal, wi_direct, pdf_direct, intersection.directLightId, num_lights, envMap, rng, dev_nodes, dev_triangles, dev_lights, ltw, wtl);

        //MIS
        float pdf_disney_for_direct = 0;
        float pdf_direct_for_disney = 0;
        glm::vec3 Li_disney_for_direct = glm::vec3(0.f);
        glm::vec3 Li_direct_for_disney = glm::vec3(0.f);

        Li_direct_for_disney = Evaluate_Li(wi_disney, intersect, pdf_direct_for_disney, intersection.directLightId, num_lights, envMap, dev_nodes, dev_triangles, dev_lights);
        Li_disney_for_direct = Evaluate_disneyBSDF(m, wi_direct, wo, pdf_disney_for_direct, false, false);

        //float weight_disney = PowerHeuristic(1, pdf_disney, 1, pdf_direct_for_disney);
        float weight_direct = PowerHeuristic(1, pdf_direct, 1, pdf_disney_for_direct);

        if (pdf_direct > 0)
        {
            currAccum += pathSegment.throughput * Li_direct * Li_disney_for_direct * HemisphereDot(wi_direct, normal) / pdf_direct * weight_direct;
        }



    }
    pathSegment.remainingBounces--;


    glm::vec3 offset = normal * (isInternal ? 1e-3f : -(1e-3f));
   

    //pathSegment.accumLight = currAccum;
	wi = glm::normalize(ltw * wi_disney);
    pathSegment.throughput *= Li_disney * AbsCosTheta(wi_disney) / pdf_disney;
    pathSegment.ray.origin = isRefract ? pathSegment.ray.origin + pathSegment.ray.direction * intersection.t + offset: intersect;
    pathSegment.ray.direction = glm::normalize(wi);

    // russian roulette
    float isSurvive = u01(rng);
    if (isSurvive > glm::max(0.1f, 1.f - dot(currAccum, { 0.2126, 0.7152, 0.0722 }) / 0.7f))
    {
        pathSegment.remainingBounces = 0;
        return;
    }

}