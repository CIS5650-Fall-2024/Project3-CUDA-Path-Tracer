#include "Light.h"

__device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    if (f == 0.f && g == 0.f)
        return 0.f;
    float result = f * f / (f * f + g * g);
    return result;
}

__device__ float Pdf_Li(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, int num_Lights, const glm::vec3& view_point, const glm::vec3& normal, const glm::vec3& wiW,
    const int chosenLightIdx)
{
    Ray ray;
    ray.origin = view_point;
    ray.direction = wiW;

    if (chosenLightIdx < num_Lights) {
        ShadeableIntersection isect;
        isect.t = FLT_MAX;
        AreaLight light = areaLights[chosenLightIdx];
        //Intersect isect with this specific light
        bool hitLight = AllLightIntersectTest(isect, ray,
            triangles, bvhNodes,
            areaLights, num_Lights);

        if (!hitLight) {
            return 0;
        }

        glm::vec3 light_sample = ray.origin + isect.t * wiW;

        ShapeType type = light.shapeType;
        if (type == RECTANGLE) {
            glm::vec3 pos = glm::vec3(light.transform * glm::vec4(0, 0, 0, 1));
            glm::vec2 sideLen = { light.scale.x, light.scale.y };
            glm::vec3 nor = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(0, 0, 1, 0)));

            float r2 = isect.t * isect.t;

            return r2 / (abs(dot(wiW, nor)) * (4 * sideLen.x * sideLen.y));
        }
        else if (type == SPHERE) {

        }
    }

    return 0;
}

__device__ glm::vec3 Sample_Li(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, int num_Lights, const glm::vec3& view_point, const glm::vec3& normal, glm::vec3& wiW, float& pdf,
    int& chosenLightIdx, int& chosenLightID, LightType& chosenLightType,
    thrust::default_random_engine& rng)
{
    // Choose a random light 
    // Area light, point light, spot lights
    // For now, let's do JUST area lights

    int n = num_Lights; //TODO- pass this in
    thrust::uniform_real_distribution<float> u01(0, 1);
    int randomLightIdx = int(u01(rng) * n);
    chosenLightIdx = randomLightIdx;

    chosenLightID = areaLights[chosenLightIdx].ID; //useful for when there are other kinds of lights ... e.g. point light
    chosenLightType = AREALIGHT;
    return DirectSampleAreaLight(triangles, bvhNodes, areaLights, randomLightIdx, num_Lights, view_point, normal, wiW, pdf, rng);
}

__device__ glm::vec3 DirectSampleAreaLight(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, 
    int idx, int num_Lights, 
    const glm::vec3& view_point, const glm::vec3& normal, 
    glm::vec3& wiW, float& pdf,
    thrust::default_random_engine& rng)
{
    const AreaLight light = areaLights[idx];
    Ray shadowRay;

    if (light.shapeType == RECTANGLE) {

        glm::vec3 pos = glm::vec3(light.transform * glm::vec4(0, 0, 0, 1));
        glm::vec2 sideLen = { light.scale.x, light.scale.y };
        glm::vec3 nor = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(0, 0, 1, 0)));

        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec2 xi = { u01(rng), u01(rng) };
        xi = xi * 2.f - glm::vec2(1.); //[-1, 1]
        glm::vec3 tan, bit;
        coordinateSystem(nor, tan, bit);

        glm::vec3 sample = xi.x * tan * sideLen.x + xi.y * bit * sideLen.y + pos;

        shadowRay.origin = view_point;
        shadowRay.direction = normalize(sample - view_point);
        wiW = shadowRay.direction; //point to light!

        // Convert PDF from w/r/t surface area to w/r/t solid angle
        float r2 = dot(sample - view_point, sample - view_point);
         ////r*r / (cos(theta_w) * area)
        pdf = r2 / (abs(dot(shadowRay.direction, nor)) * (4 * sideLen.x * sideLen.y));

        //Intersect scene with shadowRay...
        ShadeableIntersection intr;
        intr.t = -1;
        intr.areaLightId = -1;
        bool hitLight = DirectLightIntersectTest(intr, shadowRay,
            triangles, bvhNodes,
            areaLights, num_Lights);

        if (hitLight) {
            return ((float)num_Lights) * areaLights[idx].Le * areaLights[idx].emittance;
        }
        else {
            return glm::vec3(0, 0, 0);
        }
    }
    else if (light.shapeType == SPHERE) {

    }
    wiW = glm::vec3(-1);
    pdf = 0;
    return glm::vec3(0,0,0);
}

__device__ glm::vec3 MISDirectLi(MeshTriangle* triangles, BVHNode* bvhNodes, AreaLight* areaLights, cudaTextureObject_t* texObjs,
    int num_Lights, 
    const glm::vec3& woWOut, const glm::vec3& view_point, const glm::vec3& normal, 
    const Material& m,
    const glm::vec3 texCol,
    const bool useTexCol,
    const bool BVHEmpty,
    thrust::default_random_engine& rng)
{
    //In MIS, we get both the Light sample and ALSO the BSDF sample!
/// LI SAMPLING
    float pdfL_L, pdfL_B;
    glm::vec3 wiL;

    int chosenLightIdx, chosenLightI;
    LightType chosenLightType;
    glm::vec3 LiL = Sample_Li(triangles, bvhNodes, areaLights, 1, view_point,
        normal,
        wiL, pdfL_L, chosenLightIdx, chosenLightI, chosenLightType, rng);

    if (pdfL_L == 0) {
        LiL = glm::vec3(1,0,1);
    }

    glm::vec3 fL;
    f(woWOut, wiL, pdfL_B, fL, normal, m, texCol, useTexCol, rng);
    float absDotL = abs(dot(wiL, normal));

/// BSDF SAMPLING
    // Compute Sample_f, Li, absdot for BSDF sampling
    glm::vec3 wiB;
    float pdfB_B, pdfB_L;
    glm::vec3 fB;

    PathSegment path;
    path.ray.origin = view_point;
    path.ray.direction = -woWOut;
    sample_f(path, woWOut, pdfB_B, fB, normal, m, texCol, useTexCol, rng);


    //TODO- implement PDF_LI
    pdfB_L = Pdf_Li(triangles, bvhNodes, areaLights, num_Lights, view_point, normal, wiB, chosenLightIdx);

    float absDotB = abs(dot(wiB, normal));

    ShadeableIntersection isect_Li_B;
    Ray r_B;
    r_B.direction = wiB;
    r_B.origin = view_point;
    SceneIntersect(isect_Li_B, r_B, triangles, bvhNodes, texObjs, areaLights, num_Lights, BVHEmpty);

    glm::vec3 LiB = (isect_Li_B.areaLightId != -1) ? areaLights[isect_Li_B.areaLightId].Le * areaLights[isect_Li_B.areaLightId].emittance : glm::vec3(0);

    if (pdfB_B == 0) {
        return glm::vec3(0, 0, 1);
    }

    //TODOLOG: ENVIRONMENT LIGHTING LOGIC
    glm::vec3 Lo = fL * LiL * absDotL / pdfL_L * PowerHeuristic(1, pdfL_L, 1, pdfL_B)
        + fB * LiB * absDotB / pdfB_B * PowerHeuristic(1, pdfB_B, 1, pdfB_L);
    Lo = glm::clamp(Lo, glm::vec3(0), glm::max(LiB, LiL));
    return Lo;
}