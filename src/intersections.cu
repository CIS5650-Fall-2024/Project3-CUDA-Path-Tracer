#include "intersections.h"

__device__ void computeBarycentricWeights(const glm::vec3& p, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, glm::vec3& weights)
{
    // Calculate vectors

    glm::vec3 edgeAB = B - A;
    glm::vec3 edgeAC = C - A;
    glm::vec3 edgeAP = p - A;

    //Total Area (* 2)
    float S = length(cross(edgeAB, edgeAC));

    // Calculate areas of sub-triangles

    float areaPBC = length(cross(B - p,
        C - p));

    float areaPCA = length(cross(C - p, A - p));

    float areaPAB = length(cross(A - p, B - p));

    // Calculate barycentric coordinates
    weights[0] = areaPBC / S;
    weights[1] = areaPCA / S;
    weights[2] = areaPAB / S;
}

__host__ __device__ float triangleIntersectionTest(
    Ray r,
    const MeshTriangle& tri,
    glm::vec3& intersectionPoint,
    glm::vec3& normal)
{
    float t = -1;

    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;

    normal = glm::normalize(glm::cross(edge1, edge2));

    // Compute the det
    glm::vec3 pvec = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, pvec);

    // If determinant is near zero, ray is parallel
    if (fabs(det) < EPSILON) return -1;

    float invDet = 1.0f / det;

    glm::vec3 tvec = r.origin - tri.v0;

    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return -1;

    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(r.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1;

    // Calculate t, the intersection distance along the ray
    t = glm::dot(edge2, qvec) * invDet;

    // If t is negative, the intersection is behind the ray origin (no hit)
    if (t < EPSILON) return -1;

    intersectionPoint = getPointOnRay(r, t);
     
    return t;
}

__host__ __device__ float rectangleIntersectionTest(
    AreaLight light,
    Ray r,
    float radiusU,
    float radiusV,
    const glm::vec3& pos,
    const glm::vec3& normal,
    glm::vec2& UV)
{
    Ray q;
    q.origin = multiplyMV(light.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(light.inverseTransform, glm::vec4(r.direction, 0.0f)));
    float dt = dot(normal, q.direction);

    //Back face culling for lights
    if (dt > 0) return -1;
    glm::vec3 p = glm::vec3(0, 0, 0);

    float t = dot(normal, p - q.origin) / dt;
    if (t < 0.0) return -1;

    glm::vec3 hit = q.origin + q.direction * t;

    if (abs(hit.x) <= radiusU && abs(hit.y) <= radiusV) {
        glm::vec3 intersectionPoint = multiplyMV(light.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
        return glm::length(r.origin - intersectionPoint);
    }
    else {
        return -1;
    }
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = glm::min(t1, t2);
        outside = true;
    }
    else
    {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
        return glm::length(r.origin - intersectionPoint) + 0.001f;
    }
    return glm::length(r.origin - intersectionPoint) - 0.001f;
}

__device__ bool intersectAABB(const Ray& r, const AABB& aabb) {
    float tMin = (aabb.min.x - r.origin.x) / r.direction.x;
    float tMax = (aabb.max.x - r.origin.x) / r.direction.x;

    if (tMin > tMax) {
        // Swap if tMin > tMax
        float temp = tMin;
        tMin = tMax;
        tMax = temp;
    }

    float tyMin = (aabb.min.y - r.origin.y) / r.direction.y;
    float tyMax = (aabb.max.y - r.origin.y) / r.direction.y;

    if (tyMin > tyMax) {
        // Swap if tyMin > tyMax
        float temp = tyMin;
        tyMin = tyMax;
        tyMax = temp;
    }

    if ((tMin > tyMax) || (tyMin > tMax)) {
        // No intersection
        return false;
    }

    // Calculate the intersection intervals
    if (tyMin > tMin) {
        tMin = tyMin;  // Update tMin
    }
    if (tyMax < tMax) {
        tMax = tyMax;  // Update tMax
    }

    // Repeat for the Z axis
    float tzMin = (aabb.min.z - r.origin.z) / r.direction.z;
    float tzMax = (aabb.max.z - r.origin.z) / r.direction.z;

    if (tzMin > tzMax) {
        // Swap if tzMin > tzMax
        float temp = tzMin;
        tzMin = tzMax;
        tzMax = temp;
    }

    // Check for intersection
    if ((tMin > tzMax) || (tzMin > tMax)) {
        // No intersection
        return false;
    }

    return true;  // The ray intersects the bounding box
}

__device__ bool AreaLightIntersect(ShadeableIntersection& intr, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    AreaLight* areaLights,
    int areaLightIdx)
{
    float t_min = FLT_MAX;

    int shapeType = areaLights[areaLightIdx].shapeType;
    if (shapeType == RECTANGLE) {
        glm::vec3 pos = glm::vec3(0, 0, 0);
        glm::vec3 nor = glm::vec3(0, 0, 1);
        glm::vec2 halfSideLengths = glm::vec2(0.5, 0.5);
        glm::vec2 uv;

        float tmp_light_t = rectangleIntersectionTest(
            areaLights[areaLightIdx], r,
            halfSideLengths.x,
            halfSideLengths.y, pos, nor, uv);

        if (tmp_light_t > 0 && tmp_light_t < t_min) {
            t_min = tmp_light_t;
            intr.t = t_min;
            intr.areaLightId = areaLightIdx;
            intr.surfaceNormal = glm::normalize(multiplyMV(areaLights[areaLightIdx].invTranspose, glm::vec4(0, 0, 1, 0.0f)));
            return true;
        }
    }
    return false;
}

__device__ bool AllLightIntersectTest(ShadeableIntersection& intersection, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    AreaLight* areaLights,
    int num_areaLights)
{
    float t_min = intersection.t;
    if (t_min == -1) {
        t_min = FLT_MAX;
    }
    float t = -1;
    int areaLightId = -1;
    glm::vec3 surfaceNormal = glm::vec3(-1, -1, -1);

    for (int j = 0; j < num_areaLights; j++) {
        int shapeType = areaLights[j].shapeType;
        if (shapeType == RECTANGLE) {
            glm::vec3 pos = glm::vec3(0, 0, 0);
            glm::vec3 nor = glm::vec3(0, 0, 1);
            glm::vec2 halfSideLengths = glm::vec2(0.5, 0.5);
            glm::vec2 uv;

            float tmp_light_t = rectangleIntersectionTest(
                areaLights[j], r,
                halfSideLengths.x,
                halfSideLengths.y, pos, nor, uv);

            if (tmp_light_t > 0 && tmp_light_t < t_min) {
//HIT CASE!!!
                t_min = tmp_light_t;
                t = t_min;
                areaLightId = j;
                surfaceNormal = glm::normalize(multiplyMV(areaLights[j].invTranspose, glm::vec4(0,0,1, 0.0f)));
            }
        }
    }

    if (areaLightId != -1) {
//ACTUAL UPDATE!!!
        intersection.t = t;
        intersection.areaLightId = areaLightId;
        intersection.surfaceNormal = surfaceNormal;
        return true;
    }
    return false;
}

///  Returns true if there is ray intersects with a light source. Currently, only rectangular area lights are supported. INCLUDES BVH
__device__ bool DirectLightIntersectTest(ShadeableIntersection& intersection, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    AreaLight* areaLights,
    int num_areaLights)
{
    intersection.t = -1;
    intersection.areaLightId = -1;
    float t;
    float t_min = FLT_MAX;
    int matId = 0;

    int stack[16];
    int stackPtr = 0;
    stack[stackPtr] = 0;
    stackPtr++;

    //1. Lets find the closest triangle that we hit
    while (stackPtr > 0) {
        if (stackPtr >= 16) {
            // Stack overflow, exit traversal
            return;
        }

        int nodeIdx = stack[--stackPtr];

        if (nodeIdx < 0) {
            continue;
        }

        const BVHNode& node = bvhNodes[nodeIdx];

        //IF LEAF
        if (node.triangleIDs.x != -1) {
            for (int j = 0; j < 4; j++) {
                int tri_idx = node.triangleIDs[j];
                if (tri_idx != -1) {
                    const MeshTriangle& tri = triangles[tri_idx];
                    glm::vec3 tmp_intersect;
                    glm::vec3 tmp_normal;
                    glm::vec3 tmp_texCol = glm::vec3(-1, -1, -1);
                    int tmp_matId = tri.materialIndex;

                    t = triangleIntersectionTest(r, tri, tmp_intersect, tmp_normal);

                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                    }
                }
                else {
                    break;
                }
            }
        }
        else {
            //IF NOT LEAF
            int leftIdx = node.leftChild;
            int rightIdx = node.rightChild;

            bool hitLeft = intersectAABB(r, bvhNodes[leftIdx].bounds);
            bool hitRight = intersectAABB(r, bvhNodes[rightIdx].bounds);

            if (hitLeft) stack[stackPtr++] = leftIdx;
            if (hitRight) stack[stackPtr++] = rightIdx;
        }
    }

    //2. Next, we'll check if we hit a light thats closer
    ShadeableIntersection intr_light;
    intr_light.t = FLT_MAX;
    bool hitLight = AllLightIntersectTest(intr_light, r,
        triangles, bvhNodes,
        areaLights, num_areaLights);

    if (hitLight && intr_light.t < t_min) {
        intersection.t = intr_light.t;
        intersection.areaLightId = intr_light.areaLightId;
        intersection.surfaceNormal = intr_light.surfaceNormal;
        intersection.texCol = glm::vec3(-1);
        return true;
    }
    return false;
}

__device__ void BVHIntersect(Ray r, ShadeableIntersection& intersection,
    MeshTriangle* triangles, BVHNode* bvhNodes, cudaTextureObject_t* texObjs)
{
    float t;
    float t_min = FLT_MAX;
    bool hitTri = false;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    glm::vec3 texCol;
    int matId = 0;

    int stack[16];
    int stackPtr = 0;
    stack[stackPtr] = 0;
    stackPtr++;

    while (stackPtr > 0) {
        if (stackPtr >= 16) {
            // Stack overflow, exit traversal
            return;
        }

        int nodeIdx = stack[--stackPtr];

        if (nodeIdx < 0) {
            continue;
        }

        const BVHNode& node = bvhNodes[nodeIdx];

        //IF LEAF
        if (node.triangleIDs.x != -1) {
            for (int j = 0; j < 4; j++) {
                int tri_idx = node.triangleIDs[j];
                if (tri_idx != -1) {
                    const MeshTriangle& tri = triangles[tri_idx];
                    glm::vec3 tmp_intersect;
                    glm::vec3 tmp_normal;
                    glm::vec3 tmp_texCol = glm::vec3(-1, -1, -1);
                    int tmp_matId = tri.materialIndex;

                    t = triangleIntersectionTest(r, tri, tmp_intersect, tmp_normal);

                    if (t > 0.0f && t_min > t)
                    {
                        hitTri = true;
                        t_min = t;
                        intersect_point = tmp_intersect;
                        matId = tmp_matId;

                        if (tri.baseColorTexID != -1) {
                            cudaTextureObject_t texObj = texObjs[tri.baseColorTexID];
                            glm::vec2 UV = glm::vec2(0.5f, 0.5f);

                            glm::vec3 weights;
                            computeBarycentricWeights(intersect_point, tri.v0,
                                tri.v1,
                                tri.v2,
                                weights);

                            UV = weights.x * tri.uv0 +
                                weights.y * tri.uv1 +
                                weights.z * tri.uv2;
                            bool isInt = true;
                            if (isInt) {
                                int4 texColor_flt = tex2D<int4>(texObj, UV.x, UV.y);
                                tmp_texCol = glm::vec3(texColor_flt.x / 255.f, texColor_flt.y / 255.f, texColor_flt.z / 255.f);
                            }
                            else {
                                float4 texColor_flt = tex2D<float4>(texObj, UV.x, UV.y);
                                tmp_texCol = glm::vec3(texColor_flt.x, texColor_flt.y, texColor_flt.z);
                            }
                            tmp_texCol = glm::max(tmp_texCol, glm::vec3(EPSILON));
                        }

                        if (tri.normalMapTexID != -1) {
                            cudaTextureObject_t texObj = texObjs[tri.normalMapTexID];
                            glm::vec2 UV = glm::vec2(0.5f, 0.5f);

                            glm::vec3 weights;
                            computeBarycentricWeights(intersect_point, tri.v0,
                                tri.v1,
                                tri.v2,
                                weights);

                            UV = weights.x * tri.uv0 +
                                weights.y * tri.uv1 +
                                weights.z * tri.uv2;
                            bool isInt = true;
                            if (isInt) {
                                int4 normalEncoded = tex2D<int4>(texObj, UV.x, UV.y);
                                tmp_normal = glm::vec3(normalEncoded.x / 255.f, normalEncoded.y / 255.f, normalEncoded.z / 255.f);
                                tmp_normal = (tmp_normal * 2.f) - glm::vec3(1.f);
                            }
                            else {
                                float4 normalEncoded = tex2D<float4>(texObj, UV.x, UV.y);
                                tmp_normal = glm::vec3(normalEncoded.x, normalEncoded.y, normalEncoded.z);
                                tmp_normal = (tmp_normal * 2.f) - glm::vec3(1.f);
                            }
                            tmp_normal = normalize(tmp_normal); //IMPORTANT
                        }
                        //This code here is for both textured and NON textured!
                        texCol = tmp_texCol;
                        normal = tmp_normal;
                    }
                }
                else {
                    break;
                }
            }
        }
        else {
            //IF NOT LEAF
            int leftIdx = node.leftChild;
            int rightIdx = node.rightChild;

            bool hitLeft = intersectAABB(r, bvhNodes[leftIdx].bounds);
            bool hitRight = intersectAABB(r, bvhNodes[rightIdx].bounds);

            if (hitLeft) stack[stackPtr++] = leftIdx;
            if (hitRight) stack[stackPtr++] = rightIdx;
        }
    }
    if (!hitTri)
    {
        intersection.t = -1.0f;
        intersection.materialId = -1;
        intersection.texCol = glm::vec3(-1);
        intersection.areaLightId = -1;
        intersection.surfaceNormal = glm::vec3(1, 0, 0);
    }
    else
    {
        intersection.t = t_min;
        intersection.materialId = matId;
        intersection.areaLightId = -1;
        intersection.surfaceNormal = normal;
        intersection.texCol = texCol;
    }
}

__device__ void SceneIntersect(ShadeableIntersection& isect, Ray r,
    MeshTriangle* triangles, BVHNode* bvhNodes,
    cudaTextureObject_t* texObjs,
    AreaLight* areaLights,
    int num_areaLights,
    bool BVHEmpty)
{
    isect.t = -1;
    isect.areaLightId = -1;
    isect.materialId = -1;
    isect.surfaceNormal = glm::vec3(1,0,0);
    isect.texCol = glm::vec3(-1);

    float t_min = FLT_MAX;
    if (!BVHEmpty) {
        BVHIntersect(r, isect, triangles, bvhNodes, texObjs);
        if (isect.t != -1) {
            t_min = isect.t;
        }
    }
    // 2. Lights

    ShadeableIntersection intr_light;
    intr_light.t = FLT_MAX;
    bool hitLight = AllLightIntersectTest(intr_light, r,
        triangles, bvhNodes,
        areaLights, num_areaLights);

    if (hitLight && intr_light.t > 0 && intr_light.t < t_min) {
        isect.t = intr_light.t;
        isect.areaLightId = intr_light.areaLightId;
        isect.surfaceNormal = intr_light.surfaceNormal;
        isect.texCol = glm::vec3(-1);
    }
}