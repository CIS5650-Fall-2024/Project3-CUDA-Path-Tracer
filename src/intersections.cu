#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
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
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
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
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


//From 4610
__host__ __device__ float triangleIntersectionTest(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, Ray r) {
    const float offset = 0.0000001;
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = p1 - p0;
    edge2 = p2 - p0;
    h = cross(r.direction, edge2);
    a = dot(edge1, h);
    if (a > -offset && a < offset) {
        return INFINITY;
    }
    f = 1.0 / a;
    s = r.origin - p0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return INFINITY;
    q = cross(s, edge1);
    v = f * dot(r.direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return INFINITY;
    }

    float t = f * dot(edge2, q);
    if (t > EPSILON) {
        return t;
    }
    else
        return -1;
}

__host__ __device__ glm::vec3 barycentric(glm::vec3 p, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3) {
    glm::vec3 edge1 = t2 - t1;
    glm::vec3 edge2 = t3 - t2;
    float S = length(cross(edge1, edge2));

    edge1 = p - t2;
    edge2 = p - t3;
    float S1 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t3;
    float S2 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t2;
    float S3 = length(cross(edge1, edge2));

    return glm::vec3(S1 / S, S2 / S, S3 / S);
}


//To detect if a ray intersects an AABB we use the slab test
__host__ __device__ bool IntersectAABB(Ray& r, AABB aabb, float t) {
    float tx1 = (aabb.min.x - r.origin.x) / r.direction.x;
    float tx2 = (aabb.max.x - r.origin.x) / r.direction.x;
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);

    float ty1 = (aabb.min.y - r.origin.y) / r.direction.y;
    float ty2 = (aabb.max.y - r.origin.y) / r.direction.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (aabb.min.z - r.origin.z) / r.direction.z;
    float tz2 = (aabb.max.z - r.origin.z) / r.direction.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    return tmax >= tmin && tmax > 0.0f && tmin < t;
}


__host__ __device__ void IntersectBVH(
    Ray& ray, float& t_min, int& hitTriIdx, const Triangle* triangles, const BVHNode* bvhNodes, const int* triIdx, int triIdxOffset, int triIdxEnd)
{
    int stack[64];
    int stackIdx = 0;

    stack[stackIdx++] = 0;  // Start with root node

    while (stackIdx > 0) {
        int nodeIdx = stack[--stackIdx];
        const BVHNode& node = bvhNodes[nodeIdx];

        if (!IntersectAABB(ray, node.aabb, t_min)) {
            continue;
        }

        if (node.isLeaf) {
            // Traverse the triangles in this leaf node, now adjusted to use the correct triIdx range
            for (int i = node.triIndexStart; i < node.triIndexEnd; ++i) {
                //int triangleIndex = triIdx[i + triIdxOffset];  
                int triangleIndex = i + triIdxOffset;  
                if (triangleIndex >= triIdxEnd) break; 
                const Triangle& tri = triangles[triangleIndex];
                float t = triangleIntersectionTest(tri.verts[0], tri.verts[1], tri.verts[2], ray);
                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    hitTriIdx = triangleIndex; 
                }
            }
        }
        else {
            if (node.left >= 0) {
                stack[stackIdx++] = node.left;
            }
            if (node.right >= 0) {
                stack[stackIdx++] = node.right;
            }
        }
    }
}

__host__ __device__ float meshIntersectionTestBVH(Geom mesh, Ray r, glm::vec3& intersectionPoint,
    glm::vec3& normal, bool& outside, const Triangle* triangles, BVHNode* bvhNodes, glm::vec2& uv, glm::vec3& tangent, glm::vec3& bitangent, const int* triIdx) {

    // Transform the ray into object space
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float t_min = INFINITY;
    glm::vec3 tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent;
    glm::vec2 tmp_uv;
    int hitTriIdx = -1;

    // Get the correct triangle index range for this mesh
    int triIdxOffset = mesh.triIndexStart;
    int triIdxEnd = mesh.triIndexEnd;

    // BVH intersection test with adjusted triangle index range
    IntersectBVH(rt, t_min, hitTriIdx, triangles, bvhNodes, triIdx, triIdxOffset, triIdxEnd);

    if (hitTriIdx != -1) {
        const Triangle& tri = triangles[hitTriIdx];
        tmp_intersect = getPointOnRay(rt, t_min);
        tmp_normal = glm::normalize(glm::cross(tri.verts[1] - tri.verts[0], tri.verts[2] - tri.verts[0]));
        glm::vec3 bary = barycentric(tmp_intersect, tri.verts[0], tri.verts[1], tri.verts[2]);
        tmp_uv = bary.x * tri.uvs[0] + bary.y * tri.uvs[1] + bary.z * tri.uvs[2];
        tmp_tangent = tri.tangent;
        tmp_bitangent = tri.bitangent;
    }
    else {
        return -1.0f;
    }

    if (t_min < INFINITY) {
        // Transform the intersection point back to world space
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(tmp_intersect, 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmp_normal, 0.0f)));
        tangent = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmp_tangent, 0.0f)));
        bitangent = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmp_bitangent, 0.0f)));
        uv = tmp_uv;

        return t_min;  // Return the closest intersection
    }

    return -1.0f; 
}


__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r, glm::vec3& intersectionPoint,
    glm::vec3& normal, bool& outside, const Triangle* triangles, glm::vec2& uv, glm::vec3& tangent, glm::vec3& bitangent) {
    // Transform the ray into object space
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float t_min = INFINITY;
    glm::vec3 tmp_intersect, tmp_normal, tmp_tangent, tmp_bitangent;
    glm::vec2 tmp_uv;

    // Iterate over the triangles in the mesh
    for (int i = mesh.triIndexStart; i < mesh.triIndexEnd; ++i) {
        const Triangle& tri = triangles[i];

        // Perfrom tri ray-triangle intersection for each triangle
        float t = triangleIntersectionTest(tri.verts[0], tri.verts[1], tri.verts[2], rt);
        // Update closest intersection
        if (t < t_min && t > 0.0f) {
            t_min = t;
            tmp_intersect = getPointOnRay(rt, t);
            tmp_normal = glm::normalize(glm::cross(tri.verts[1] - tri.verts[0], tri.verts[2] - tri.verts[0]));
            //check if this correct 
            glm::vec3 bary = barycentric(tmp_intersect, tri.verts[0], tri.verts[1], tri.verts[2]);
            tmp_uv = bary.x * tri.uvs[0] + bary.y * tri.uvs[1] + bary.z * tri.uvs[2];
            tmp_tangent = tri.tangent;
            tmp_bitangent = tri.bitangent;
        }
    }

    // If no intersection was found, transform the point and normal back to world space
    if (t_min < INFINITY) {
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(tmp_intersect, 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmp_normal, 0.0f)));
        tangent = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmp_tangent, 0.0f)));
        bitangent = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmp_bitangent, 0.0f)));
        uv = tmp_uv;

        return t_min;
    }

    // No intersection
    return -1.0f;
}