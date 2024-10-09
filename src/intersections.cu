#include "intersections.h"

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
        //normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    Geom tri,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{

    Ray q;
    q.origin = multiplyMV(tri.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(tri.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 v0 = tri.triData.verts[0];
    glm::vec3 v1 = tri.triData.verts[1];
    glm::vec3 v2 = tri.triData.verts[2];

    glm::vec3 n0 = tri.triData.normals[0];
    glm::vec3 n1 = tri.triData.normals[1];
    glm::vec3 n2 = tri.triData.normals[2];

    glm::vec2 uv0 = tri.triData.uvs[0];
    glm::vec2 uv1 = tri.triData.uvs[1];
    glm::vec2 uv2 = tri.triData.uvs[2];

    glm::vec3 baryCoords;
    bool intersects = glm::intersectRayTriangle(q.origin, q.direction, v0, v1, v2, baryCoords);

    if (intersects)
    {
        intersectionPoint = multiplyMV(tri.transform, glm::vec4(getPointOnRay(q, baryCoords.z), 1.0f));
        glm::vec3 interpolatedNormal = (1.f - baryCoords.x - baryCoords.y) * n0 + baryCoords.x * n1 + baryCoords.y * n2;
        normal = glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(interpolatedNormal, 0.0f)));
        
        uv = (1.f - baryCoords.x - baryCoords.y) * uv0 + baryCoords.x * uv1 + baryCoords.y * uv2;

        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ bool AABBIntersectionTest(
    BVHNode node,
    Ray r) 
{
    //printf("Ray direction (%f, %f, %f)\n", r.direction.x, r.direction.y, r.direction.z);
    glm::vec3 mins = node.mins;
    glm::vec3 maxs = node.maxs;
    //printf("Mins (%f, %f, %f)\n", mins.x, mins.y, mins.z);

    float t1 = (mins[0] - r.origin[0]) / r.direction[0];
    float t2 = (maxs[0] - r.origin[0]) / r.direction[0];

    float tmin = fminf(t1, t2);
    float tmax = fmaxf(t1, t2);

    for (int axis = 1; axis < 3; ++axis)
    {
        //parametric distances where min/max plane intersects the axis
        t1 = (mins[axis] - r.origin[axis]) / r.direction[axis];
        t2 = (maxs[axis] - r.origin[axis]) / r.direction[axis];
        //update tmin & tmax
        tmin = fmaxf(tmin, fminf(t1, t2));
        tmax = fminf(tmax, fmaxf(t1, t2));
    }

    tmin = fmaxf(tmin, 0.f);
    return tmax > tmin;
}