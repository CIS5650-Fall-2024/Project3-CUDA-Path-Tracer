#include "intersections.h"

__host__ __device__ float bboxIntersectionTest(
    BBox bbox,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    glm::vec3 &times)
{
    glm::vec3 o = r.origin;
    glm::vec3 d = r.direction;

    glm::vec3 invdir = glm::vec3(1.f / r.direction.x, 1.f / r.direction.y, 1.f / r.direction.z);

    float t0x, t0y, t0z, t1x, t1y, t1z;

    glm::vec3 min = bbox.min;
    glm::vec3 max = bbox.max;

    // check if 2D rays are outside bbox
    if(invdir.x >= 0) {
        t0x = (min.x - o.x) * invdir.x;
        t1x = (max.x - o.x) * invdir.x;
    } else {
        t0x = (max.x - o.x) * invdir.x;
        t1x = (min.x - o.x) * invdir.x;
    }
    if(invdir.y >= 0) {
        t0y = (min.y - o.y) * invdir.y;
        t1y = (max.y - o.y) * invdir.y;
    } else {
        t0y = (max.y - o.y) * invdir.y;
        t1y = (min.y - o.y) * invdir.y;
    }
    if(invdir.z >= 0) {
        t0z = (min.z - o.z) * invdir.z;
        t1z = (max.z - o.z) * invdir.z;
    } else {
        t0z = (max.z - o.z) * invdir.z;
        t1z = (min.z - o.z) * invdir.z;
    }

    if((t0x > t1y) || (t0y > t1x)) return false;
    if(t0y > t0x) t0x = t0y;
    if(t1y < t1x) t1x = t1y;

    if((t0x > t1z) || (t0z > t1x)) return false;
    if(t0z > t0x) t0x = t0z;
    if(t1z < t1x) t1x = t1z;

    times[0] = t0x;
    times[1] = t1x;

    return true;
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
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}
