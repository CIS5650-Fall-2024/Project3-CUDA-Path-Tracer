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
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ bool triangleIntersectionTest(
    Ray r,
    Triangle triangle,
    float& t,
    glm::vec3& baryPosition)
{
    // Möller–Trumbore intersection algorithm
    const float epsilon = 1e-8f;
    glm::vec3 vertex0 = triangle.v0;
    glm::vec3 vertex1 = triangle.v1;
    glm::vec3 vertex2 = triangle.v2;

    glm::vec3 edge1 = vertex1 - vertex0;
    glm::vec3 edge2 = vertex2 - vertex0;
    glm::vec3 h = glm::cross(r.direction, edge2);
    float a = glm::dot(edge1, h);

    if (fabs(a) < epsilon)
        return false; // Ray is parallel to triangle

    float f = 1.0f / a;
    glm::vec3 s = r.origin - vertex0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(r.direction, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float temp_t = f * glm::dot(edge2, q);
    if (temp_t > epsilon)
    {
        t = temp_t;
        baryPosition = glm::vec3(u, v, 0.0f);
        return true;
    }
    else
        return false;
}

__device__ void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
};

__device__ bool aabbIntersectionTest(const Ray& ray, const glm::vec3& min, const glm::vec3& max, float& t)
{
    float tmin = (min.x - ray.origin.x) / ray.direction.x;
    float tmax = (max.x - ray.origin.x) / ray.direction.x;

    // Swap tmin and tmax if needed
    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (min.y - ray.origin.y) / ray.direction.y;
    float tymax = (max.y - ray.origin.y) / ray.direction.y;

    // Swap tymin and tymax if needed
    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (min.z - ray.origin.z) / ray.direction.z;
    float tzmax = (max.z - ray.origin.z) / ray.direction.z;

    // Swap tzmin and tzmax if needed
    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    // Check if the ray starts inside the box
    bool inside = (ray.origin.x > min.x && ray.origin.x < max.x) &&
        (ray.origin.y > min.y && ray.origin.y < max.y) &&
        (ray.origin.z > min.z && ray.origin.z < max.z);

    // If the ray starts inside the box, we need to use tmax as the exit point
    if (inside)
    {
        t = tmax;
    }
    else
    {
        t = tmin;
    }

    return t >= 0.0f;
}
