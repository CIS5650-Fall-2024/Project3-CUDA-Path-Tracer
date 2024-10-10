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
    /*if (!outside)
    {
        normal = -normal;
    }*/

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    Geom sphere,
    Primitive p,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    //Moller Trumbore Algorithm for triangle intersection test
    glm::vec3 E1 = p.p2 - p.p1;
    glm::vec3 E2 = p.p3 - p.p1;
    glm::vec3 S = r.origin - p.p1;
    glm::vec3 S1 = glm::cross(r.direction, E2);
    glm::vec3 S2 = glm::cross(S, E1);
    glm::vec3 result = glm::vec3(glm::dot(S2, E2), glm::dot(S1, S), glm::dot(S2, r.direction));
    result *= (1.0f / glm::dot(S1, E1));

    if ((0.0f <= result.x) &&
        (0.0f <= result.y && result.y <= 1.0f) &&
        (0.0f <= result.z && result.z <= 1.0f) &&
        (0.0f <= (1 - result.y - result.z) && (1 - result.y - result.z) <= 1.0f))
    {
        intersectionPoint = r.origin + r.direction * result.x;
        normal = glm::normalize(p.n2 * result.y + p.n3 * result.z + p.n1 * (1 - result.y - result.z));

        return result.x;
    }
    else
    {
        return -1;
    }
}

__device__ bool boxIntersection(
    Geom& box,
    Primitive& p,
    Ray& r,
    ShadeableIntersection& intersection)
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
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
        }

        if (r.tmin <= tmin && tmin <= r.tmax)
        {
            r.tmax = tmin;

            intersection.t = tmin;
            intersection.surfaceNormal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));

            return true;
        }
        
        return false;
    }
    else 
    {
        return false;
    }
}

__device__ bool sphereIntersection(
    Geom& sphere,
    Primitive& p,
    Ray& r,
    ShadeableIntersection& intersection)
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
        return false;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return false;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = fminf(t1, t2);
    }
    else
    {
        t = fmaxf(t1, t2);
    }

    if (r.tmin <= t && t <= r.tmax)
    {
        r.tmax = t;

        intersection.t = t;
        glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
        intersection.surfaceNormal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

        return true;
    }
    else 
    {
        return false;
    }
}

__device__ bool triangleIntersection(
    Geom& tri,
    Primitive& p,
    Ray& r,
    ShadeableIntersection& intersection)
{
    //Moller Trumbore Algorithm for triangle intersection test
    glm::vec3 E1 = p.p2 - p.p1;
    glm::vec3 E2 = p.p3 - p.p1;
    glm::vec3 S = r.origin - p.p1;
    glm::vec3 S1 = glm::cross(r.direction, E2);
    glm::vec3 S2 = glm::cross(S, E1);
    glm::vec3 result = glm::vec3(glm::dot(S2, E2), glm::dot(S1, S), glm::dot(S2, r.direction));
    result *= (1.0f / glm::dot(S1, E1));

    if ((r.tmin <= result.x && result.x <= r.tmax) &&
        (0.0f <= result.y && result.y <= 1.0f) &&
        (0.0f <= result.z && result.z <= 1.0f) &&
        (0.0f <= (1 - result.y - result.z) && (1 - result.y - result.z) <= 1.0f))
    {
        r.tmax = result.x;

        intersection.t = result.x;
        intersection.surfaceNormal = glm::normalize(p.n2 * result.y + p.n3 * result.z + p.n1 * (1 - result.y - result.z));

        return true;
    }
    else 
    {
        return false;
    }
}