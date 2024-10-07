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
    
    return glm::length(r.origin - intersectionPoint);
}
__host__ __device__ bool rayIntersectsAABB(const Ray& ray, const AABB& box) {
    
    //const float EPSILON = 0.00001f;
    float tmin = (box.AABBmin.x - ray.origin.x) / (fabs(ray.direction.x) > 0.00001f ? ray.direction.x : 0.00001f);
    float tmax = (box.AABBmax.x - ray.origin.x) / (fabs(ray.direction.x) > 0.00001f ? ray.direction.x : 0.00001f);

    if (tmin > tmax) {
        float temp = tmin;
        tmin = tmax;
        tmax = temp;
    } 

    float tymin = (box.AABBmin.y - ray.origin.y) / (fabs(ray.direction.y) > 0.00001f ? ray.direction.y : 0.00001f);
    float tymax = (box.AABBmax.y - ray.origin.y) / (fabs(ray.direction.y) > 0.00001f ? ray.direction.y : 0.00001f);

   
    if (tymin > tymax) {
        float temp = tymin;
        tymin = tymax;
        tymax = temp;
    }

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (box.AABBmin.z - ray.origin.z) / (fabs(ray.direction.z) > 0.00001f ? ray.direction.z : 0.00001f);
    float tzmax = (box.AABBmax.z - ray.origin.z) / (fabs(ray.direction.z) > 0.00001f ? ray.direction.z : 0.00001f);

    if (tzmin > tzmax) {
        float temp = tzmin;
        tzmin = tzmax;
        tzmax = temp;
    }


    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    return true;



}
__host__ __device__ float meshIntersectionTest_BVH(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside) 
{
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_min = FLT_MAX;
    bool hit = false;
    glm::vec3 tempIntersectionPoint;
    glm::vec3 tempNormal;
    glm::vec2 tempUV;


    int stack[64];
    int stackPtr = 0;
    //the last nodes is root, buildBVH is post order
    stack[stackPtr++] = geom.numBVHNodes - 1;

    while (stackPtr > 0)
    {
        int nodeIdx = stack[--stackPtr];
        BVHNode node = geom.bvhNodes[nodeIdx];

        
        if (rayIntersectsAABB(q, node.bound))
        {
            if (node.isLeaf)
            {
                for (int j = node.start; j < node.end; ++j)
                {
                    Triangle tri = geom.triangles[j];
                    glm::vec3 baryPosition;
                    bool intersects = glm::intersectRayTriangle(
                        q.origin,
                        q.direction,
                        tri.v0,
                        tri.v1,
                        tri.v2,
                        baryPosition);


                    if (intersects)
                    {
                        float t = baryPosition.z;
                        if (t > 0.0f && t < t_min)
                        {
                            t_min = t;
                            hit = true;
                            tempIntersectionPoint = q.origin + t * q.direction;
                            tempNormal = tri.normal;

                            tempUV = (1.0f - baryPosition.x - baryPosition.y) * tri.uv0 +
                                baryPosition.x * tri.uv1 +
                                baryPosition.y * tri.uv2;
                        }
                    }

                
                }
            }
            else
            {
                stack[stackPtr++] = node.left;
                stack[stackPtr++] = node.right;
            }
        }



    }
    if (hit)
    {
        //Transform intersection point and normal back to world space
        intersectionPoint = multiplyMV(geom.transform, glm::vec4(tempIntersectionPoint, 1.0f));
        normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(tempNormal, 0.0f)));
        uv = tempUV;
        outside = glm::dot(r.direction, normal) < 0.0f;
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;


    
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
    float t_min = FLT_MAX;
    bool hit = false;
    glm::vec3 tempIntersectionPoint;
    glm::vec3 tempNormal;
    glm::vec2 tempUV;

    for (int i = 0; i < mesh.numTriangles; ++i)
    {
        const Triangle& tri = mesh.triangles[i];

        glm::vec3 baryPosition;

        bool intersects = glm::intersectRayTriangle(
            q.origin,
            q.direction,
            tri.v0,
            tri.v1,
            tri.v2,
            baryPosition);
        if (intersects) {
            float t = baryPosition.z;
            if (t > 0.0f && t < t_min)
            {
                t_min = t;
                hit = true;
                tempIntersectionPoint = q.origin + t * q.direction;
                tempNormal = tri.normal;

                tempUV = (1.0f - baryPosition.x - baryPosition.y) * tri.uv0 +
                    baryPosition.x * tri.uv1 +
                    baryPosition.y * tri.uv2;
            }
        }
    }
    if (hit)
    {
        // Transform intersection point and normal back to world space
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(tempIntersectionPoint, 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tempNormal, 0.0f)));
        uv = tempUV;
        outside = glm::dot(r.direction, normal) < 0.0f;
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}
