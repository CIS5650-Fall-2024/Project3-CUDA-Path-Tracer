#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    glm::vec2& uv)
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

        glm::vec3 localIntersection = getPointOnRay(q, tmin);
        if (abs(tmin_n.x) > 0.5f) {
            uv = glm::vec2(localIntersection.z + 0.5f, localIntersection.y + 0.5f);
        }
        else if (abs(tmin_n.y) > 0.5f) {
            uv = glm::vec2(localIntersection.x + 0.5f, localIntersection.z + 0.5f);
        }
        else {
            uv = glm::vec2(localIntersection.x + 0.5f, localIntersection.y + 0.5f);
        }
        uv = glm::mod(uv, 1.0f);

        intersectionPoint = multiplyMV(box.transform, glm::vec4(localIntersection, 1.0f));
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
    bool &outside,
    glm::vec2& uv)
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
        t = fmin(t1, t2);
        outside = true;
    }
    else
    {
        t = fmax(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    float theta = acos(objspaceIntersection.y / radius);
    float phi = atan2(objspaceIntersection.z, objspaceIntersection.x);
    uv.x = (phi + PI) / (2.0f * PI);
    uv.y = theta / PI;

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// reference https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
__host__ __device__ float meshIntersectionTestBVH(
    Geom& geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    //Mesh data
    BVHNode* bvh, MeshTriangle* meshes, glm::vec3* vertices, glm::vec3* normals, glm::vec2* texcoords, int& materialid){
    float t_min = FLT_MAX;
    
    int stack[64];
    int* stackPtr = stack;
    *stackPtr++ = -1;

    int nodeIdx = geom.bvhrootidx;

    do
    {
        BVHNode node = bvh[nodeIdx];

        int idxL = node.left;
        BVHNode childL = bvh[idxL];
        int idxR = node.right;
        BVHNode childR = bvh[idxR];

        bool overlapL = aabbIntersectionTest(childL.aabb, r);
        bool overlapR = aabbIntersectionTest(childR.aabb, r);

        if (overlapL && (childL.left == -1 && childL.right == -1))
        {
            finalIntersectionTest(
                meshes[childL.meshidx], vertices, normals, texcoords,
                r,
                t_min, intersectionPoint, materialid, normal, uv);
        }

        if (overlapR && (childR.left == -1 && childR.right == -1))
        {
            finalIntersectionTest(
                meshes[childR.meshidx], vertices, normals, texcoords,
                r,
                t_min, intersectionPoint, materialid, normal, uv);
        }
        
        bool traverseL = overlapL && !(childL.left == -1 && childL.right == -1);
        bool traverseR = overlapR && !(childR.left == -1 && childR.right == -1);

        if (!traverseL && !traverseR)
            nodeIdx = *--stackPtr;
        else
        {
            nodeIdx = (traverseL) ? idxL : idxR;
            if (traverseL && traverseR)
                *stackPtr++ = idxR;
        }

    } while (nodeIdx != -1);
    return t_min;
}

// reference https://tavianator.com/2011/ray_box.html
__host__ __device__ bool aabbIntersectionTest(const AABB& aabb, const Ray& ray)
{
    float invDirX = 1.f / ray.direction.x;
    float invDirY = 1.f / ray.direction.y;
    float invDirZ = 1.f / ray.direction.z;

    float tx1 = (aabb.min.x - ray.origin.x) * invDirX;
    float tx2 = (aabb.max.x - ray.origin.x) * invDirX;

    float tmin = glm::min(tx1, tx2);
    float tmax = glm::max(tx1, tx2);

    float ty1 = (aabb.min.y - ray.origin.y) * invDirY;
    float ty2 = (aabb.max.y - ray.origin.y) * invDirY;

    tmin = glm::max(tmin, glm::min(ty1, ty2));
    tmax = glm::min(tmax, glm::max(ty1, ty2));

    float tz1 = (aabb.min.z - ray.origin.z) * invDirZ;
    float tz2 = (aabb.max.z - ray.origin.z) * invDirZ;

    tmin = glm::max(tmin, glm::min(tz1, tz2));
    tmax = glm::min(tmax, glm::max(tz1, tz2));

    return tmax >= glm::max(0.f, tmin);
}

__host__ __device__ void finalIntersectionTest(
    const MeshTriangle& m, const glm::vec3* vertices, const glm::vec3* normals, const glm::vec2* texcoords,
    const Ray& r,
    float& t_min, glm::vec3& intersectionPoint, int& materialid, glm::vec3& normal, glm::vec2& texcoord)
{
    glm::vec3 v0 = vertices[m.v[0]];
    glm::vec3 v1 = vertices[m.v[1]];
    glm::vec3 v2 = vertices[m.v[2]];

    glm::vec3 baryPosition;

    if (glm::intersectRayTriangle(r.origin, r.direction, v0, v1, v2, baryPosition)||
        glm::intersectRayTriangle(r.origin, r.direction, v0, v2, v1, baryPosition))
    {
        glm::vec3 point = (1 - baryPosition.x - baryPosition.y) * v0 +
            baryPosition.x * v1 +
            baryPosition.y * v2;
        float t = glm::length(r.origin - point);
        if (t > 0.0f && t < t_min)
        {
            t_min = t;
            intersectionPoint = point;
            materialid = m.materialid;
            if (m.vn[0] == -1 || m.vn[1] == -1 || m.vn[2] == -1)
                normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            else
                normal = (1 - baryPosition.x - baryPosition.y) * normals[m.vn[0]] +
                baryPosition.x * normals[m.vn[1]] +
                baryPosition.y * normals[m.vn[2]];
            if (m.vt[0] == -1 || m.vt[1] == -1 || m.vt[2] == -1)
                texcoord = glm::vec2(-1.f); // no texture
            else
                texcoord = (1 - baryPosition.x - baryPosition.y) * texcoords[m.vt[0]] +
                baryPosition.x * texcoords[m.vt[1]] +
                baryPosition.y * texcoords[m.vt[2]];
            
        }
    }
}

__host__ __device__ float meshIntersectionTestNaive(
    Geom& geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    MeshTriangle* meshes, glm::vec3* vertices, glm::vec3* normals, glm::vec2* texcoords, int& materialid)
{
    float t_min = FLT_MAX;

    for (int i = 0; i < geom.meshcnt; i++)
    {
        MeshTriangle& mesh = meshes[geom.meshidx + i];

        glm::vec3 v0 = vertices[mesh.v[0]];
        glm::vec3 v1 = vertices[mesh.v[1]];
        glm::vec3 v2 = vertices[mesh.v[2]];

        glm::vec3 tmpIntersectionPoint;
        glm::vec3 tmpNormal;
        glm::vec2 tmpUV;
        float t;

        if (rayTriangleIntersect(r, v0, v1, v2, t, tmpIntersectionPoint, tmpNormal, tmpUV))
        {
            if (t > 0 && t < t_min)
            {
                t_min = t;
                intersectionPoint = tmpIntersectionPoint;
                normal = tmpNormal;
                uv = tmpUV;
                materialid = mesh.materialid;
                outside = true;
            }
        }
    }

    return t_min;
}

__host__ __device__ bool rayTriangleIntersect(
    const Ray& ray, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    float& t, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv)
{
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 p = glm::cross(ray.direction, e2);
    float det = glm::dot(e1, p);

    if (det > -EPSILON && det < EPSILON)
        return false;

    float inv_det = 1.0f / det;
    glm::vec3 t_vec = ray.origin - v0;
    float u = glm::dot(t_vec, p) * inv_det;

    if (u < 0.0f || u > 1.0f)
        return false;

    glm::vec3 q = glm::cross(t_vec, e1);
    float v = glm::dot(ray.direction, q) * inv_det;

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = glm::dot(e2, q) * inv_det;

    if (t > EPSILON)
    {
        intersectionPoint = ray.origin + ray.direction * t;
        normal = glm::normalize(glm::cross(e1, e2));
        uv = glm::vec2(u, v);
        return true;
    }

    return false;
}