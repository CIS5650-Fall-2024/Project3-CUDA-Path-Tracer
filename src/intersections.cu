#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
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

__host__ __device__ float squareIntersectionTest(
    Geom square,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal)
{
    Ray q;
    q.origin = multiplyMV(square.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(square.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = -q.origin.z / q.direction.z;
    if (t <= 0)
    {
        return -1;
    }

    glm::vec3 point = getPointOnRay(q, t);
    if (abs(point.x) > 0.5 || abs(point.y) > 0.5)
    {
        return -1;
    }

    intersectionPoint = multiplyMV(square.transform, glm::vec4(point, 1.0f));
    normal = glm::normalize(multiplyMV(square.invTranspose, glm::vec4(0, 0, 1, 0)));
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float doubleTriangleArea(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2)
{
    return glm::length(glm::cross(p1 - p0, p2 - p0));
}

__host__ __device__ float meshIntersectionTest(
    Geom geom,
    const Mesh *meshes,
    const int *indices,
    const glm::vec3 *points,
    const glm::vec2 *uvs,
    Ray ray,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    glm::vec2 &albedoUv,
    glm::vec2 &emissiveUv
    )
{
    glm::vec3 d = multiplyMV(geom.inverseTransform, glm::vec4(ray.direction, 0));
    float l = glm::length(d);
    Ray q{
        .origin = multiplyMV(geom.inverseTransform, glm::vec4(ray.origin, 1)),
        .direction = glm::normalize(d)};
    Mesh mesh = meshes[geom.meshId];

    float tMin = FLT_MAX;
    glm::vec3 localNormal;
    glm::vec3 localIntersect;
    glm::vec2 tmpAlbedoUv;
    glm::vec2 tmpEmissiveUv;

    int indOffset = mesh.indOffset;
    for (int i = 0; i < mesh.triCount; i++)
    {
        int triangleIndices[3];
        glm::vec3 trianglePoints[3];
        for (int j = 0; j < 3; j++)
        {
            triangleIndices[j] = indices[indOffset++];
            trianglePoints[j] = points[mesh.pointOffset + triangleIndices[j]];
        }

        glm::vec3 intersectResult;
        if (!glm::intersectRayTriangle(
                q.origin, q.direction,
                trianglePoints[0],
                trianglePoints[1],
                trianglePoints[2],
                intersectResult) ||
            intersectResult.z >= tMin)
        {
            continue;
        }

        tMin = intersectResult.z;
        glm::vec3 baryPos(1 - intersectResult.x - intersectResult.y, intersectResult.x, intersectResult.y);
        
        localNormal = glm::normalize(
            glm::cross(
                trianglePoints[1] - trianglePoints[0],
                trianglePoints[2] - trianglePoints[0]
            ));

        localIntersect = glm::vec3();
        for (size_t j = 0; j < 3; j++)
        {
            localIntersect += baryPos[j] * trianglePoints[j];
        }

        if (mesh.albedoUvOffset != -1)
        {
            tmpAlbedoUv = glm::vec2();
            tmpEmissiveUv = glm::vec2();
            for (size_t j = 0; j < 3; j++)
            {
                tmpAlbedoUv += baryPos[j] * uvs[mesh.albedoUvOffset + triangleIndices[j]];
                tmpEmissiveUv += baryPos[j] * uvs[mesh.emissiveUvOffset + triangleIndices[j]];
            }
        }
    }

    if (tMin == FLT_MAX)
    {
        return -1;
    }

    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0)));
    intersectionPoint = multiplyMV(geom.transform, glm::vec4(localIntersect, 1));
    albedoUv = tmpAlbedoUv;
    emissiveUv = tmpEmissiveUv;

    return tMin / l;
}

__device__ ShadeableIntersection queryIntersection(
    Ray ray,
    const Geom *geoms,
    int geomsSize,
    const Mesh *meshes,
    const int *indices,
    const glm::vec3 *points,
    const glm::vec2 *uvs)
{
    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_material = INT_MAX;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_albedouv;
    glm::vec2 tmp_emissiveuv;
    
    glm::vec2 albedouv;
    glm::vec2 emissiveuv;

    for (int i = 0; i < geomsSize; i++)
    {
        const Geom &geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SQUARE)
        {
            t = squareIntersectionTest(geom, ray, tmp_intersect, tmp_normal);
        }
        else
        {
            t = meshIntersectionTest(geom, meshes, indices, points, uvs, ray, tmp_intersect, tmp_normal, outside, tmp_albedouv, tmp_emissiveuv);
        }

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_material = geoms[i].materialid;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
            albedouv = tmp_albedouv;
            emissiveuv = tmp_emissiveuv;
        }
    }

    ShadeableIntersection intersection;

    intersection.materialId = hit_material;
    if (hit_material == INT_MAX)
    {
        intersection.t = -1.0f;
    }
    else
    {
        // The ray hits something
        intersection.t = t_min;
        intersection.materialId = hit_material;
        intersection.surfaceNormal = normal;
        intersection.albedoUv = albedouv;
        intersection.emissiveUv = emissiveuv;
    }

    return intersection;
}

__device__ int queryIntersectionGeometryIndex(
    Ray ray,
    const Geom *geoms,
    int geomsSize,
    const Mesh *meshes,
    const int *indices,
    const glm::vec3 *points,
    const glm::vec2 *uvs)
{
    float t_min = FLT_MAX;
    float t;
    int index = -1;
    bool outside;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_albedouv;
    glm::vec2 tmp_emissiveuv;

    for (int i = 0; i < geomsSize; i++)
    {
        const Geom &geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SQUARE)
        {
            t = squareIntersectionTest(geom, ray, tmp_intersect, tmp_normal);
        }
        else
        {
            t = meshIntersectionTest(geom, meshes, indices, points, uvs, ray, tmp_intersect, tmp_normal, outside, tmp_albedouv, tmp_emissiveuv);
        }

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            index = i;
        }
    }

    return index;
}