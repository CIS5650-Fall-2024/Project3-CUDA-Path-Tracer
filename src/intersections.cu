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
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ glm::vec3 barycentricInterpolation(glm::vec3 p, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {
	glm::vec3 edge1 = v2 - v1;
	glm::vec3 edge2 = v3 - v1;
    float s = glm::length(glm::cross(edge1, edge2));

    edge1 = p - v2;
    edge2 = p - v3;
	float s1 = glm::length(glm::cross(edge1, edge2)) / s;

	edge1 = p - v1;
	edge2 = p - v3;
	float s2 = glm::length(glm::cross(edge1, edge2)) / s;

	edge1 = p - v1;
	edge2 = p - v2;
	float s3 = glm::length(glm::cross(edge1, edge2)) / s;

	return glm::vec3(s1, s2, s3);
}


__host__ __device__ bool IntersectAABB(Ray& ray, const bbox& aabb)
{
    glm::vec3 bmin = aabb.boundsMin;
    glm::vec3 bmax = aabb.boundsMax;

    float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
    float tmin = glm::min(tx1, tx2), tmax = glm::max(tx1, tx2);
    float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
    tmin = glm::max(tmin, glm::min(ty1, ty2)), tmax = glm::min(tmax, glm::max(ty1, ty2));
    float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
    tmin = glm::max(tmin, glm::min(tz1, tz2)), tmax = glm::min(tmax, glm::max(tz1, tz2));

    return tmax >= tmin && tmax > 0;
}

__host__ __device__ float bvhMeshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    Triangle* triangles,
    BVHNode* nodes,
    int &meshId
) {
    const int STACK_SIZE = 64;
    bool intersected = false;
    float local_t, global_t = FLT_MAX;
    int tri = -1;

    int stack[STACK_SIZE];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        BVHNode& node = nodes[nodeIdx];

        if (!IntersectAABB(r, node.aabb)) {
            continue;
        }

        if (node.isLeaf()) {
            for (int i = 0; i < node.numTriangles; i++) {
                glm::vec3 barycentricIntersection;
                int triIdx = i + node.leftFirst;
                if (glm::intersectRayTriangle(r.origin, r.direction, triangles[triIdx].v1.pos, triangles[triIdx].v2.pos, triangles[triIdx].v3.pos, barycentricIntersection)) {
                    intersected = true;

                    float temp_t = barycentricIntersection.z;
                    if (temp_t < global_t || global_t == -1) {
                        global_t = temp_t;
                        tri = triIdx;
                    }
                }
            }
        }
        else {
            stack[stackPtr++] = node.leftFirst;
            stack[stackPtr++] = node.leftFirst + 1;
        }
    }

    if (stackPtr > STACK_SIZE || !intersected) {
        return -1.f;
    }

    glm::vec3 barycentric, intersection, localNormal;

    // Calculate intersection
    intersection = r.origin + global_t * r.direction;

    // Calcuate Barycentric Weights and use for uvs and normals if necessary
    Vertex v1 = triangles[tri].v1;
    Vertex v2 = triangles[tri].v2;
    Vertex v3 = triangles[tri].v3;
    // Record Triangle textureId since we don't know the specific mesh
    meshId = triangles[tri].meshId;

    barycentric = barycentricInterpolation(intersection, v1.pos, v2.pos, v3.pos);

	// TODO Add Bump Mapping with Normal Interpolation and Hard Normals
    normal = mesh.usesNormals ? barycentric.x * v1.nor + barycentric.y * v2.nor + barycentric.z * v3.nor :
        glm::normalize(glm::cross(v2.pos - v1.pos, v3.pos - v1.pos));

    if (mesh.usesUVs) {
        uv = glm::vec2(barycentric.x * v1.uv[0] + barycentric.y * v2.uv[0] + barycentric.z * v3.uv[0], barycentric.x * v1.uv[1] + barycentric.y * v2.uv[1] + barycentric.z * v3.uv[1]);
    }
    else {
        uv = glm::vec2(0.0f);
    }

	intersectionPoint = intersection;
    return glm::length(r.origin - intersection);
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    glm::vec2 &uv,
    Triangle* triangles,
    int start, int end
) {
	bool intersected = false;
    float t = FLT_MAX;
    int tri = -1;
    // Go through each triangle
    for (int i = start; i <= end; i++) {
        glm::vec3 barycentricIntersection;
        if (glm::intersectRayTriangle(r.origin, r.direction, triangles[i].v1.pos, triangles[i].v2.pos, triangles[i].v3.pos, barycentricIntersection)) {
            intersected = true;

            float temp_t = barycentricIntersection.z;
            if (temp_t < t) {
                t = temp_t;
                tri = i;
            }
        }
    }

    if (!intersected) {
        return -1.0f;
    }

    glm::vec3 barycentric, intersection, localNormal;

    // Calculate intersection
    intersection = r.origin + t * r.direction;

    // Calcuate Barycentric Weights and use for uvs and normals if necessary
    Vertex v1 = triangles[tri].v1;
    Vertex v2 = triangles[tri].v2;
    Vertex v3 = triangles[tri].v3;

    barycentric = barycentricInterpolation(intersection, v1.pos, v2.pos, v3.pos);

    normal = mesh.usesNormals ? barycentric.x * v1.nor + barycentric.y * v2.nor + barycentric.z * v3.nor :
        glm::normalize(glm::cross(v2.pos - v1.pos, v3.pos - v1.pos));

	if (mesh.usesUVs) {
		uv = glm::vec2(barycentric.x * v1.uv[0] + barycentric.y * v2.uv[0] + barycentric.z * v3.uv[0], barycentric.x * v1.uv[1] + barycentric.y * v2.uv[1] + barycentric.z * v3.uv[1]);
    }
    else {
		uv = glm::vec2(0.0f);
	}

    return glm::length(r.origin - intersection);

}