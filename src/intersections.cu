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
    //if (!outside)
    //{
    //    normal = -normal;
    //}

    return glm::length(r.origin - intersectionPoint);
}








__host__ __device__ bool triangleIntersectionTest(
    const Triangle& tri,
    const Vertex* dev_vertices,  // Use device pointer instead of vector
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    float& t)
{
    // Retrieve the triangle's vertices using the indices
    glm::vec3 v0 = dev_vertices[tri.idx_v0].pos;
    glm::vec3 v1 = dev_vertices[tri.idx_v1].pos;
    glm::vec3 v2 = dev_vertices[tri.idx_v2].pos;

    // Calculate edges
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    // Calculate the determinant
    glm::vec3 h = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, h);

    // If the determinant is near zero, the ray is parallel to the triangle
    if (fabs(det) < 1e-6f) return false;

    float f = 1.0f / det;
    glm::vec3 s = r.origin - v0; // s is used to determine where the ray originates relative to the triangle. 
    float u = f * glm::dot(s, h); // Computes one of the barycentric coordinates u for the intersection point.

    // Check if intersection lies outside the triangle
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 qvec = glm::cross(s, edge1);
    float v = f * glm::dot(r.direction, qvec);

    // Check if intersection lies outside the triangle
    if (v < 0.0f || u + v > 1.0f) return false;

    // Calculate the distance along the ray to the intersection
    t = f * glm::dot(edge2, qvec);

    // If the intersection is valid and in front of the ray origin
    if (t > 1e-6f) {
        // Calculate intersection point and normal
        intersectionPoint = r.origin + t * r.direction;
        normal = glm::normalize(glm::cross(edge1, edge2));
        return true;
    }

    return false;
}



__host__ __device__ float objMeshIntersectionTest(
    const Geom& obj,
    const Vertex* dev_vertices,  // Use device pointer
    const Triangle* dev_triangles,  // Use device pointer
    int numTriangles, 
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // Transform ray into object space
    Ray q;
    q.origin = multiplyMV(obj.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(obj.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float closestT = 1e38f;
    bool hit = false;

    // Iterate over all triangles in the mesh
   // Iterate over all triangles in the mesh
    for (int i = 0; i < numTriangles; i++) {
        glm::vec3 tempIntersectionPoint, tempNormal;
        float t;

        // Use the single-triangle intersection function
        if (triangleIntersectionTest(dev_triangles[i], dev_vertices, q, tempIntersectionPoint, tempNormal, t)) {
            // Check if this intersection is the closest one
            if (t < closestT) {
                closestT = t;
                hit = true;

                // Update intersection point and normal
                intersectionPoint = multiplyMV(obj.transform, glm::vec4(tempIntersectionPoint, 1.0f));
                normal = glm::normalize(multiplyMV(obj.invTranspose, glm::vec4(tempNormal, 0.0f)));

                // Determine if the intersection is outside
                outside = glm::dot(q.direction, tempNormal) < 0;
            }
        }
    }

    if (hit) {
        // Return the distance to the closest intersection
        return glm::length(r.origin - intersectionPoint);
    }

    // No intersection found
    return -1;
}





