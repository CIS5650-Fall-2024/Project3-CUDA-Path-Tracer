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

__host__ __device__ void CoordinateSystem(const glm::vec3& v1, glm::vec3* v2, glm::vec3* v3)
{
    if (std::abs(v1.x) > std::abs(v1.y)) {
        *v2 = glm::normalize(glm::vec3(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z));
    }
    else {
        *v2 = glm::normalize(glm::vec3(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z));
    }
    *v3 = glm::normalize(glm::cross(v1, *v2));
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

__host__ __device__ float customMeshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    int vertex_size,
    Vertex* vertices,
    Material* materials,
    Texture* textures,
    bool& outside)
{
    // from the big lists:  
    //      get current mesh's vertices (treat multiple obj parts in an obj file as one mesh here)
    // for each triangle:
    //      test intersect. 
    //          If not, return -1
    //          If intersected, determine inside/outside based on cos(dir, normal), return t
    Ray q = r;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_final = -1;
    int t_v_idx = -1;
    glm::vec2 vertex_indices = mesh.vertex_indices;
    for (int vertex_idx = vertex_indices.x; vertex_idx <= vertex_indices.y; vertex_idx += 3) {

        // Möller–Trumbore intersection
        glm::vec3 p0 = vertices[vertex_idx].pos;
        glm::vec3 p1 = vertices[vertex_idx + 1].pos;
        glm::vec3 p2 = vertices[vertex_idx + 2].pos;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 ray_cross_e2 = cross(q.direction, edge2);
        float det = dot(edge1, ray_cross_e2);

        if (det > -EPSILON && det < EPSILON) continue;

        float inv_det = 1.0 / det;
        glm::vec3 s = q.origin - p0;
        float u = inv_det * dot(s, ray_cross_e2);

        if (u < 0 || u > 1) continue;

        glm::vec3 s_cross_e1 = cross(s, edge1);
        float v = inv_det * dot(q.direction, s_cross_e1);

        if (v < 0 || u + v > 1) continue;

        float t = inv_det * dot(edge2, s_cross_e1);
        if (t > EPSILON) {
            if (t_final > t || t_final == -1) {
                t_final = t;
                t_v_idx = vertex_idx;
            }
        }
    }
    if (t_final < 0) {
        return -1;
    }

    // compute intersection
    intersectionPoint = getPointOnRay(q, t_final);
    glm::vec3 baryCentric_factor = baryCentric_interpolation(
        vertices[t_v_idx].pos,
        vertices[t_v_idx + 1].pos,
        vertices[t_v_idx + 2].pos,
        intersectionPoint);

    uv = vertices[t_v_idx].uv * baryCentric_factor.x +
        vertices[t_v_idx + 1].uv * baryCentric_factor.y +
        vertices[t_v_idx + 2].uv * baryCentric_factor.z;

    // compute normal
    int bumpMapIdx = materials[mesh.materialid].bumpMapTextIdx;
    if (bumpMapIdx < 0) {
        normal = glm::normalize(vertices[t_v_idx].normal * baryCentric_factor.x +
                vertices[t_v_idx + 1].normal * baryCentric_factor.y +
                vertices[t_v_idx + 2].normal * baryCentric_factor.z);
    }
    else {
        glm::vec3 normalColor = 2.f * getColorFromTexture(uv, textures[bumpMapIdx]) - 1.f;
        glm::vec3 tangent, bitangent = glm::vec3(0.0);
        CoordinateSystem(normalColor, &tangent, &bitangent);
        glm::mat3 TBN = glm::mat3(tangent, bitangent, normalColor);
        normal = glm::normalize(TBN * normalColor);
    }

    



    // evaluate inside or outside
    outside = dot(normalize(normal), normalize(q.direction)) < 0.f;

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normal, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return t_final;
}

__host__ __device__ glm::vec3 baryCentric_interpolation(const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, const glm::vec3& p) {
    float S = triangle_area(p1, p2, p3);
    float S1 = triangle_area(p, p2, p3);
    float S2 = triangle_area(p, p3, p1);
    float S3 = triangle_area(p, p1, p2);
    return glm::vec3(S1 / S, S2 / S, S3 / S);
}

__host__ __device__ float triangle_area(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) {
    return 0.5 * glm::length(glm::cross(p2 - p1, p0 - p1));
}

__host__ __device__ glm::vec3 getColorFromTexture(glm::vec2 uv, const Texture& texture) {
    int x = static_cast<int>(uv.x * texture.width) % texture.width;
    int y = static_cast<int>((1.f - uv.y) * texture.height) % texture.height;
    int index = (y * texture.width + x) * texture.channel;
    float r = texture.imgData[index] / 255.0f;
    float g = texture.imgData[index + 1] / 255.0f;
    float b = texture.imgData[index + 2] / 255.0f;
    return glm::vec3(r, g, b);
}