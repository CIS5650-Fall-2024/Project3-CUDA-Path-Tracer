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

__host__ __device__ float meshIntersectionTest(
    Geom geom,
    Ray ray,
    glm::vec3* dev_meshes_positions,
    uint16_t* dev_meshes_indices,
    glm::vec3* dev_meshes_normals,
    glm::vec2* dev_meshes_uvs,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside
) {
    float t_min = FLT_MAX;
    bool hit = false;

    for (int i = geom.indexOffset; i < geom.indexOffset + geom.indexCount; i += 3) {
        glm::vec3 v0 = glm::vec3(geom.transform * glm::vec4(dev_meshes_positions[geom.offset+dev_meshes_indices[i]], 1.0f));
        glm::vec3 v1 = glm::vec3(geom.transform * glm::vec4(dev_meshes_positions[geom.offset+dev_meshes_indices[i+1]], 1.0f));
        glm::vec3 v2 = glm::vec3(geom.transform * glm::vec4(dev_meshes_positions[geom.offset+dev_meshes_indices[i+2]], 1.0f));

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        glm::vec3 h = glm::cross(ray.direction, edge2);
        float a = glm::dot(edge1, h);

        if (fabs(a) < 1e-8f) {
            continue;
        }

        float f = 1.0f / a;
        glm::vec3 s = ray.origin - v0;
        float u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) {
            continue;
        }

        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) {
            continue;
        }

        float t = f * glm::dot(edge2, q);

        if (t > 0.0001f && t < t_min) {
            t_min = t;
            intersectionPoint = ray.origin + t * ray.direction;

            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(geom.transform)));
            normal = glm::normalize(normalMatrix * glm::cross(edge1, edge2));

            outside = glm::dot(ray.direction, normal) < 0.0f;

            if (!outside) {
                normal = -normal;
            }

            hit = true;
        }
    }

    if (!hit) {
        return -1.0f;
    }

    return t_min;
}

__host__ __device__ bool intersectMeshBVH(
    const Geom& mesh,              
    const BVHNode* meshBVH,        
    int nodeIdx,                   
    const Ray& ray,                
    glm::vec3* dev_meshes_positions, 
    uint16_t* dev_meshes_indices,  
    glm::vec3* dev_meshes_normals, 
    glm::vec3& intersectionPoint,  
    glm::vec3& normal,             
    bool& outside,                 
    float& t_min                   
) {
    const BVHNode& node = meshBVH[nodeIdx];
    float tMin = 0.0f, tMax = FLT_MAX;

    if (!node.bounds.intersect(ray, tMin, tMax)) {
        return false;  
    }

    if (node.isLeaf) {
        bool hit = false;

        for (int i = node.start; i < node.end; ++i) {
            int triIdx = mesh.triangleIndices[i];  

            glm::vec3 v0 = dev_meshes_positions[mesh.offset + dev_meshes_indices[triIdx * 3]];
            glm::vec3 v1 = dev_meshes_positions[mesh.offset + dev_meshes_indices[triIdx * 3 + 1]];
            glm::vec3 v2 = dev_meshes_positions[mesh.offset + dev_meshes_indices[triIdx * 3 + 2]];

            v0 = glm::vec3(mesh.transform * glm::vec4(v0, 1.0f));
            v1 = glm::vec3(mesh.transform * glm::vec4(v1, 1.0f));
            v2 = glm::vec3(mesh.transform * glm::vec4(v2, 1.0f));

            // Perform ray-triangle intersection using Möller-Trumbore algorithm.
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 h = glm::cross(ray.direction, edge2);
            float a = glm::dot(edge1, h);

            // If the determinant is near zero, the ray is parallel to the triangle.
            if (fabs(a) < 1e-8f) continue;

            float f = 1.0f / a;
            glm::vec3 s = ray.origin - v0;
            float u = f * glm::dot(s, h);

            // The intersection lies outside the triangle.
            if (u < 0.0f || u > 1.0f) continue;

            glm::vec3 q = glm::cross(s, edge1);
            float v = f * glm::dot(ray.direction, q);

            // The intersection lies outside the triangle.
            if (v < 0.0f || u + v > 1.0f) continue;

            float t = f * glm::dot(edge2, q);

            if (t > 0.0001f && t < t_min) {
                t_min = t;
                intersectionPoint = ray.origin + t * ray.direction;
                normal = glm::normalize(glm::cross(edge1, edge2));

                outside = glm::dot(ray.direction, normal) < 0.0f;
                if (!outside) {
                    normal = -normal; 
                }

                hit = true;
            }
        }

        return hit;
    }

    bool hitLeft = intersectMeshBVH(mesh, meshBVH, node.left, ray, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, intersectionPoint, normal, outside, t_min);
    bool hitRight = intersectMeshBVH(mesh, meshBVH, node.right, ray, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, intersectionPoint, normal, outside, t_min);

    return hitLeft || hitRight;
}


__host__ __device__ bool intersectBVH(
    const BVHNode* bvhNodes,       
    const Ray& ray,                
    int nodeIdx,                   
    const Geom* geoms,             
    glm::vec3* dev_meshes_positions,
    uint16_t* dev_meshes_indices,  
    glm::vec3* dev_meshes_normals, 
    int& hit_geom_index,           
    glm::vec3& intersectionPoint,  
    glm::vec3& normal,             
    bool& outside,                 
    float& t_min                   
) {
    const BVHNode& node = bvhNodes[nodeIdx];
    float tMin = 0.0f, tMax = FLT_MAX;

    if (!node.bounds.intersect(ray, tMin, tMax)) {
        return false;  
    }

    bool hit = false;

    if (node.isLeaf) {
        for (int i = node.start; i < node.end; ++i) {
            const Geom& mesh = geoms[i];

            glm::vec3 tmp_intersect_point;
            glm::vec3 tmp_normal;
            bool tmp_outside;
            float tmp_t_min = t_min; 

            bool hitMesh = intersectMeshBVH(
                mesh,                         
                mesh.meshBVH,                 
                mesh.bvhRoot,                 
                ray,                          
                dev_meshes_positions,         
                dev_meshes_indices,           
                dev_meshes_normals,           
                tmp_intersect_point,          
                tmp_normal,                   
                tmp_outside,                  
                tmp_t_min                     
            );

            if (hitMesh && tmp_t_min < t_min) {
                hit = true;
                t_min = tmp_t_min; 
                hit_geom_index = i;
                intersectionPoint = tmp_intersect_point; 
                normal = tmp_normal;              
                outside = tmp_outside;            
            }
        }

        return hit;
    }

    bool hitLeft = intersectBVH(bvhNodes, ray, node.left, geoms, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, hit_geom_index, intersectionPoint, normal, outside, t_min);
    bool hitRight = intersectBVH(bvhNodes, ray, node.right, geoms, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, hit_geom_index, intersectionPoint, normal, outside, t_min);

    return hitLeft || hitRight;
}
