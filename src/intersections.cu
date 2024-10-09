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

bool rayIntersectsAABB(const Ray& ray, const glm::vec3& min, const glm::vec3& max) {
	// Slabs Method for Ray-AABB intersection
	float tmin = (min.x - ray.origin.x) / ray.direction.x;
	float tmax = (max.x - ray.origin.x) / ray.direction.x;

	if (tmin > tmax) std::swap(tmin, tmax);

	float tymin = (min.y - ray.origin.y) / ray.direction.y;
	float tymax = (max.y - ray.origin.y) / ray.direction.y;

	if (tymin > tymax) std::swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (min.z - ray.origin.z) / ray.direction.z;
	float tzmax = (max.z - ray.origin.z) / ray.direction.z;

	if (tzmin > tzmax) std::swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	return true;
}

__host__ __device__ 
float meshIntersectionTest(
    Geom mesh,
	Triangle* tris,
    Ray ray,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
	glm::vec2& uvOut,
    bool& outside) {
#if BOUNDING_VOLUME_INTERSECTION_CULLING_ENABLED
	// culling box test
    if (!rayIntersectsAABB(ray, mesh.min, mesh.max)){
        return -1.0f;
    }
#endif
	glm::vec3 objOrigin = multiplyMV(mesh.inverseTransform, glm::vec4(ray.origin, 1.0f));
	glm::vec3 objDir = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(ray.direction, 0.0f)));

	float closestT = FLT_MAX;
	glm::vec3 closestNormal(0.0f);
	glm::vec2 closestUV(0.0f);
	bool hit = false;

	for (int i = mesh.startTriangleIndex; i <= mesh.endTriangleIndex; ++i) {
		const Triangle& tri = tris[i];
		glm::vec3 baryPosition;

		if (glm::intersectRayTriangle(objOrigin, objDir,
			tri.v0.position, tri.v1.position, tri.v2.position,
			baryPosition)) {
			float t = baryPosition.z;
			if (t > 0.0f && t < closestT) {
				closestT = t;
				hit = true;

                if (mesh.hasNormals) {
                    glm::vec3 n0 = tri.v0.normal;
                    glm::vec3 n1 = tri.v1.normal;
                    glm::vec3 n2 = tri.v2.normal;
                    closestNormal = glm::normalize(
                        (1.0f - baryPosition.x - baryPosition.y) * n0 +
                        baryPosition.x * n1 +
                        baryPosition.y * n2
                    );

				}else {
					closestNormal = glm::cross(tri.v1.position - tri.v0.position, tri.v2.position - tri.v0.position);
				}

                if (mesh.hasUVs) {
					glm::vec2 uv0 = tri.v0.uv;
					glm::vec2 uv1 = tri.v1.uv;
					glm::vec2 uv2 = tri.v2.uv;
					closestUV = (1.0f - baryPosition.x - baryPosition.y) * uv0 +
						baryPosition.x * uv1 +
						baryPosition.y * uv2;
                }else {
					closestUV = glm::vec2(0.0f);
                }
			}
		}
	}

	if (!hit) {
		return -1.0f;
	}

	glm::vec3 objIntersect = objOrigin + closestT * objDir;
	intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objIntersect, 1.0f));
	normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(closestNormal, 0.0f)));
	uvOut = closestUV;
	return glm::length(ray.origin - intersectionPoint);
}

