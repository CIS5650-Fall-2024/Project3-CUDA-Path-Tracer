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

__host__ __device__ glm::vec3 barycentric(glm::vec3 p, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3) {
    glm::vec3 edge1 = t2 - t1;
    glm::vec3 edge2 = t3 - t2;
    float S = length(cross(edge1, edge2));

    edge1 = p - t2;
    edge2 = p - t3;
    float S1 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t3;
    float S2 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t2;
    float S3 = length(cross(edge1, edge2));

    return glm::vec3(S1 / S, S2 / S, S3 / S);
}

__host__ __device__ float triangleIntersectionTest(
	glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
    Geom geom, Triangle tri,
	Ray r,
	glm::vec3& intersectionPoint,
	glm::vec3& normal,
	glm::vec3& tangent,
	glm::vec3& bitangent,
    glm::vec2& uv,
    //glm::mat3& TBN,
	bool& outside)
{
    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
    glm::vec2 deltaUV1 = tri.uvs[1] - tri.uvs[0];
    glm::vec2 deltaUV2 = tri.uvs[2] - tri.uvs[0];
	glm::vec3 h = cross(rt.direction, e2);
    float a = glm::dot(e1, h);

	if (a > -0.00001f && a < 0.00001f) // parallel
	{
		return -1;
	}

	float f = 1.0f / a;
	glm::vec3 s = rt.origin - v0;
	float u = f * glm::dot(s, h);
	if (u < 0.0f || u > 1.0f)
	{
		return -1;
	}
	glm::vec3 q = cross(s, e1);
	float v = f * glm::dot(rt.direction, q);
	if (v < 0.0f || u + v > 1.0f)
	{
		return -1;
	}

	float t = f * glm::dot(e2, q);
    float f2 = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

	if (t > 0.00001f)
	{
		
        glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
		intersectionPoint = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.f));
		normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(objspaceIntersection, 0.f)));     
        //glm::vec3 bary = barycentric(objspaceIntersection, v0, v1, v2);
        glm::vec2 triUV0 = tri.uvs[0];
        glm::vec2 triUV1 = tri.uvs[1];
        glm::vec2 triUV2 = tri.uvs[2];
        uv = (1.0f - u - v) * tri.uvs[0] + u * tri.uvs[1] +v * tri.uvs[2];

        // compute TBN
        //glm::vec3 tangent;
        tangent.x = f * (deltaUV2.y * e1.x - deltaUV1.y * e2.x);
        tangent.y = f * (deltaUV2.y * e1.y - deltaUV1.y * e2.y);
        tangent.z = f * (deltaUV2.y * e1.z - deltaUV1.y * e2.z);
        tangent = glm::normalize(tangent);

        //glm::vec3 bitangent;
        bitangent.x = f * (-deltaUV2.x * e1.x + deltaUV1.x * e2.x);
        bitangent.y = f * (-deltaUV2.x * e1.y + deltaUV1.x * e2.y);
        bitangent.z = f * (-deltaUV2.x * e1.z + deltaUV1.x * e2.z);
        bitangent = glm::normalize(bitangent);

        // Gram-Schmidt to orthogonalize
        tangent = glm::normalize(tangent - glm::dot(tangent, normal) * normal);
        bitangent = glm::cross(normal, tangent);
        bitangent = glm::normalize(bitangent);

        //TBN = glm::mat3(tangent, bitangent, normal);

        
		outside = true;
		return glm::length(r.origin - intersectionPoint);
	}
    else
	{
		return -1;
	}
}

__host__ __device__ float meshIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
	glm::vec3& tangent,
	glm::vec3& bitangent,
    glm::vec2& uv,
    //glm::mat3& TBN,
    bool& outside
) {
    float closestT = FLT_MAX;
    int hitTriangle = -1;
    int num_triangles = geom.mesh->num_triangles;
	Mesh* mesh = geom.mesh;

    for (int i = 0; i < num_triangles; ++i) {
        const Triangle& tri = geom.mesh->triangles[i];
        glm::vec3 tempIntersectionPoint, tempNormal, tempTangent, tempBitangent;
        glm::vec2 tempUV;
        glm::mat3 tempTBN;
        bool tempOutside;

        glm::vec3 p0 = tri.points[0];
		float p0x = p0.x;
		float p0y = p0.y;
		float p0z = p0.z;

		glm::vec3 p1 = tri.points[1];
		glm::vec3 p2 = tri.points[2];

        float t = 0.0;
        
        t = triangleIntersectionTest(
            tri.points[0], tri.points[1], tri.points[2],
            geom, tri,
            r,
            tempIntersectionPoint,
            tempNormal,
			tempTangent,
			tempBitangent,
            tempUV,
            //tempTBN,
            tempOutside
        );
        

        if (t > 0 && t < closestT) {
            closestT = t;
            intersectionPoint = tempIntersectionPoint;
            normal = tempNormal;
			tangent = tempTangent;
			bitangent = tempBitangent;
            uv = tempUV;
            //TBN = tempTBN;
            outside = tempOutside;
            hitTriangle = i;
        }
    }

    return (hitTriangle != -1) ? closestT : -1.0f;
}
