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

__host__ __device__ bool aabbIntersectionTest(
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
		{
			float t1 = (box.mesh_aabb_min[xyz] - q.origin[xyz]) / qdxyz;
			float t2 = (box.mesh_aabb_max[xyz] - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);

			if (ta > 0 && ta > tmin) tmin = ta;
			if (tb < tmax) tmax = tb;
		}
	}
	return (tmax >= tmin && tmax > 0);
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
	if (!outside)
	{
		normal = -normal;
	}

	return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r, glm::vec3& intersectionPoint,
	glm::vec3& normal, glm::vec2& uv, bool& outside, Vertex* vertices)
{
	float t_min = FLT_MAX;
	int hit_face_index = -1;
	// Transform ray to object space
	Ray q;
	q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	// Get closest face hit
	for (int i = mesh.vertex_offset; i < mesh.vertex_offset + mesh.vertex_count; i += 3)
	{
		glm::vec3 weights;
		if (glm::intersectRayTriangle(q.origin, q.direction, vertices[i + 0].pos, vertices[i + 1].pos, vertices[i + 2].pos, weights)) {
			const auto edge1 = vertices[i + 1].pos - vertices[i + 0].pos;
			const auto edge2 = vertices[i + 2].pos - vertices[i + 0].pos;
			glm::vec3 intersection = vertices[i + 0].pos + weights.x * edge1 + weights.y * edge2;

			float t = glm::length(intersection - q.origin);
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_face_index = i;
			}
		}
	}

	// If we found a hit, compute intersection point and interpolated normal
	if (hit_face_index != -1) {
		glm::vec3 intersect = getPointOnRay(q, t_min);

		// Get normal using barycentric interpolation
		Vertex verts[3] = { vertices[hit_face_index + 0], vertices[hit_face_index + 1], vertices[hit_face_index + 2] };
		glm::vec3 edge_1, edge_2;
		float size;

		edge_1 = verts[1].pos - verts[0].pos;
		edge_2 = verts[2].pos - verts[1].pos;
		size = glm::length(glm::cross(edge_1, edge_2));

		normal = glm::vec3(0.0f);
		uv = glm::vec2(0.0f);
		for (int j = 0; j < 3; j++) {
			edge_1 = intersect - verts[(j + 1) % 3].pos;
			edge_2 = intersect - verts[(j + 2) % 3].pos;
			normal += glm::length(glm::cross(edge_1, edge_2)) * verts[j].norm;
			uv += glm::length(glm::cross(edge_1, edge_2)) * verts[j].uv / size;
		}

		// if the normal of vertex is not accurate, the direcly calculate every face norm
		if (mesh.vertex_count < 200) normal = glm::normalize(glm::cross(edge_1, edge_2));
		outside = glm::dot(normal, q.direction) < 0;
		if (!outside) {
			normal = -normal;
		}

		normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normal / size, 0.0f)));

		intersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersect, 1.0f));

		return glm::length(r.origin - intersectionPoint);
	}

	return -1;
}