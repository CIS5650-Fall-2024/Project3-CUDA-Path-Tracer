#pragma once

#include<deque>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utilities.h"
#include "intersections.h"

class BBox
{
public:
	BBox();

	BBox(const glm::vec3& p);

	BBox(const glm::vec3 minC, const glm::vec3 maxC);

	BBox(const BBox& bbox);

	BBox& operator=(const BBox& bbox);

	void expand(const BBox& bbox);

	void expand(const glm::vec3& p);

	glm::vec3 centroid() const;

	float surfaceArea() const;

	bool empty() const;

	__device__ bool intersect(const Ray& r, double& t0, double& t1) const;

	glm::vec3 maxC;

	glm::vec3 minC;

	glm::vec3 extent;
};

class BVHNode
{
public:
	BVHNode(BBox bbox);

	BVHNode(const BVHNode& node);

	BVHNode& operator=(const BVHNode& node);

	void setPrims(const std::vector<int>& pI);

	__device__ bool isLeaf() const;

	BBox bb;

	int leftNodeIndex;

	int rightNodeIndex;

	int escapseIndex;

	int p1I;

	int p2I;

	int p3I;

	int p4I;
};

BBox getBBox(const Primitive& prim);

int findSplitAxis(glm::vec3 cen);

int constructBVH(const std::vector<Primitive>& prims, const std::vector<int>& primsIndices, std::vector<BVHNode>& bvh, size_t maxLeafSize = 4);

void buidlStackless(std::vector<BVHNode>& bvh);

void setEscape(std::vector<BVHNode>& bvh, int nodeIndex, int es);

__device__ void intersectBVH(Ray& ray, ShadeableIntersection& intersection, Geom* geoms, Primitive* prims, BVHNode* bvh, int cur = 0);