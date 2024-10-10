#include "bvh.h"

// BBox class
BBox::BBox() : minC(glm::vec3(INF_F)), maxC(glm::vec3(-INF_F))
{
	extent = maxC - minC;
}

BBox::BBox(const glm::vec3& p) : minC(p), maxC(p)
{
	extent = maxC - minC;
}

BBox::BBox(const glm::vec3 min, const glm::vec3 max) : minC(min), maxC(max), extent(max - min) {}

BBox::BBox(const BBox& bbox) : minC(bbox.minC), maxC(bbox.maxC), extent(extent) {}

BBox& BBox::operator=(const BBox& bbox)
{
	minC = bbox.minC;
	maxC = bbox.maxC;
	extent = bbox.extent;

	return *this;
}

void BBox::expand(const BBox& bbox) 
{
	minC.x = std::min(minC.x, bbox.minC.x);
	minC.y = std::min(minC.y, bbox.minC.y);
	minC.z = std::min(minC.z, bbox.minC.z);
	maxC.x = std::max(maxC.x, bbox.maxC.x);
	maxC.y = std::max(maxC.y, bbox.maxC.y);
	maxC.z = std::max(maxC.z, bbox.maxC.z);
	extent = maxC - minC;
}

void BBox::expand(const glm::vec3& p)
{
	minC.x = std::min(minC.x, p.x);
	minC.y = std::min(minC.y, p.y);
	minC.z = std::min(minC.z, p.z);
	maxC.x = std::max(maxC.x, p.x);
	maxC.y = std::max(maxC.y, p.y);
	maxC.z = std::max(maxC.z, p.z);
	extent = maxC - minC;
}

glm::vec3 BBox::centroid() const
{
	return (minC + maxC) / 2.0f;
}

float BBox::surfaceArea() const
{
	if (empty())
	{
		return 0.0f;
	}
	else 
	{
		return 2 * (extent.x * extent.z + extent.x * extent.y + extent.y * extent.z);
	}
}

bool BBox::empty() const
{
	return minC.x > maxC.x || minC.y > maxC.y || minC.z > maxC.z;
}

__device__ bool BBox::intersect(const Ray& r, double& t0, double& t1) const
{
	glm::vec3 testMinC = minC;
	double tXMin = (minC.x - r.origin.x) / r.direction.x;
	double tXMax = (maxC.x - r.origin.x) / r.direction.x;
	if (tXMax < tXMin)
	{
		double tmp = tXMin;
		tXMin = tXMax;
		tXMax = tmp;
	}

	double tYMin = (minC.y - r.origin.y) / r.direction.y;
	double tYMax = (maxC.y - r.origin.y) / r.direction.y;
	if (tYMax < tYMin)
	{
		double tmp = tYMin;
		tYMin = tYMax;
		tYMax = tmp;
	}

	double tZMin = (minC.z - r.origin.z) / r.direction.z;
	double tZMax = (maxC.z - r.origin.z) / r.direction.z;
	if (tZMax < tZMin)
	{
		double tmp = tZMin;
		tZMin = tZMax;
		tZMax = tmp;
	}

	double tmin = fmaxf(fmaxf(tXMin, tYMin), tZMin);
	double tmax = fminf(fminf(tXMax, tYMax), tZMax);

	if (tmin > tmax)
	{
		return false;
	}

	if ((tmin >= t0) && (tmax <= t1))
	{
		t0 = tmin;
		t1 = tmax;
	}

	return (t0 < r.tmax) && (t1 > r.tmin);
}

// BVHNode class
BVHNode::BVHNode(BBox bbox) : bb(bbox), leftNodeIndex(-1), rightNodeIndex(-1), escapseIndex(-1), p1I(-1), p2I(-1), p3I(-1), p4I(-1) {}

BVHNode::BVHNode(const BVHNode& node) 
{
	bb = node.bb;
	leftNodeIndex = node.leftNodeIndex;
	rightNodeIndex = node.rightNodeIndex;
	escapseIndex = node.escapseIndex;
	p1I = node.p1I;
	p2I = node.p2I;
	p3I = node.p3I;
	p4I = node.p4I;
}

BVHNode& BVHNode::operator=(const BVHNode& node)
{
	if (this != &node) 
	{
		bb = node.bb;
		leftNodeIndex = node.leftNodeIndex;
		rightNodeIndex = node.rightNodeIndex;
		escapseIndex = node.escapseIndex;
		p1I = node.p1I;
		p2I = node.p2I;
		p3I = node.p3I;
		p4I = node.p4I;
	}

	return *this;
}

void BVHNode::setPrims(const std::vector<int>& pI) 
{
	int pSize = pI.size();

	if (pSize >= 1) 
	{
		p1I = pI[0];
	}

	if (pSize >= 2)
	{
		p2I = pI[1];
	}

	if (pSize >= 3)
	{
		p3I = pI[2];
	}

	if (pSize == 4)
	{
		p4I = pI[3];
	}
}

__device__ bool BVHNode::isLeaf() const
{
	return p1I >= 0;
}

// BVH Construction Helper
BBox getBBox(const Primitive& prim) 
{
	if (prim.type == TRIANGLE) 
	{
		BBox bbox(prim.p1);
		bbox.expand(prim.p2);
		bbox.expand(prim.p3);
		return bbox;
	}
	else if (prim.type == SPHEREP)
	{
		return BBox(prim.p2, prim.p3);
	}
	else if (prim.type == CUBEP)
	{
		return BBox(prim.p2, prim.p3);
	}
}

int findSplitAxis(glm::vec3 cen) 
{
	int axis;

	if (cen.x >= cen.y)
	{
		if (cen.x >= cen.z)
		{
			axis = 0;
		}
		else
		{
			axis = 2;
		}
	}
	else
	{
		if (cen.y >= cen.z)
		{
			axis = 1;
		}
		else
		{
			axis = 2;
		}
	}

	return axis;
}

// Construct the BVH for the primitives with indices in primsIndices, store the BVHNodes in bvhs, and return the index of the root BVHNode
int constructBVH(const std::vector<Primitive>& prims, const std::vector<int>& primsIndices, std::vector<BVHNode>& bvh, size_t maxLeafSize)
{
	// Populate all primitives
	BBox cetroidBox, bbox;
	for (int i : primsIndices)
	{
		BBox bb = getBBox(prims[i]);
		bbox.expand(bb);
		cetroidBox.expand(bb.centroid());
	}

	// Construct root BVHNode
	BVHNode node(bbox);
	int curIndex = bvh.size();
	bvh.push_back(node);

	if (primsIndices.size() <= maxLeafSize)
	{
		bvh[curIndex].setPrims(primsIndices);
	}
	else 
	{
		int axis = findSplitAxis(cetroidBox.extent);
		float splitPoint = cetroidBox.centroid()[axis];

		double left = 0, right = 0;
		std::vector<int> leftIndices, rightIndices;

		while (leftIndices.empty() || rightIndices.empty()) 
		{
			for (int j : primsIndices)
			{
				float pCentroidAxis = getBBox(prims[j]).centroid()[axis];

				if (pCentroidAxis >= splitPoint) 
				{
					rightIndices.push_back(j);
					right += pCentroidAxis;
				}
				else 
				{
					leftIndices.push_back(j);
					left += pCentroidAxis;
				}
			}

			if (leftIndices.empty()) 
			{
				splitPoint = right / rightIndices.size();
				right = 0;
				rightIndices.clear();
			}
			else if (rightIndices.empty())
			{
				splitPoint = left / leftIndices.size();
				left = 0;
				leftIndices.clear();
			}
			else 
			{
				int rIndex = constructBVH(prims, rightIndices, bvh);;
				int lIndex = constructBVH(prims, leftIndices, bvh);
				bvh[curIndex].rightNodeIndex = rIndex;
				bvh[curIndex].leftNodeIndex = lIndex;
			}
		}
	}
	return curIndex;
}

void buidlStackless(std::vector<BVHNode>& bvh) 
{
	BVHNode& root = bvh[0];
	bvh[root.leftNodeIndex].escapseIndex = root.rightNodeIndex;
	setEscape(bvh, root.leftNodeIndex, root.rightNodeIndex);
	setEscape(bvh, root.rightNodeIndex, -1);
}

void setEscape(std::vector<BVHNode>& bvh, int nodeIndex, int es)
{
	BVHNode& node = bvh[nodeIndex];
	if (node.leftNodeIndex != -1 && node.rightNodeIndex != -1)
	{
		bvh[node.leftNodeIndex].escapseIndex = node.rightNodeIndex;
		bvh[node.rightNodeIndex].escapseIndex = es;
		setEscape(bvh, node.leftNodeIndex, node.rightNodeIndex);
		setEscape(bvh, node.rightNodeIndex, es);
	}
}

__device__ void intersectBVH(Ray& ray, ShadeableIntersection& intersection, Geom* geoms, Primitive* prims, BVHNode* bvh, int cur)
{
	bool hit = false;

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	while (cur != -1) 
	{
		BVHNode& node = bvh[cur];

		double t0 = ray.tmin, t1 = ray.tmax;
		if (node.bb.intersect(ray, t0, t1))
		{
			if (node.isLeaf())
			{
				int primIndices[4] = { node.p1I, node.p2I, node.p3I, node.p4I };

				for (int i = 0; i < 4; i++)
				{
					if (primIndices[i] > -1)
					{
						Primitive& p = prims[primIndices[i]];

						if (p.type == SPHEREP)
						{
							t = sphereIntersectionTest(geoms[p.geomId], ray, tmp_intersect, tmp_normal, outside);
						}
						else if (p.type == CUBEP)
						{
							t = boxIntersectionTest(geoms[p.geomId], ray, tmp_intersect, tmp_normal, outside);
						}
						else if (p.type == TRIANGLE)
						{
							t = triangleIntersectionTest(geoms[p.geomId], p, ray, tmp_intersect, tmp_normal, outside);
						}
						else 
						{
							t = -1;
						}

						if (t > 0.0f && t_min > t)
						{
							t_min = t;
							hit_geom_index = primIndices[i];
							intersect_point = tmp_intersect;
							normal = tmp_normal;
						}
					}
				}
			}
			else 
			{
				cur = node.leftNodeIndex;
				continue;
			}
		}
		cur = node.escapseIndex;
	}

	if (hit_geom_index == -1)
	{
		intersection.t = -1.0f;
	}
	else
	{
		// The ray hits something
		intersection.t = t_min;
		intersection.materialId = geoms[prims[hit_geom_index].geomId].materialid;
		intersection.surfaceNormal = normal;
	}
}