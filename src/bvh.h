#pragma once
#include "utilities.h"
#include "sceneStructs.h"
#include "memoryArena.h"
#include <stack>
#include <queue>
#include "PTDirectives.h"
class BVHAccel
{
public:
	std::vector<Triangle*> primitives;
	const int maxPrimsInNode;
	int bvhNodes;
	BVHAccel(std::vector<Triangle>& triangles, int numTriangles, int maxPrimsInNode = 4) : maxPrimsInNode(maxPrimsInNode), bvhNodes(0)
	{
		primitives.reserve(numTriangles);
		for (int i = 0; i < numTriangles; ++i)
		{
			primitives.push_back(&triangles[i]);
		}
		LinearBVHNode* nodes = nullptr;
	}

	struct BVHPrimitiveInfo
	{
		int primitiveNumber;
		glm::vec3 centroid;
		AABB bounds;
	};

	struct MortonPrimitive {
		int primitiveIndex;
		uint32_t mortonCode;
	};


	struct BVHBuildNode
	{
		AABB bounds;
		BVHBuildNode* children[2];
		int splitAxis;
		int firstPrimOffset;
		int nPrimitives;

		BVHBuildNode() : bounds(), children{ nullptr, nullptr }, splitAxis(0), firstPrimOffset(0), nPrimitives(0) {}

		void initLeaf(int first, int n, const AABB& b)
		{
			firstPrimOffset = first;
			nPrimitives = n;
			bounds = b;
			children[0] = children[1] = nullptr;
		}
		void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1)
		{
			children[0] = c0;
			children[1] = c1;
			bounds = AABB::Union(c0->bounds, c1->bounds);
			splitAxis = axis;
			nPrimitives = 0;
		}

	};

	struct LBVHTreelet {
		int startIndex, nPrimitives;
		BVHBuildNode* buildNodes;
	};

	struct LinearBVHNode {
		AABB bounds;
		union {
			int primitivesOffset;    // leaf
			int secondChildOffset;   // interior
		};
		uint16_t nPrimitives;  // 0 -> interior node
		uint8_t axis;          // interior node: xyz
		uint8_t pad[1];        // ensure 32 byte total size
	};


	static __inline__ __host__ __device__ uint32_t LeftShift3(uint32_t x) {
		if (x == (1 << 10)) --x;
		x = (x | (x << 16)) & 0b00000011000000000000000011111111;
		x = (x | (x << 8)) & 0b00000011000000001111000000001111;
		x = (x | (x << 4)) & 0b00000011000011000011000011000011;
		x = (x | (x << 2)) & 0b00001001001001001001001001001001;
		return x;
	}

	static __inline__ __host__ __device__ uint32_t EncodeMorton3(const glm::vec3& v) {
		return (LeftShift3(v.z) << 2) | (LeftShift3(v.y) << 1) |
			LeftShift3(v.x);
	}

	static void RadixSort(std::vector<MortonPrimitive>* v);

	BVHBuildNode* recursiveBuild(MemoryArena& arena,
		std::vector<BVHPrimitiveInfo>& primitiveInfo, int start,
		int end, int* totalNodes,
		std::vector<Triangle*>& orderedPrims); 

	BVHBuildNode* HLBVHBuild(MemoryArena& arena,
		const std::vector<BVHPrimitiveInfo>& primitiveInfo,
		int* totalNodes,
		std::vector<Triangle*>& orderedPrims) const;

	void updateMortonCodes(std::vector<MortonPrimitive>& mortonPrims, const std::vector<BVHPrimitiveInfo>& primitiveInfo, AABB& bounds, int chunkSize) const;

	BVHBuildNode* emitLBVH(BVHBuildNode*& buildNodes,
		const std::vector<BVHPrimitiveInfo>& primitiveInfo,
		MortonPrimitive* mortonPrims, int nPrimitives, int* totalNodes,
		std::vector<Triangle*>& orderedPrims,
		std::atomic<int>* orderedPrimsOffset, int bitIndex, int recursionDepth = 0) const;

	BVHBuildNode *buildUpperSAH(MemoryArena &arena,
    std::vector<BVHBuildNode *> &treeletRoots, int start, int end,
    int *totalNodes, int recursionDepth = 0) const;

	int flattenBVHTree(BVHBuildNode* node, int* offset, int maxNodeNumber);

	void build(std::vector<Triangle>& trangles, int numTriangles);

	LinearBVHNode* nodes = nullptr;

	void traverseBVH(BVHBuildNode* node, int* nodeTraversed,int depth = 0);
	void traverseLBVH(BVHAccel::LinearBVHNode* node, int totalNodes, int depth = 0);
};
using LinearBVHNode = BVHAccel::LinearBVHNode;
extern LinearBVHNode* dev_nodes;

bool __device__ BVHIntersect(const Ray& ray, LinearBVHNode* dev_nodes, Triangle* dev_triangles, ShadeableIntersection* isect = NULL);