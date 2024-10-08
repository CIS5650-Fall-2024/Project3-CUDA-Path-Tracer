#include "bvh.h"


// Compute the bounding box for a geometric object
AABB computeBoundingBox(const Geom& geom, const std::vector<Vertex>& vertices) {
    switch (geom.type) {
    case SPHERE: {
        // The bounding box of a sphere is centered around its translation with a half-width of its scale.
        glm::vec3 min = geom.translation - geom.scale.x; // Assuming uniform scaling for the sphere
        glm::vec3 max = geom.translation + geom.scale.x;
        return AABB(min, max);
    }
    case CUBE: {
        // The bounding box of a cube is defined by its transformed vertices.
        glm::vec3 min = geom.translation - geom.scale * 0.5f;
        glm::vec3 max = geom.translation + geom.scale * 0.5f;
        return AABB(min, max);
    }
    case OBJ: {
        // Compute bounding box based on OBJ vertices
        AABB bbox;
        for (const auto& vertex : vertices) {
            glm::vec4 worldPos = geom.transform * glm::vec4(vertex.pos, 1.0f);
            bbox.min = glm::min(bbox.min, glm::vec3(worldPos));
            bbox.max = glm::max(bbox.max, glm::vec3(worldPos));
        }
        return bbox;
    }
    default:
        // Return an empty bounding box for unsupported types
        return AABB();
    }
}


// Public method to build the BVH for a list of triangles
void BVH::buildBVH(std::vector<Triangle>& triangles) {
    nodes.clear();
    nodes.push_back(BVH_Node()); // Initialize the root node
    subdivide(0, triangles, 0, triangles.size());
}




// Subdivide the BVH nodes recursively
void BVH::subdivide(int nodeIdx, std::vector<Triangle>& triangles, int start, int end) {
    BVH_Node& node = nodes[nodeIdx];

    // Compute the bounding box for this node
    AABB bbox;
    for (int i = start; i < end; ++i) {
        bbox.Union(AABB(triangles[i].vertices[0], triangles[i].vertices[1], triangles[i].vertices[2]));
    }
    node.aabb = bbox;

    // If we are at a leaf node (less than a threshold of triangles), stop subdividing
    int numTriangles = end - start;
    if (numTriangles <= 1) {
        node.firstT_idx = start;
        node.TCount = numTriangles;
        return;
    }

    // Determine the axis to split on using the longest axis of the bounding box
    int splitAxis = bbox.LongestAxisIndex();
    node.splitAxis = splitAxis;

    // Use the surface area heuristic (SAH) to find the optimal split position
    int mid = findOptimalSplit(triangles, start, end, splitAxis, bbox);

    // Create child nodes
    node.leftChild = nodes.size();
    nodes.push_back(BVH_Node());
    subdivide(node.leftChild, triangles, start, mid);

    node.rightChild = nodes.size();
    nodes.push_back(BVH_Node());
    subdivide(node.rightChild, triangles, mid, end);
}




// Find the optimal split using the surface area heuristic (SAH)
int BVH::findOptimalSplit(std::vector<Triangle>& triangles, int start, int end, int axis, const AABB& parentAABB) {
    // Sort triangles based on their centroids along the chosen axis
    std::sort(triangles.begin() + start, triangles.begin() + end,
        [axis](const Triangle& a, const Triangle& b) {
            glm::vec3 centroidA = (a.vertices[0] + a.vertices[1] + a.vertices[2]) / 3.0f;
            glm::vec3 centroidB = (b.vertices[0] + b.vertices[1] + b.vertices[2]) / 3.0f;
            return centroidA[axis] < centroidB[axis];
        });

    // Use the surface area heuristic to find the best split point
    float bestCost = FLT_MAX;
    int bestSplit = start;
    for (int i = start; i < end - 1; ++i) {
        AABB leftAABB, rightAABB;
        for (int j = start; j <= i; ++j) {
            leftAABB.Union(AABB(triangles[j].vertices[0], triangles[j].vertices[1], triangles[j].vertices[2]));
        }
        for (int j = i + 1; j < end; ++j) {
            rightAABB.Union(AABB(triangles[j].vertices[0], triangles[j].vertices[1], triangles[j].vertices[2]));
        }

        float leftSurfaceArea = leftAABB.SurfaceArea();
        float rightSurfaceArea = rightAABB.SurfaceArea();
        float cost = leftSurfaceArea * (i - start + 1) + rightSurfaceArea * (end - i - 1);

        if (cost < bestCost) {
            bestCost = cost;
            bestSplit = i;
        }
    }

    return bestSplit + 1; // Split point
}
