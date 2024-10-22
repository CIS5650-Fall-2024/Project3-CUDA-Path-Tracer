#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_gltf.h>
#include <iostream>

using uint = unsigned int;

struct AABB {
    glm::vec3 min, max;
};

struct BVHNode {
    AABB bounds;
    int leftChild;
    int rightChild;
    glm::ivec4 triangleIDs;
};

struct MeshTriangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;


    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;  // UV coordinates for each vertex of the triangle

    int baseColorTexID;
    int materialIndex;
    int normalMapTexID;
};



struct CompareTriangles {
    int axis;  // Longest axis to sort on
    const std::vector<MeshTriangle> *triangles;

    CompareTriangles(int axis, const std::vector<MeshTriangle>* tris) : axis(axis), triangles(tris) {}

    glm::vec3 centroid(const MeshTriangle& tri) const {
        return (tri.v0 + tri.v1 + tri.v2) * 0.333333f;
    }

    bool operator()(int a, int b) const {
        glm::vec3 centroidA = centroid((*triangles)[a]);
        glm::vec3 centroidB = centroid((*triangles)[b]);

        switch (axis) {
        case 0: return centroidA.x < centroidB.x;
        case 1: return centroidA.y < centroidB.y;
        case 2: return centroidA.z < centroidB.z;
        default: return false;
        }
    }
};

class glTFLoader {
public:

    struct Mesh {
        std::vector<float> positions; // Stores x, y, z coordinates consecutively
        std::vector<uint32_t> indices;
        std::vector<float> uvs;
        std::vector<int> baseColorTextureIDs;
        std::vector<int> normalMapTextureIDs;
        std::vector<int> matIDs;
    };

    glTFLoader() {}

    bool loadModel(const std::string& filename);
    void extractTriangles() {
        triangles = std::make_unique<std::vector<MeshTriangle>>();
        for (const auto& mesh : meshes) {
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                MeshTriangle tri;
                tri.v0 = getVertex(mesh, mesh.indices[i]);
                tri.v1 = getVertex(mesh, mesh.indices[i + 1]);
                tri.v2 = getVertex(mesh, mesh.indices[i + 2]);

                tri.uv0 = getUV(mesh, mesh.indices[i]);
                tri.uv1 = getUV(mesh, mesh.indices[i + 1]);
                tri.uv2 = getUV(mesh, mesh.indices[i + 2]);

                if (mesh.baseColorTextureIDs.size() > 0) {
                    tri.baseColorTexID = getBaseColorTextureID(mesh, mesh.indices[i]);
                }
                else {
                    tri.baseColorTexID = -1;
                }
                if (mesh.matIDs.size() > 0) {
                    tri.materialIndex = getMatID(mesh, mesh.indices[i]);
                }

                
                if (mesh.normalMapTextureIDs.size() > 0) {
                    tri.normalMapTexID = getNormalMapTextureID(mesh, mesh.indices[i]);
                }
                else {
                    tri.normalMapTexID = -1;
                }

                triangles->push_back(tri);
            }
            primNum++;
        }
    }

    std::vector<MeshTriangle>* getTriangles() {
        if (triangles == nullptr) {
            extractTriangles();
            //std::cout << "PT0: TRIANGLE BUFFER INITIALIZED, GLTF SPACE, NOT WORLD SPACE YET.\n";
        }
        return triangles.get();
    }

    std::vector<tinygltf::Image> getImages() const {
        return images;
    }

    //get BVH
    std::vector<BVHNode> getBVHTree() {
        if (nodes.size() == 0) {
            buildBVH();
        }
        return nodes;
    }

private:
    unsigned int rootNodeIdx = 0;
    int nodesUsed = -1;
    std::vector<Mesh> meshes;
    std::vector<tinygltf::Image> images;
    std::vector<BVHNode> nodes;
    std::vector<int> BVHtriangleIndexBuffer;
    std::unique_ptr<std::vector<MeshTriangle>> triangles;
    std::vector<unsigned int> triIdx;

    int primNum = 0;

    void processNodes(const tinygltf::Model& model);
    void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentTransform);
    glm::mat4 getNodeTransform(const tinygltf::Node& node);
    void loadImages(const tinygltf::Model& model);
    void extractWorldSpaceTriangleBuffers(const tinygltf::Model& model, const tinygltf::Primitive& primitive, const glm::mat4& transform);

    glm::vec3 getVertex(const Mesh& mesh, uint32_t index) const {
        size_t i = index * 3;
        return { mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2] };
    }

    glm::vec2 getUV(const Mesh& mesh, uint32_t index) const {
        size_t i = index * 2;
        return { mesh.uvs[i], mesh.uvs[i + 1] };
    }
    int getBaseColorTextureID(const Mesh& mesh, uint32_t index) const {
        return mesh.baseColorTextureIDs[index];
    }
    int getNormalMapTextureID(const Mesh& mesh, uint32_t index) const {
        return mesh.normalMapTextureIDs[index];
    }

    int getMatID(const Mesh& mesh, uint32_t index) const {
        return mesh.matIDs[index];
    }


    //BVH

    AABB calculateBounds(int start, int end);
    int buildBVHRecursive(int start, int end, int depth);

    void buildBVH() {
        nodes.clear();
        nodes.resize(triangles->size() * 2 - 1);
        //std::cout << "calling rec. start is 0 and end is " << triangles->size() << "\n";

        //index buffer
        BVHtriangleIndexBuffer.clear();
        BVHtriangleIndexBuffer.resize(triangles->size());
        for (int i = 0; i < triangles->size(); i++) {
            BVHtriangleIndexBuffer[i] = i;
        }
        //std::cout << "index buffer used\n";
        rootNodeIdx = buildBVHRecursive(0, triangles->size(), 0);
        //std::cout << "PART 2: THE TRIANGLE BUFFER HAS BEEN MODIFIED DUE TO BVH CREATION" << "\n";
    }

    int longestAxis(const AABB& bounds);
};