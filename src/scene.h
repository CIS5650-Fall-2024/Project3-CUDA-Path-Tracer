#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "image.h"
#include <tiny_gltf.h>
#include <stb_image.h>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    template <typename T>
    void populateTriangleData(T* indices, tinygltf::Model& model, tinygltf::Accessor& accessor, const tinygltf::Primitive& primitive, float* positions, Geom& mesh) 
	{
		for (size_t i = 0; i < accessor.count; i += 3) {
			Triangle tri;
			tri.meshId = mesh.meshid;
			// vertex positions
			tri.v1.pos = glm::vec3(mesh.transform * glm::vec4(positions[indices[i] * 3], positions[indices[i] * 3 + 1], positions[indices[i] * 3 + 2], 1));
			tri.v2.pos = glm::vec3(mesh.transform * glm::vec4(positions[indices[i + 1] * 3], positions[indices[i + 1] * 3 + 1], positions[indices[i + 1] * 3 + 2], 1));
			tri.v3.pos = glm::vec3(mesh.transform * glm::vec4(positions[indices[i + 2] * 3], positions[indices[i + 2] * 3 + 1], positions[indices[i + 2] * 3 + 2], 1));
			tri.centroid = (tri.v1.pos + tri.v2.pos + tri.v3.pos) / 3.0f;

			// vertex normals
			auto normals = primitive.attributes.find("NORMAL");
			if (normals != primitive.attributes.end()) {
				int norAccessorIndex = primitive.attributes.at("NORMAL");
				tinygltf::Accessor& norAccessor = model.accessors[norAccessorIndex];
				tinygltf::BufferView& norBufferView = model.bufferViews[norAccessor.bufferView];
				tinygltf::Buffer& normalBuffer = model.buffers[norBufferView.buffer];
				float* normals = reinterpret_cast<float*>(&(normalBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset]));

				tri.v1.nor = glm::normalize(glm::vec3(mesh.invTranspose * glm::vec4(normals[indices[i] * 3], normals[indices[i] * 3 + 1], normals[indices[i] * 3 + 2], 0)));
				tri.v2.nor = glm::normalize(glm::vec3(mesh.invTranspose * glm::vec4(normals[indices[i + 1] * 3], normals[indices[i + 1] * 3 + 1], normals[indices[i + 1] * 3 + 2], 0)));
				tri.v3.nor = glm::normalize(glm::vec3(mesh.invTranspose * glm::vec4(normals[indices[i + 2] * 3], normals[indices[i + 2] * 3 + 1], normals[indices[i + 2] * 3 + 2], 0)));

				mesh.usesNormals = true;
			}

			// vertex uvs
			auto uvs = primitive.attributes.find("TEXCOORD_0");
			if (uvs != primitive.attributes.end()) {
				int uvAccessorIndex = primitive.attributes.at("TEXCOORD_0");
				tinygltf::Accessor& uvAccessor = model.accessors[uvAccessorIndex];
				tinygltf::BufferView& uvBufferView = model.bufferViews[uvAccessor.bufferView];
				tinygltf::Buffer& uvBuffer = model.buffers[uvBufferView.buffer];
				float* uvs = reinterpret_cast<float*>(&(uvBuffer.data[uvBufferView.byteOffset + uvAccessor.byteOffset]));

				tri.v1.uv = glm::vec2(uvs[indices[i] * 2], uvs[indices[i] * 2 + 1]);
				tri.v2.uv = glm::vec2(uvs[indices[i + 1] * 2], uvs[indices[i + 1] * 2 + 1]);
				tri.v3.uv = glm::vec2(uvs[indices[i + 2] * 2], uvs[indices[i + 2] * 2 + 1]);

				mesh.usesUVs = true;
			}

			triangles.push_back(tri);

		}
	}


    void loadGLTFMesh(const std::string& filename, Geom& newGeom);

	void updateNodeBounds(int);
	void subdivideBounds(int);
	void buildBVH();

    std::vector<Geom> geoms;
	std::vector<Triangle> triangles;

    std::vector<Material> materials;
	std::vector<Texture> textures;
    std::vector<glm::vec3> textureData;
	std::unique_ptr<Texture> envMap;
	std::vector<glm::vec4> envMapData;

	// For BVH
	int nodesUsed = 1;
	std::vector<BVHNode> bvhNodes;


	struct aabb
	{
		glm::vec3 bmin = glm::vec3(FLT_MAX), bmax = glm::vec3(FLT_MIN);
		void grow(glm::vec3 p) { 
			bmin = glm::vec3(glm::min(bmin.x, p.x), glm::min(bmin.y, p.y), glm::min(bmin.z, p.z));
			bmax = glm::vec3(glm::max(bmax.x, p.x), glm::max(bmax.y, p.y), glm::max(bmax.z, p.z));
		}
		float area()
		{
			glm::vec3 e = bmax - bmin; // box extent
			return e.x * e.y + e.y * e.z + e.z * e.x;
		}
	};

	struct Bin { aabb bounds; int triCount = 0; };

	float EvaluateSAH(BVHNode& node, int axis, float pos);
	float FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos);

    RenderState state;
};