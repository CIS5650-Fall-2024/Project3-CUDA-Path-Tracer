#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"
using namespace std;

class Scene
{
private:
    ifstream fp_in;
	void loadFromGltf(const std::string& gltfName);
	void parsePrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive, Mesh& mesh);
	void buildBvh();
	void splitNode(const Mesh& mesh, int parentIdx, int depth);

public:
    Scene(string sceneFile, string envMapFile);
    ~Scene();
    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<Mesh> meshes;
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<Triangle> triangles;
	std::vector<Texture> textures;
	std::vector<glm::vec2> baseColorUvs;
	std::vector<glm::vec2> normalUvs;
	std::vector<glm::vec2> emissiveUvs;
	std::vector<BvhNode> bvhNodes;
	Texture envMap;
    RenderState state;
};
