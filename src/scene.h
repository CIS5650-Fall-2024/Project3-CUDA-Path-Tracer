#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <tiny_gltf.h>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string &jsonName);
    void loadFromGltf(const std::string &gltfName);

    void setupCamera(
        glm::ivec2 resolution = glm::ivec2(800, 800),
        glm::vec3 position = glm::vec3(0, 0, 1),
        glm::vec3 lookAt = glm::vec3(0, 0, 0),
        float fovy = 45,
        glm::vec3 up = glm::vec3(0, 1, 0),
        float lensSize = 0,
        float focalDist = 0);

    void createLightIndices();

    void loadGltfTexture(const tinygltf::Model &model, int textureId);
    void loadGltfMaterial(const tinygltf::Model &model, int materialId);
    void loadGltfMesh(const tinygltf::Model &model, int meshId);
    void loadGltfNode(const tinygltf::Model &model, int node);

public:
    Scene(string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;

    std::vector<Mesh> meshes;
    std::vector<int> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> uvs;
    std::vector<TextureData> texes;
    
    std::vector<Material> materials;

    std::vector<int> lightIndices;

    RenderState state;
    bool useDirectLighting;
};