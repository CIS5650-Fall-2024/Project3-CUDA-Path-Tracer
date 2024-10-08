#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "gltfLoader.h"
#include <memory>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfFilename);
public:
    Scene(string filename);
    ~Scene();

    void LoadMaterialsFromFromGLTF();
    void LoadTexturesFromGLTF();
    void loadGeometryFromGLTF();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::unique_ptr<GLTFLoader> gltfLoader;
    std::vector<GLTFTextureData> meshesTextures;
    std::vector<GLTFMaterialData> meshesMaterials;
    std::vector<glm::vec3> meshesPositions;
    std::vector<uint16_t> meshesIndices;
    std::vector<glm::vec3> meshesNormals;
    std::vector<glm::vec2> meshesUVs;
    BVHNode* topLevelBVH;
    int topLevelBVHCount;
    RenderState state;
};
