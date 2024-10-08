#pragma once
#include <string>
#include <vector>
#include <memory>
#include "Math.h"
#include "glm/glm.hpp"

#define __STDC_LIB_EXT1__
#include "../external/include/tiny_gltf.h"

using namespace std;

// Loaded data of a single primitive.
struct GLTFPrimitiveData {
    vector<glm::vec3> positions;
    vector<uint16_t> indices;
    vector<glm::vec3> normals;
    vector<glm::vec2> uvs;
    int texture;
    int material;
};

struct GLTFTextureData {
    string uri;
};

struct GLTFMaterialData {
    int baseColorMap;
    int normalMap;
    int metallicRoughnessMap;
};

class GLTFLoader {
public:
    GLTFLoader(string& filename);

    static GLTFPrimitiveData Load(string &filename);

    void LoadModel();

    unsigned int getPrimitiveCount(int nodeIdx = 0) const;

    unsigned int GLTFLoader::getNodeCount() const;

    // Loads a single primitive from the specified node.
    GLTFPrimitiveData LoadPrimitive(int nodeIdx, int primitiveIdx) const;

    // Loads all textures.
    vector<GLTFTextureData> LoadTextures();

    // Loads all materials.
    vector<GLTFMaterialData> LoadMaterials(
        const std::vector<GLTFTextureData> &textures
    );

private:
    string mFilename;
    string mAssetsDirectory;

    tinygltf::Model mModel;

    void LoadPrimitiveIndices(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
    ) const;

    void LoadPrimitivePositions(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
    ) const;

    void LoadPrimitiveNormals(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
    ) const;

    void LoadPrimitiveUVs(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
    ) const;

    void LoadPrimitiveTexture(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
    ) const;

    void LoadPrimitiveMaterial(
      tinygltf::Primitive& primitive,
      GLTFPrimitiveData& primitiveData
    ) const;
};