// Need to be defined only once. See github.com/syoyo/tinygltf#loading-gltf-20-model.
#define TINYGLTF_IMPLEMENTATION
// Already defined in stb.cpp. Can only be defined for one of the inclusions of stb_image_write.h.
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "gltfLoader.h"
#include "FileSystem.h"

GLTFLoader::GLTFLoader(string &filename) 
    : mFilename(filename),
      mAssetsDirectory(FileSystem::GetDirectory(filename))
{}

unsigned int GLTFLoader::getPrimitiveCount(int nodeIdx) const {
    const tinygltf::Scene &scene = mModel.scenes[mModel.defaultScene];
    auto &node = mModel.nodes[scene.nodes[nodeIdx]];
    if (node.mesh < 0) return 0;
    auto &mesh = mModel.meshes[node.mesh];
    return mesh.primitives.size();
}

unsigned int GLTFLoader::getNodeCount() const {
    return mModel.nodes.size();
}

GLTFPrimitiveData GLTFLoader::Load(string &filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    string err;
    string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }
    if (!ret) {
        printf("Failed to parse glTF\n");
    }

    GLTFPrimitiveData loadedData;

    const tinygltf::Scene &scene = model.scenes[model.defaultScene];

    for (int j = 0; j < scene.nodes.size(); ++j) {
        auto &node = model.nodes[scene.nodes[j]];
        if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
            auto &mesh = model.meshes[node.mesh];

            const size_t index_remap[] = {0, 2, 1};

            for (size_t i = 0; i < mesh.primitives.size(); ++i) {
                tinygltf::Primitive primitive = mesh.primitives[i];

                tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer &indexBuffer = model.buffers[indexBufferView.buffer];
                const uint8_t *indexes = indexBuffer.data.data() + indexBufferView.byteOffset;
                int stride = indexAccessor.ByteStride(indexBufferView);

                loadedData.indices.resize(indexAccessor.count);

                int indexOffset = 0;
                int vertexOffset = 0;

                if (stride == 1) {
                    for (size_t i = 0; i < indexAccessor.count; i += 3) {
                        loadedData.indices[indexOffset + i + 0] = vertexOffset + indexes[i + index_remap[0]];
                        loadedData.indices[indexOffset + i + 1] = vertexOffset + indexes[i + index_remap[1]];
                        loadedData.indices[indexOffset + i + 2] = vertexOffset + indexes[i + index_remap[2]];
                    }
                } else if (stride == 2) {
                    for (size_t i = 0; i < indexAccessor.count; i += 3) {
                        loadedData.indices[indexOffset + i + 0] = vertexOffset + ((uint16_t *)indexes)[i + index_remap[0]];
                        loadedData.indices[indexOffset + i + 1] = vertexOffset + ((uint16_t *)indexes)[i + index_remap[1]];
                        loadedData.indices[indexOffset + i + 2] = vertexOffset + ((uint16_t *)indexes)[i + index_remap[2]];
                    }
                } else if (stride == 4) {
                    for (size_t i = 0; i < indexAccessor.count; i += 3) {
                        loadedData.indices[indexOffset + i + 0] = vertexOffset + ((uint32_t *)indexes)[i + index_remap[0]];
                        loadedData.indices[indexOffset + i + 1] = vertexOffset + ((uint32_t *)indexes)[i + index_remap[1]];
                        loadedData.indices[indexOffset + i + 2] = vertexOffset + ((uint32_t *)indexes)[i + index_remap[2]];
                    }
                } else {
                    assert(0 && "unsupported index stride!");
                }

                const tinygltf::Accessor &accessor = model.accessors[primitive.attributes["POSITION"]];
                const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
                stride = accessor.ByteStride(bufferView);
                const uint8_t *data = buffer.data.data() + accessor.byteOffset + bufferView.byteOffset;

                for (size_t i = 0; i < accessor.count; ++i) {
                    loadedData.positions.push_back(*(const glm::vec3 *)(data + i * stride));
                }
            }
        }
    }

    return loadedData;
}

void GLTFLoader::LoadModel() {
    tinygltf::TinyGLTF loader;

    string err;
    string warn;

    bool ret = loader.LoadASCIIFromFile(&mModel, &err, &warn, mFilename);
    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }
    if (!ret) {
        printf("Failed to parse glTF\n");
    }
}

// Loads a single primitive from the specified node.
GLTFPrimitiveData GLTFLoader::LoadPrimitive(int nodeIdx, int primitiveIdx) const {
    GLTFPrimitiveData primitiveData;

    const tinygltf::Scene &scene = mModel.scenes[mModel.defaultScene];
    auto &node = mModel.nodes[scene.nodes[nodeIdx]];
    auto &mesh = mModel.meshes[node.mesh];

    // The primitive to load.
    tinygltf::Primitive primitive = mesh.primitives[primitiveIdx];

    // Load indices.
    LoadPrimitiveIndices(primitive, primitiveData);

    // Load vertex positions.
    LoadPrimitivePositions(primitive, primitiveData);

    // Load vertex normals.
    LoadPrimitiveNormals(primitive, primitiveData);

    // Load vertex UVs.
    LoadPrimitiveUVs(primitive, primitiveData);

    // Load material.
    LoadPrimitiveMaterial(primitive, primitiveData);

    return primitiveData;
}

void GLTFLoader::LoadPrimitiveIndices(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
) const {
    const size_t indexRemap[] = {0, 1, 2};
    tinygltf::Accessor indexAccessor = mModel.accessors[primitive.indices];
    const tinygltf::BufferView &indexBufferView = mModel.bufferViews[indexAccessor.bufferView];
    const tinygltf::Buffer &indexBuffer = mModel.buffers[indexBufferView.buffer];
    const uint8_t *indexes = indexBuffer.data.data() + indexAccessor.byteOffset + indexBufferView.byteOffset;
    int stride = indexAccessor.ByteStride(indexBufferView);
    primitiveData.indices.resize(indexAccessor.count);

    if (stride == 1) {
        for (size_t i = 0; i < indexAccessor.count; i += 3) {
            primitiveData.indices[i + 0] = indexes[i + indexRemap[0]];
            primitiveData.indices[i + 1] = indexes[i + indexRemap[1]];
            primitiveData.indices[i + 2] = indexes[i + indexRemap[2]];
        }
    } else if (stride == 2) {
        for (size_t i = 0; i < indexAccessor.count; i += 3) {
            primitiveData.indices[i + 0] = ((uint16_t *)indexes)[i + indexRemap[0]];
            primitiveData.indices[i + 1] = ((uint16_t *)indexes)[i + indexRemap[1]];
            primitiveData.indices[i + 2] = ((uint16_t *)indexes)[i + indexRemap[2]];
        }
    } else if (stride == 4) {
        for (size_t i = 0; i < indexAccessor.count; i += 3) {
            primitiveData.indices[i + 0] = ((uint32_t *)indexes)[i + indexRemap[0]];
            primitiveData.indices[i + 1] = ((uint32_t *)indexes)[i + indexRemap[1]];
            primitiveData.indices[i + 2] = ((uint32_t *)indexes)[i + indexRemap[2]];
        }
    } else {
        assert(0 && "unsupported index stride");
    }
}

void GLTFLoader::LoadPrimitivePositions(
    tinygltf::Primitive &primitive, GLTFPrimitiveData &primitiveData
) const {
    const tinygltf::Accessor &accessor = mModel.accessors[primitive.attributes["POSITION"]];
    const tinygltf::BufferView &bufferView = mModel.bufferViews[accessor.bufferView];
    const tinygltf::Buffer &buffer = mModel.buffers[bufferView.buffer];
    int stride = accessor.ByteStride(bufferView);
    const uint8_t *data = buffer.data.data() + accessor.byteOffset + bufferView.byteOffset;

    for (size_t i = 0; i < accessor.count; ++i) {
        primitiveData.positions.push_back(*(const glm::vec3 *)(data + i * stride));
    }
}

void GLTFLoader::LoadPrimitiveNormals(
    tinygltf::Primitive &primitive, GLTFPrimitiveData &primitiveData
) const {
    const tinygltf::Accessor &accessor = mModel.accessors[primitive.attributes["NORMAL"]];
    const tinygltf::BufferView &bufferView = mModel.bufferViews[accessor.bufferView];
    const tinygltf::Buffer &buffer = mModel.buffers[bufferView.buffer];
    int stride = accessor.ByteStride(bufferView);
    const uint8_t *data = buffer.data.data() + accessor.byteOffset + bufferView.byteOffset;

    for (size_t i = 0; i < accessor.count; ++i) {
        primitiveData.normals.push_back(*(const glm::vec3 *)(data + i * stride));
    }
}

void GLTFLoader::LoadPrimitiveUVs(
    tinygltf::Primitive &primitive,
    GLTFPrimitiveData &primitiveData
) const {
    const tinygltf::Accessor &accessor = mModel.accessors[primitive.attributes["TEXCOORD_0"]];
    const tinygltf::BufferView &bufferView = mModel.bufferViews[accessor.bufferView];
    const tinygltf::Buffer &buffer = mModel.buffers[bufferView.buffer];
    int stride = accessor.ByteStride(bufferView);
    const uint8_t *data = buffer.data.data() + accessor.byteOffset + bufferView.byteOffset;

    for (size_t i = 0; i < accessor.count; ++i) {
        primitiveData.uvs.push_back(*(const glm::vec2 *)(data + i * stride));
    }
}

void GLTFLoader::LoadPrimitiveTexture(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
) const {
    primitiveData.texture = mModel.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index;
}

void GLTFLoader::LoadPrimitiveMaterial(
        tinygltf::Primitive &primitive,
        GLTFPrimitiveData &primitiveData
) const {
    primitiveData.material = primitive.material;
}

vector<GLTFTextureData> GLTFLoader::LoadTextures() {
    unsigned int textureCount = mModel.textures.size();

    vector<GLTFTextureData> textureData(textureCount);

    for (int i = 0; i < textureCount; ++i) {
        tinygltf::Texture &texture = mModel.textures[i];
        tinygltf::Image &image = mModel.images[texture.source];
        textureData[i].uri = FileSystem::GetDirectory(mFilename) + "/" + image.uri;
    }

    return textureData;
}
 
vector<GLTFMaterialData> GLTFLoader::LoadMaterials(const std::vector<GLTFTextureData> &textures) {
    unsigned int materialCount = mModel.materials.size();

    vector<GLTFMaterialData> materialData(materialCount);

    for (int i = 0; i < materialCount; ++i) {
        tinygltf::Material &material = mModel.materials[i];
        materialData[i].baseColorMap = material.pbrMetallicRoughness.baseColorTexture.index;
        materialData[i].metallicRoughnessMap = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
        materialData[i].normalMap = material.normalTexture.index;
    }

    return materialData;
}