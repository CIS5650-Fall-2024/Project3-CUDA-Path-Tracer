// scene.cpp

#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#define TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp> // For translate, rotate, scale
#include <glm/gtc/type_ptr.hpp>         // For make_mat4
#include <unordered_map>
#include "json.hpp"
#include "scene.h"

using json = nlohmann::json;

Scene::Scene(std::string filename)
{
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
    }
    else if (ext == ".gltf" || ext == ".glb")
    {
        loadFromGLTF(filename);
    }
    else
    {
        std::cerr << "Unsupported file format: " << ext << std::endl;
        exit(-1);
    }
}
void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;

    for (const auto& item : materialsData.items()) {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        std::cout << "Processing material: " << name << "\n";

        std::string type = p["TYPE"];

        if (type == "Diffuse") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.baseColorTextureIndex = newMaterial.normalTextureIndex = -1;
            newMaterial.isProcedural = false;
            std::cout << "  Type: Diffuse\n";
        }
        else if (type == "Emitting") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.baseColorTextureIndex = newMaterial.normalTextureIndex = -1;
            newMaterial.isProcedural = false;
            std::cout << "  Type: Emitting\n";
            std::cout << "  Emittance: " << newMaterial.emittance << "\n";
        }
        else if (type == "Specular") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.baseColorTextureIndex = newMaterial.normalTextureIndex = -1;
            newMaterial.hasReflective = 1.0f; // Specular means reflective
            newMaterial.isProcedural = false;
            std::cout << "  Type: Specular\n";
        }
        else if (type == "ProceduralChecker") {
            // Set up procedural material
            newMaterial.isProcedural = true;
            const auto& col1 = p["COLOR1"];
            const auto& col2 = p["COLOR2"];
            newMaterial.proceduralColor1 = glm::vec3(col1[0], col1[1], col1[2]);
            newMaterial.proceduralColor2 = glm::vec3(col2[0], col2[1], col2[2]);
            newMaterial.proceduralScale = p["SCALE"];
            newMaterial.baseColorTextureIndex = -1;
            newMaterial.normalTextureIndex = -1;
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
            newMaterial.emittance = 0.0f;

            std::cout << "  Type: ProceduralChecker\n";
            std::cout << "  Color1: (" << newMaterial.proceduralColor1.r << ", "
                << newMaterial.proceduralColor1.g << ", "
                << newMaterial.proceduralColor1.b << ")\n";
            std::cout << "  Color2: (" << newMaterial.proceduralColor2.r << ", "
                << newMaterial.proceduralColor2.g << ", "
                << newMaterial.proceduralColor2.b << ")\n";
            std::cout << "  Scale: " << newMaterial.proceduralScale << "\n";
        }
        else {
            std::cerr << "Unknown material type: " << type << "\n";
            continue;
        }

        // Log material color
        std::cout << "  Color: (" << newMaterial.color.r << ", "
            << newMaterial.color.g << ", "
            << newMaterial.color.b << ")\n";

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else {
            std::cerr << "Unknown object type: " << type << "\n";
            continue;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // Calculate FOV based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // Set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}


glm::vec2 mapUV(const glm::vec2& uv) {
    glm::vec2 mappedUV;
    // If u or v is 3 or greater, map it to 1; otherwise, map to 0
    mappedUV.x = (uv.x >= 3.0f) ? 1.0f : 0.0f;
    mappedUV.y = (uv.y >= 3.0f) ? 1.0f : 0.0f;

    return mappedUV;
}

void Scene::addPlaneAsMesh(const glm::vec3& position, const glm::vec3& normal, float width, float height, int materialId) {
    Geom newGeom;
    newGeom.type = CUSTOM_MESH;
    newGeom.materialid = materialId;

    // Generate four vertices of the plane based on the width and height
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::cross(normal, up);
    if (glm::length(right) < 0.001f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);  // In case the normal is up, use an arbitrary right direction
    }
    right = glm::normalize(right);
    up = glm::normalize(glm::cross(right, normal));

    glm::vec3 halfRight = right * (width * 0.5f);
    glm::vec3 halfUp = up * (height * 0.5f);

    // Define the four corners of the plane
    glm::vec3 v0 = position - halfRight - halfUp;
    glm::vec3 v1 = position + halfRight - halfUp;
    glm::vec3 v2 = position + halfRight + halfUp;
    glm::vec3 v3 = position - halfRight + halfUp;


    // Create the two triangles
    std::vector<Triangle> triangles(2);
    triangles[0].v0 = v0;
    triangles[0].v1 = v1;
    triangles[0].v2 = v2;

    triangles[1].v0 = v2;
    triangles[1].v1 = v3;
    triangles[1].v2 = v0;

    // Set normal for each vertex (same normal for all vertices in a flat plane)
    for (int i = 0; i < 2; i++) {
        triangles[i].n0 = triangles[i].n1 = triangles[i].n2 = normal;
    }

    // Set UV coordinates (simple planar mapping)
    triangles[0].uv0 = glm::vec2(0.0f, 0.0f);
    triangles[0].uv1 = glm::vec2(1.0f, 0.0f);
    triangles[0].uv2 = glm::vec2(1.0f, 1.0f);

    triangles[1].uv0 = glm::vec2(1.0f, 1.0f);
    triangles[1].uv1 = glm::vec2(0.0f, 1.0f);
    triangles[1].uv2 = glm::vec2(0.0f, 0.0f);

    // Add the triangles to the global triangle list
    int triangleStartIndex = this->triangles.size();
    this->triangles.insert(this->triangles.end(), triangles.begin(), triangles.end());

    // Set the bounding box (AABB) for the plane
    glm::vec3 bboxMin = glm::min(glm::min(v0, v1), glm::min(v2, v3));
    glm::vec3 bboxMax = glm::max(glm::max(v0, v1), glm::max(v2, v3));
    newGeom.bboxMin = bboxMin;
    newGeom.bboxMax = bboxMax;

    // Store geometry's triangle range
    newGeom.triangleStartIndex = triangleStartIndex;
    newGeom.triangleCount = 2;

    // Add the new geometry to the list
    geoms.push_back(newGeom);
}


void Scene::loadFromGLTF(const std::string& gltfName)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    // Load the GLTF file
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
    // If your GLTF file is binary (.glb), use LoadBinaryFromFile:
    // turned off b/c of issues
    // bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);

    if (!warn.empty()) {
        std::cerr << "GLTF Warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "GLTF Error: " << err << std::endl;
    }

    if (!ret) {
        std::cerr << "Failed to load GLTF file: " << gltfName << std::endl;
        return;
    }

    // Mapping from GLTF material index to our material index
    std::unordered_map<int, int> materialIdMap;

    bool hasEmissiveMaterial = false;

    // Process materials


    for (size_t i = 0; i < model.materials.size(); ++i) {
        const tinygltf::Material& gltfMaterial = model.materials[i];
        Material newMaterial{};

        // Default values
        newMaterial.color = glm::vec3(1.0f);
        newMaterial.specular.color = glm::vec3(1.0f);
        newMaterial.specular.exponent = 0.0f; // Not used for perfect specular reflection
        newMaterial.hasReflective = 0.0f;
        newMaterial.hasRefractive = 0.0f;
        newMaterial.indexOfRefraction = 1.0f;
        newMaterial.emittance = 0.0f;
        newMaterial.baseColorTextureIndex = -1;
        newMaterial.normalTextureIndex = -1;

        std::cout << "Material " << i << ":\n";

        // Check if the material has pbrMetallicRoughness
        if (gltfMaterial.pbrMetallicRoughness.baseColorFactor.size() > 0 ||
            gltfMaterial.pbrMetallicRoughness.metallicFactor >= 0.0f ||
            gltfMaterial.pbrMetallicRoughness.baseColorTexture.index >= 0) {

            const auto& pbr = gltfMaterial.pbrMetallicRoughness;

            // Extract baseColorFactor
            if (pbr.baseColorFactor.size() == 4) {
                newMaterial.color = glm::vec3(
                    static_cast<float>(pbr.baseColorFactor[0]),
                    static_cast<float>(pbr.baseColorFactor[1]),
                    static_cast<float>(pbr.baseColorFactor[2])
                );
                float alpha = static_cast<float>(pbr.baseColorFactor[3]);  // Store alpha if needed
                std::cout << "  Base Color Factor: (" << newMaterial.color.r << ", "
                    << newMaterial.color.g << ", "
                    << newMaterial.color.b << ", "
                    << alpha << ")\n";
            }
            else {
                std::cout << "  No baseColorFactor provided for material " << i << "\n";
            }

            // Extract metallicFactor
            float metallicFactor = (pbr.metallicFactor == pbr.metallicFactor) ?
                static_cast<float>(pbr.metallicFactor) : 1.0f; // Default to 1.0 if NaN
            std::cout << "  Metallic Factor: " << metallicFactor << "\n";

            // Set hasReflective based on metallicFactor
            if (metallicFactor >= 0.5f) { // Threshold can be adjusted as needed
                newMaterial.hasReflective = 1.0f;
                newMaterial.specular.color = newMaterial.color; // Use base color as specular color
                std::cout << "  Material is reflective\n";
            }
            else {
                newMaterial.hasReflective = 0.0f;
                std::cout << "  Material is not reflective\n";
            }

            // Extract base color texture if present
            if (pbr.baseColorTexture.index >= 0) {
                int textureIndex = pbr.baseColorTexture.index;
                newMaterial.baseColorTextureIndex = textureIndex;
                std::cout << "  Base Color Texture Index: " << textureIndex << "\n";
            }
        }
        else {
            std::cout << "  No pbrMetallicRoughness found for material " << i << "\n";
        }

        // Extract emissive factor if present
        if (gltfMaterial.emissiveFactor.size() == 3) {
            glm::vec3 emissiveColor = glm::vec3(
                static_cast<float>(gltfMaterial.emissiveFactor[0]),
                static_cast<float>(gltfMaterial.emissiveFactor[1]),
                static_cast<float>(gltfMaterial.emissiveFactor[2])
            );
            newMaterial.emittance = glm::length(emissiveColor);  // Emittance is the length of emissive color vector
            std::cout << "  Emissive Color: (" << emissiveColor.r << ", "
                << emissiveColor.g << ", "
                << emissiveColor.b << ")\n";
            if (newMaterial.emittance > 0.0f) {
                hasEmissiveMaterial = true;
            }
        }

        // Check for normal texture
        if (gltfMaterial.normalTexture.index >= 0) {
            int textureIndex = gltfMaterial.normalTexture.index;
            newMaterial.normalTextureIndex = textureIndex;
            std::cout << "  Normal Texture Index: " << textureIndex << "\n";
        }

        // Log material properties
        std::cout << "  Color: (" << newMaterial.color.r << ", "
            << newMaterial.color.g << ", "
            << newMaterial.color.b << ")\n";
        std::cout << "  Emittance: " << newMaterial.emittance << "\n";
        std::cout << "  hasReflective: " << newMaterial.hasReflective << "\n";

        // Map GLTF material index to our material index
        materialIdMap[i] = materials.size();
        materials.push_back(newMaterial);
    }


    // Load textures
    std::vector<Texture> textures;
    // Load textures
    for (size_t i = 0; i < model.textures.size(); ++i) {
        const tinygltf::Texture& gltfTexture = model.textures[i];
        const tinygltf::Image& image = model.images[gltfTexture.source];

        Texture texture{};
        texture.width = image.width;
        texture.height = image.height;
        texture.components = image.component; // Number of color channels

        // Convert image data to RGBA if necessary
        if (image.component == 3) {
            // Convert RGB to RGBA
            texture.imageData.resize(image.width * image.height * 4);
            for (int y = 0; y < image.height; ++y) {
                for (int x = 0; x < image.width; ++x) {
                    int srcIndex = (y * image.width + x) * 3;
                    int dstIndex = (y * image.width + x) * 4;

                    texture.imageData[dstIndex + 0] = image.image[srcIndex + 0];
                    texture.imageData[dstIndex + 1] = image.image[srcIndex + 1];
                    texture.imageData[dstIndex + 2] = image.image[srcIndex + 2];
                    texture.imageData[dstIndex + 3] = 255; // Set alpha to 255
                }
            }
        }
        else if (image.component == 4) {
            // Image is already RGBA
            texture.imageData = image.image;
        }
        else {
            // Handle other component counts if necessary
        }

        textures.push_back(texture);
    }




    // Function to process nodes recursively
    std::function<void(int, const glm::mat4&)> processNode;
    processNode = [&](int nodeIdx, const glm::mat4& parentTransform) {
        const tinygltf::Node& node = model.nodes[nodeIdx];

        glm::mat4 localTransform = parentTransform;

        // Node's local transformation
        glm::mat4 nodeMatrix(1.0f);
        if (node.matrix.size() == 16) {
            // Use the node's matrix directly
            nodeMatrix = glm::make_mat4(node.matrix.data());
        }
        else {
            // Compose the matrix from TRS components
            glm::vec3 translation(0.0f);
            if (node.translation.size() == 3) {
                translation = glm::vec3(
                    static_cast<float>(node.translation[0]),
                    static_cast<float>(node.translation[1]),
                    static_cast<float>(node.translation[2])
                );
            }

            glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
            if (node.rotation.size() == 4) {
                rotation = glm::quat(
                    static_cast<float>(node.rotation[3]), // w
                    static_cast<float>(node.rotation[0]), // x
                    static_cast<float>(node.rotation[1]), // y
                    static_cast<float>(node.rotation[2])  // z
                );
            }

            glm::vec3 scale(1.0f);
            if (node.scale.size() == 3) {
                scale = glm::vec3(
                    static_cast<float>(node.scale[0]),
                    static_cast<float>(node.scale[1]),
                    static_cast<float>(node.scale[2])
                );
            }

            nodeMatrix = glm::translate(glm::mat4(1.0f), translation) *
                glm::mat4_cast(rotation) *
                glm::scale(glm::mat4(1.0f), scale);
        }

        // Update local transform
        localTransform = parentTransform * nodeMatrix;

        // If the node has a mesh, process it
        if (node.mesh >= 0) {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];

            for (const auto& primitive : mesh.primitives) {
                Geom newGeom;
                newGeom.type = CUSTOM_MESH;

                // Set transformation matrices
                newGeom.transform = localTransform;
                newGeom.inverseTransform = glm::inverse(localTransform);
                newGeom.invTranspose = glm::inverseTranspose(localTransform);

                // Set default translation, rotation, and scale
                newGeom.translation = glm::vec3(0.0f);
                newGeom.rotation = glm::vec3(0.0f);
                newGeom.scale = glm::vec3(1.0f);

                // Set the material ID
                int materialIdx = 0; // Default material index
                if (primitive.material >= 0) {
                    auto it = materialIdMap.find(primitive.material);
                    if (it != materialIdMap.end()) {
                        materialIdx = it->second;
                    }
                }
                newGeom.materialid = materialIdx;

                // Handle mesh data
                // Extract indices
                std::vector<unsigned int> indices;
                {
                    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                    const tinygltf::BufferView& bufferView = model.bufferViews[indexAccessor.bufferView];
                    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                    const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + indexAccessor.byteOffset;
                    size_t indexCount = indexAccessor.count;
                    size_t indexByteStride = indexAccessor.ByteStride(bufferView);
                    if (indexByteStride == 0) {
                        indexByteStride = tinygltf::GetComponentSizeInBytes(indexAccessor.componentType);
                    }

                    indices.resize(indexCount);

                    for (size_t i = 0; i < indexCount; ++i) {
                        const unsigned char* offsetPtr = dataPtr + i * indexByteStride;
                        unsigned int indexValue = 0;

                        switch (indexAccessor.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            indexValue = *reinterpret_cast<const uint8_t*>(offsetPtr);
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            indexValue = *reinterpret_cast<const uint16_t*>(offsetPtr);
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            indexValue = *reinterpret_cast<const uint32_t*>(offsetPtr);
                            break;
                        default:
                            std::cerr << "Unsupported index component type: " << indexAccessor.componentType << std::endl;
                            break;
                        }

                        indices[i] = indexValue;
                    }
                }

                // Extract positions
                std::vector<glm::vec3> positions;
                {
                    auto attrib = primitive.attributes.find("POSITION");
                    if (attrib == primitive.attributes.end()) {
                        std::cerr << "Mesh is missing POSITION attribute" << std::endl;
                        continue;
                    }
                    const tinygltf::Accessor& accessor = model.accessors[attrib->second];
                    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                    const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
                    size_t vertexCount = accessor.count;
                    size_t byteStride = accessor.ByteStride(bufferView);
                    if (byteStride == 0) {
                        byteStride = sizeof(float) * 3; // Assuming vec3 positions
                    }

                    positions.resize(vertexCount);

                    for (size_t i = 0; i < vertexCount; ++i) {
                        const float* p = reinterpret_cast<const float*>(dataPtr + i * byteStride);
                        positions[i] = glm::vec3(p[0], p[1], p[2]);
                    }
                }

                // Extract normals
                std::vector<glm::vec3> normals;
                {
                    auto attrib = primitive.attributes.find("NORMAL");

                    if (attrib != primitive.attributes.end()) {
                        const tinygltf::Accessor& accessor = model.accessors[attrib->second];
                        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                        const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
                        size_t vertexCount = accessor.count;
                        size_t byteStride = accessor.ByteStride(bufferView);
                        if (byteStride == 0) {
                            byteStride = sizeof(float) * 3; // Assuming vec3 normals
                        }

                        normals.resize(vertexCount);

                        for (size_t i = 0; i < vertexCount; ++i) {
                            const float* p = reinterpret_cast<const float*>(dataPtr + i * byteStride);
                            normals[i] = glm::vec3(p[0], p[1], p[2]);
                        }
                    }
                    else {
                        // Generate normals if not provided (optional)
                        normals.resize(positions.size(), glm::vec3(0.0f));
                        //std::cout << "Mesh is missing POSITION attribute" << std::endl;
                        continue;
                    }
                }
                const float epsilon = 0.1f; // Small value to avoid flat or degenerate AABBs
                glm::vec3 bboxMin(FLT_MAX);
                glm::vec3 bboxMax(-FLT_MAX);

                for (const glm::vec3& pos : positions) {
                    bboxMin = glm::min(bboxMin, pos);
                    bboxMax = glm::max(bboxMax, pos);
                }


                // Check if any dimension is flat and expand it by epsilon
                if (bboxMin.x == bboxMax.x) {
                    bboxMin.x -= epsilon;
                    bboxMax.x += epsilon;
                }
                if (bboxMin.y == bboxMax.y) {
                    bboxMin.y -= epsilon;
                    bboxMax.y += epsilon;
                }
                if (bboxMin.z == bboxMax.z) {
                    bboxMin.z -= epsilon;
                    bboxMax.z += epsilon;
                }

                newGeom.bboxMin = bboxMin;
                newGeom.bboxMax = bboxMax;
                std::cout << "Primitive AABBs:\n";
                std::cout << "  bboxMin: (" << bboxMin.x << ", " << bboxMin.y << ", " << bboxMin.z << ")\n";
                std::cout << "  bboxMax: (" << bboxMax.x << ", " << bboxMax.y << ", " << bboxMax.z << ")\n";

                std::vector<glm::vec2> uvs;
                {
                    auto attrib = primitive.attributes.find("TEXCOORD_0");
                    if (attrib != primitive.attributes.end()) {
                        const tinygltf::Accessor& accessor = model.accessors[attrib->second];
                        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                        const unsigned char* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
                        size_t vertexCount = accessor.count;
                        size_t byteStride = accessor.ByteStride(bufferView);
                        if (byteStride == 0) {
                            byteStride = sizeof(float) * 2; // Assuming vec2 UVs
                        }

                        uvs.resize(vertexCount);

                        for (size_t i = 0; i < vertexCount; ++i) {
                            const float* p = reinterpret_cast<const float*>(dataPtr + i * byteStride);
                            uvs[i] = glm::vec2(p[0], p[1]);
                        }
                    }
                    else {
                        // No UV coordinates provided
                        uvs.resize(positions.size(), glm::vec2(0.0f));
                        //std::cout << "Mesh is missing POSITION attribute" << std::endl;

                    }
                }

                // Create triangles
                int triangleStartIndex = triangles.size();
                int triangleCount = indices.size() / 3;

                for (size_t i = 0; i < indices.size(); i += 3) {
                    Triangle triangle;
                    unsigned int idx0 = indices[i];
                    unsigned int idx1 = indices[i + 1];
                    unsigned int idx2 = indices[i + 2];

                    triangle.v0 = positions[idx0];
                    triangle.v1 = positions[idx1];
                    triangle.v2 = positions[idx2];

                    // Log vertex positions
                    //std::cout << "Triangle " << i / 3 << " vertices:\n";
                    //std::cout << "  v0: (" << triangle.v0.x << ", " << triangle.v0.y << ", " << triangle.v0.z << ")\n";
                    //std::cout << "  v1: (" << triangle.v1.x << ", " << triangle.v1.y << ", " << triangle.v1.z << ")\n";
                    //std::cout << "  v2: (" << triangle.v2.x << ", " << triangle.v2.y << ", " << triangle.v2.z << ")\n";

                    if (!normals.empty()) {
                        // Assign normals from the normals vector
                        triangle.n0 = normals[idx0];
                        triangle.n1 = normals[idx1];
                        triangle.n2 = normals[idx2];

                        // Log vertex normals
                        //std::cout << "  n0: (" << triangle.n0.x << ", " << triangle.n0.y << ", " << triangle.n0.z << ")\n";
                        //std::cout << "  n1: (" << triangle.n1.x << ", " << triangle.n1.y << ", " << triangle.n1.z << ")\n";
                        //std::cout << "  n2: (" << triangle.n2.x << ", " << triangle.n2.y << ", " << triangle.n2.z << ")\n";
                    }
                    else {
                        // Calculate face normal
                        glm::vec3 edge1 = triangle.v1 - triangle.v0;
                        glm::vec3 edge2 = triangle.v2 - triangle.v0;
                        glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

                        // Assign the face normal to all three vertices
                        triangle.n0 = triangle.n1 = triangle.n2 = faceNormal;

                        // Log calculated face normal
                        //std::cout << "  Calculated face normal: (" << faceNormal.x << ", " << faceNormal.y << ", " << faceNormal.z << ")\n";
                    }

                    triangle.uv0 = uvs[idx0];
                    triangle.uv1 = uvs[idx1];
                    triangle.uv2 = uvs[idx2];
                    //triangle.uv0 = glm::vec2(0.0f, 0.0f);
                    //triangle.uv1 = glm::vec2(1.0f, 0.0f);
                    //triangle.uv2 = glm::vec2(1.0f, 1.0f);



                    //std::cout << "Triangle " << i / 3 << " UVs:\n";
                    //std::cout << "  uv0: (" << triangle.uv0.x << ", " << triangle.uv0.y << ")\n";
                    //std::cout << "  uv1: (" << triangle.uv1.x << ", " << triangle.uv1.y << ")\n";
                    //std::cout << "  uv2: (" << triangle.uv2.x << ", " << triangle.uv2.y << ")\n";




                    triangles.push_back(triangle);
                }



                newGeom.triangleStartIndex = triangleStartIndex;
                newGeom.triangleCount = triangleCount;

                geoms.push_back(newGeom);
            }
        }

        // Recursively process children
        for (size_t i = 0; i < node.children.size(); ++i) {
            processNode(node.children[i], localTransform);
        }
        };

    this->textures = textures;

    


    // Start processing nodes from the default scene
    const tinygltf::Scene& gltfScene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
    for (size_t i = 0; i < gltfScene.nodes.size(); ++i) {
        processNode(gltfScene.nodes[i], glm::mat4(1.0f));
    }

    glm::vec3 sceneBboxMin(FLT_MAX);
    glm::vec3 sceneBboxMax(-FLT_MAX);


    for (const Geom& geom : geoms) {
        sceneBboxMin = glm::min(sceneBboxMin, geom.bboxMin);
        sceneBboxMax = glm::max(sceneBboxMax, geom.bboxMax);
    }

    // If no emissive materials are present, add the room and ceiling light
    if (!hasEmissiveMaterial) {
        // Calculate room center and size (10x the size of the scene AABB)
        glm::vec3 roomCenter = (sceneBboxMin + sceneBboxMax) * 0.5f;
        glm::vec3 roomSize = (sceneBboxMax - sceneBboxMin) * 10.0f;
        float roomHeight = roomSize.y;

        // Add a white diffuse material for the walls
        Material whiteDiffuse{};
        whiteDiffuse.color = glm::vec3(0.3f, 0.3f, 0.3f);
        whiteDiffuse.emittance = 0.0f;
        whiteDiffuse.baseColorTextureIndex = -1;
        whiteDiffuse.normalTextureIndex = -1;
        int whiteDiffuseMatId = materials.size();
        materials.push_back(whiteDiffuse);

        // Adjust each plane to fully enclose the scene

        // Floor (width = roomSize.x, depth = roomSize.z)
        addPlaneAsMesh(glm::vec3(roomCenter.x, sceneBboxMin.y - roomSize.y * 0.5f, roomCenter.z), glm::vec3(0, 1, 0), roomSize.x, roomSize.z, whiteDiffuseMatId);

        // Back wall (width = roomSize.x, height = roomHeight)
        addPlaneAsMesh(glm::vec3(roomCenter.x, roomCenter.y, sceneBboxMin.z - roomSize.z * 0.5f), glm::vec3(0, 0, 1), roomSize.x, roomHeight, whiteDiffuseMatId);

        // Left wall (depth = roomSize.z, height = roomHeight)
        addPlaneAsMesh(glm::vec3(sceneBboxMin.x - roomSize.x * 0.5f, roomCenter.y, roomCenter.z), glm::vec3(1, 0, 0), roomSize.z, roomHeight, whiteDiffuseMatId);

        // Right wall (depth = roomSize.z, height = roomHeight)
        addPlaneAsMesh(glm::vec3(sceneBboxMax.x + roomSize.x * 0.5f, roomCenter.y, roomCenter.z), glm::vec3(-1, 0, 0), roomSize.z, roomHeight, whiteDiffuseMatId);

        // Add ceiling light with emissive material
        Material ceilingLight{};
        ceilingLight.color = glm::vec3(1.0f, 1.0f, 1.0f);  // White light
        ceilingLight.emittance = 5.0f;  // Adjust intensity as needed
        ceilingLight.baseColorTextureIndex = ceilingLight.normalTextureIndex = -1;
        int ceilingLightMatId = materials.size();
        materials.push_back(ceilingLight);

        // Ceiling (width = roomSize.x, depth = roomSize.z)
        addPlaneAsMesh(glm::vec3(roomCenter.x, sceneBboxMax.y + roomSize.y * 0.5f - 0.1f, roomCenter.z), glm::vec3(0, -1, 0), roomSize.x, roomSize.z, ceilingLightMatId);  // Ceiling light
    }



    // Process camera (if available)
    Camera& camera = state.camera;
    if (!model.cameras.empty()) {
        const tinygltf::Camera& gltfCamera = model.cameras[0];

        // Set camera type and parameters
        if (gltfCamera.type == "perspective") {
            const auto& persp = gltfCamera.perspective;
            camera.fov.y = glm::degrees(static_cast<float>(persp.yfov));
            float aspectRatio = static_cast<float>(persp.aspectRatio);
            camera.fov.x = glm::degrees(2.0f * atan(tan(glm::radians(camera.fov.y * 0.5f)) * aspectRatio));
        }

        // Find the node that references this camera to get its transform
        for (const auto& node : model.nodes) {
            if (node.camera == 0) {
                glm::mat4 cameraTransform = glm::mat4(1.0f);

                // Apply node's transformation
                if (node.matrix.size() == 16) {
                    cameraTransform = glm::make_mat4(node.matrix.data());
                }
                else {
                    glm::vec3 translation(0.0f);
                    if (node.translation.size() == 3) {
                        translation = glm::vec3(
                            static_cast<float>(node.translation[0]),
                            static_cast<float>(node.translation[1]),
                            static_cast<float>(node.translation[2])
                        );
                    }

                    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
                    if (node.rotation.size() == 4) {
                        rotation = glm::quat(
                            static_cast<float>(node.rotation[3]), // w
                            static_cast<float>(node.rotation[0]), // x
                            static_cast<float>(node.rotation[1]), // y
                            static_cast<float>(node.rotation[2])  // z
                        );
                    }

                    glm::vec3 scale(1.0f);
                    if (node.scale.size() == 3) {
                        scale = glm::vec3(
                            static_cast<float>(node.scale[0]),
                            static_cast<float>(node.scale[1]),
                            static_cast<float>(node.scale[2])
                        );
                    }

                    cameraTransform = glm::translate(glm::mat4(1.0f), translation) *
                        glm::mat4_cast(rotation) *
                        glm::scale(glm::mat4(1.0f), scale);
                }

                // Extract position and orientation
                glm::vec3 cameraPosition = glm::vec3(cameraTransform[3]);
                glm::vec3 cameraDirection = glm::normalize(glm::vec3(cameraTransform * glm::vec4(0, 0, -1, 0)));
                glm::vec3 cameraUp = glm::normalize(glm::vec3(cameraTransform * glm::vec4(0, 1, 0, 0)));

                camera.position = cameraPosition;
                camera.lookAt = cameraPosition + cameraDirection;
                camera.up = cameraUp;

                camera.view = glm::normalize(camera.lookAt - camera.position);
                camera.right = glm::normalize(glm::cross(camera.view, camera.up));

                // Set resolution if available (default to 800x600)
                camera.resolution = glm::ivec2(800, 600);
                if (gltfCamera.type == "perspective" && gltfCamera.perspective.aspectRatio > 0.0) {
                    camera.resolution.x = static_cast<int>(camera.resolution.y * gltfCamera.perspective.aspectRatio);
                }

                break;

                camera.lensRadius = 0.005f;      // Adjust aperture size (controls the amount of blur)
                glm::vec3 sceneCenter = (sceneBboxMin + sceneBboxMax) * 0.5f;

                // Compute the distance from the camera to the scene center
                float focalDistance = glm::length(sceneCenter - camera.position);

                // Set the focal distance
                camera.focalDistance = focalDistance;
            }
        }
    }
    else {
        // Set a default camera if none is provided
        camera.resolution = glm::ivec2(800, 600);
        camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
        camera.lookAt = glm::vec3(0.0f);
        camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
        camera.view = glm::normalize(camera.lookAt - camera.position);
        camera.right = glm::normalize(glm::cross(camera.view, camera.up));
        camera.fov = glm::vec2(45.0f, 45.0f);

        camera.lensRadius = 0.005f;      // Adjust aperture size (controls the amount of blur)
        // Assuming you have sceneBboxMin and sceneBboxMax
        glm::vec3 sceneCenter = (sceneBboxMin + sceneBboxMax) * 0.5f;

        // Compute the distance from the camera to the scene center
        float focalDistance = glm::length(sceneCenter - camera.position);

        // Set the focal distance
        camera.focalDistance = focalDistance;
    }

    // Calculate camera parameters
    float yscaled = tan(glm::radians(camera.fov.y * 0.5f));
    float xscaled = yscaled * static_cast<float>(camera.resolution.x) / static_cast<float>(camera.resolution.y);
    camera.pixelLength = glm::vec2(
        (2.0f * xscaled) / static_cast<float>(camera.resolution.x),
        (2.0f * yscaled) / static_cast<float>(camera.resolution.y)
    );

    // Initialize render state
    RenderState& state = this->state;
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3(0.0f));

    // Set default render parameters
    state.iterations = 5000; // Adjust as needed
    state.traceDepth = 8;    // Adjust as needed
    state.imageName = "output";
}