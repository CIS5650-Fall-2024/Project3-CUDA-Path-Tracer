#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.color = glm::vec3(0.0f);

            // Read roughness if provided, default to 0.0f
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = p["ROUGHNESS"];
            } else {
                newMaterial.roughness = 0.0f;
            }
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;

            // check if IOR exists, otherwise set a default
            if (p.contains("IOR")) {
                newMaterial.indexOfRefraction = p["IOR"];
            } else {
                newMaterial.indexOfRefraction = 1.0f;
            }

            // specular reflection for refractive materials
            if (p.contains("SPECULAR_COLOR")) {
                const auto& specCol = p["SPECULAR_COLOR"];
                newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
                newMaterial.hasReflective = 1.0f;
            } else {
                newMaterial.specular.color = glm::vec3(1.0f);
            }

            // Read roughness
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = p["ROUGHNESS"];
            } else {
                newMaterial.roughness = 0.0f;
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
#if DEBUG
    for (const auto& pair : MatNameToID) {
        std::cout << "Material Name: " << pair.first << ", ID: " << pair.second << std::endl;
    }
#endif
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        Geom newGeom;
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation,
                newGeom.rotation,
                newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        const auto& type = p["TYPE"];
        if (type == "gltf") {
            newGeom.type = MESH;
            const std::string filename = p["FILE"].get<std::string>();

            std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            std::string fullPath = sceneDir + filename;

            // Load the mesh
            int ret = loadGlTF(fullPath, newGeom);
            if (ret != 0) {
                std::cerr << "Failed to load gltf mesh: " << filename << std::endl;
                exit(-1);
            }
        } else if (type == "obj") {
            newGeom.type = MESH;
            const std::string filename = p["FILE"].get<std::string>();

            std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            std::string fullPath = sceneDir + filename;

            // Load the mesh
            int ret = loadObj(fullPath, newGeom);
            if (ret != 0) {
                std::cerr << "Failed to load obj mesh: " << filename << std::endl;
                exit(-1);
            }
        } else if (type == "cube") {
            newGeom.type = CUBE;
        } else if (type == "sphere") {
            newGeom.type = SPHERE;
        } else {
            std::cerr << "Unknown object type: " << type << std::endl;
            exit(-1);
        }
        geoms.push_back(newGeom);
    }

    // Build Linear BVH
//    buildBVH();

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

    if (cameraData.contains("FOCALDISTANCE")) {
        camera.focalDistance = cameraData["FOCALDISTANCE"];
    } else {
        camera.focalDistance = 1.0f;
    }

    if (cameraData.contains("APERTURE")) {
        camera.aperture = cameraData["APERTURE"];
    } else {
        camera.aperture = 0.0f;
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

int Scene::loadGlTF(const string &fullPath, Geom &geom) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, fullPath);
    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF\n");
        return -1;
    }

    std::cout << "Loaded glTF file: " << fullPath << std::endl;

    geom.meshStart = triangles.size();
    geom.meshCount = 0;

    // Step 1: Load materials from the glTF model
    std::unordered_map<int, int> materialIDMap;
    for (size_t i = 0; i < model.materials.size(); ++i) {
        const tinygltf::Material& gltfMaterial = model.materials[i];

        // Initialize with geom's material
        Material material = materials[geom.materialid];

        if (gltfMaterial.values.find("baseColorFactor") != gltfMaterial.values.end()) {
            const tinygltf::Parameter& baseColor = gltfMaterial.values.at("baseColorFactor");
            tinygltf::ColorValue colorValues = baseColor.ColorFactor();

            material.color = glm::vec3(
                    static_cast<float>(colorValues[0]),
                    static_cast<float>(colorValues[1]),
                    static_cast<float>(colorValues[2])
            );
        }

        if (gltfMaterial.additionalValues.find("emissiveFactor") != gltfMaterial.additionalValues.end()) {
            const tinygltf::Parameter& emissiveColor = gltfMaterial.additionalValues.at("emissiveFactor");
            tinygltf::ColorValue emissiveValues = emissiveColor.ColorFactor();

            material.emittance = glm::length(glm::vec3(
                    static_cast<float>(emissiveValues[0]),
                    static_cast<float>(emissiveValues[1]),
                    static_cast<float>(emissiveValues[2])
            ));
        }

        if (gltfMaterial.values.find("roughnessFactor") != gltfMaterial.values.end()) {
            material.roughness = static_cast<float>(gltfMaterial.values.at("roughnessFactor").Factor());
        }

        if (gltfMaterial.values.find("metallicFactor") != gltfMaterial.values.end()) {
            float metallic = static_cast<float>(gltfMaterial.values.at("metallicFactor").Factor());
            if (metallic > 0.0f) {
                material.hasReflective = metallic;
            }
        }

        int materialID = materials.size();
        materials.push_back(material);
        materialIDMap[i] = materialID;
    }

    // Iterate over all meshes in the glTF model
    for (const auto& mesh : model.meshes) {
        // Iterate over all primitives in the mesh
        for (const auto& primitive : mesh.primitives) {
            // Only process triangle primitives
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                std::cout << "Skipping non-triangle primitive." << std::endl;
                continue;
            }

            // Ensure the primitive has position data
            if (primitive.attributes.find("POSITION") == primitive.attributes.end()) {
                std::cerr << "Primitive is missing POSITION attribute." << std::endl;
                continue;
            }

            // Accessors for POSITION, NORMAL, TEXCOORD_0
            const tinygltf::Accessor &posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView &posView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer &posBuffer = model.buffers[posView.buffer];
            const float* posData = reinterpret_cast<const float*>(&posBuffer.data[posView.byteOffset
                                                                                  + posAccessor.byteOffset]);

            // Append positions
            int baseVertexIndex = static_cast<int>(vertices.size());
            for (size_t i = 0; i < posAccessor.count; ++i) {
                glm::vec3 vertex(
                        posData[3 * i + 0],
                        posData[3 * i + 1],
                        posData[3 * i + 2]
                );
                // Apply transformation
                vertex = glm::vec3(geom.transform * glm::vec4(vertex, 1.0f));
                vertices.emplace_back(vertex);
            }

            // Access indices if available
            bool hasIndices = primitive.indices >= 0;
            std::vector<unsigned int> indices;
            if (hasIndices) {
                const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView &indexView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer &indexBuffer = model.buffers[indexView.buffer];
                const unsigned char* indexData = &indexBuffer.data[indexView.byteOffset + indexAccessor.byteOffset];

                // Determine the index type
                if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                    const unsigned short* buf = reinterpret_cast<const unsigned short*>(indexData);
                    indices.reserve(indexAccessor.count);
                    for (size_t i = 0; i < indexAccessor.count; ++i) {
                        indices.push_back(static_cast<unsigned int>(buf[i]) + baseVertexIndex);
                    }
                }
                else if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
                    const unsigned int* buf = reinterpret_cast<const unsigned int*>(indexData);
                    indices.reserve(indexAccessor.count);
                    for (size_t i = 0; i < indexAccessor.count; ++i) {
                        indices.push_back(buf[i] + baseVertexIndex);
                    }
                }
                else {
                    std::cerr << "Unsupported index component type." << std::endl;
                    continue;
                }
            }
            else {
                // If no indices, generate them sequentially
                size_t vertexCount = posAccessor.count;
                indices.reserve(vertexCount);
                for (size_t i = 0; i < vertexCount; ++i) {
                    indices.push_back(static_cast<unsigned int>(baseVertexIndex + i));
                }
            }

            // Access normals if available
            bool hasNormals = false;
            int baseNormalIndex = -1;
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                hasNormals = true;
                const tinygltf::Accessor &normAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::BufferView &normView = model.bufferViews[normAccessor.bufferView];
                const tinygltf::Buffer &normBuffer = model.buffers[normView.buffer];
                const float* normData = reinterpret_cast<const float*>(&normBuffer.data[normView.byteOffset
                                                                                        + normAccessor.byteOffset]);

                // Append normals
                baseNormalIndex = static_cast<int>(normals.size());
                for (size_t i = 0; i < normAccessor.count; ++i) {
                    glm::vec3 norm(
                            normData[3 * i + 0],
                            normData[3 * i + 1],
                            normData[3 * i + 2]
                    );
                    // Apply normal transformation: inverse transpose of the model matrix
                    norm = glm::normalize(glm::mat3(geom.invTranspose) * norm);
                    normals.emplace_back(norm);
                }
            }

            // Access texture coordinates if available
            bool hasTexcoords = false;
            int baseUVIndex = -1;
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                hasTexcoords = true;
                const tinygltf::Accessor &uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                const tinygltf::BufferView &uvView = model.bufferViews[uvAccessor.bufferView];
                const tinygltf::Buffer &uvBuffer = model.buffers[uvView.buffer];
                const float* uvData = reinterpret_cast<const float*>(&uvBuffer.data[uvView.byteOffset + uvAccessor.byteOffset]);

                // Append UVs to the global uvs array
                baseUVIndex = static_cast<int>(uvs.size());
                for (size_t i = 0; i < uvAccessor.count; ++i) {
                    glm::vec2 uv(
                            uvData[2 * i + 0],
                            uvData[2 * i + 1]
                    );
                    uvs.emplace_back(uv);
                }
            }

            // Iterate over triangles
            size_t triangleCount = (indices.size() / 3) * 3;
            for (size_t i = 0; i + 2 < triangleCount; i += 3) {
                Triangle triangle{};
                int gltfMaterialId = primitive.material;
                if (materialIDMap.find(gltfMaterialId) != materialIDMap.end()) {
                    triangle.materialId = materialIDMap[gltfMaterialId];
                } else {
                    triangle.materialId = geom.materialid;
                }

                // Assign vertex indices
                triangle.v[0] = indices[i];
                triangle.v[1] = indices[i + 1];
                triangle.v[2] = indices[i + 2];

                // Assign normal indices with baseNormalIndex
                if (hasNormals && baseNormalIndex != -1) {
                    triangle.n[0] = triangle.v[0] + baseNormalIndex;
                    triangle.n[1] = triangle.v[1] + baseNormalIndex;
                    triangle.n[2] = triangle.v[2] + baseNormalIndex;
                }
                else {
                    triangle.n[0] = -1;
                    triangle.n[1] = -1;
                    triangle.n[2] = -1;
                }

                // Assign UV indices with baseUVIndex
                if (hasTexcoords && baseUVIndex != -1) {
                    triangle.uv[0] = triangle.v[0] + baseUVIndex;
                    triangle.uv[1] = triangle.v[1] + baseUVIndex;
                    triangle.uv[2] = triangle.v[2] + baseUVIndex;
                }
                else {
                    triangle.uv[0] = -1;
                    triangle.uv[1] = -1;
                    triangle.uv[2] = -1;
                }

                // Retrieve vertex positions
                glm::vec3 v0 = vertices[triangle.v[0]];
                glm::vec3 v1 = vertices[triangle.v[1]];
                glm::vec3 v2 = vertices[triangle.v[2]];

                // Compute the Axis-Aligned Bounding Box (AABB) for the triangle
                triangle.aabb.minBounds = glm::min(glm::min(v0, v1), v2);
                triangle.aabb.maxBounds = glm::max(glm::max(v0, v1), v2);
                triangle.aabb.centroid = (triangle.aabb.minBounds + triangle.aabb.maxBounds) * 0.5f;

                // Add the triangle to the global triangles vector
                triangles.emplace_back(triangle);
                geom.meshCount++;
            }
        }
    }

#if DEBUG
    std::cout << "Vertices: " << vertices.size() << std::endl;
    std::cout << "Normals: " << normals.size() << std::endl;
    std::cout << "UVs: " << uvs.size() << std::endl;
    std::cout << "Triangles: " << geom.meshCount << std::endl;
#endif

    if (geom.meshCount > 0) {
        geom.bvhRootIndex = buildBVHRecursive(geom.meshStart, geom.meshCount, 0);
    }

    return 0;
}

int Scene::loadObj(const string &fullPath, Geom &geom) {
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(fullPath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error() << std::endl;
        }
        return -1;
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning() << std::endl;
    }

    std::cout << "Loaded obj file: " << fullPath << std::endl;

    auto& objAttrib = reader.GetAttrib();
    auto& objShapes = reader.GetShapes();
    auto& objMaterials = reader.GetMaterials();

    // Precompute the normal transformation matrix
    glm::mat3 normalTransform = glm::transpose(glm::inverse(glm::mat3(geom.transform)));

    // Reserve space in vectors to prevent multiple reallocations
    size_t numVertices = objAttrib.vertices.size() / 3;
    vertices.reserve(vertices.size() + numVertices);

    size_t numNormals = objAttrib.normals.size() / 3;
    normals.reserve(normals.size() + numNormals);

    size_t numTexcoords = objAttrib.texcoords.size() / 2;
    uvs.reserve(uvs.size() + numTexcoords);

    // Store the base indices before adding new elements
    size_t baseVertexIndex = vertices.size();
    size_t baseNormalIndex = normals.size();
    size_t baseUVIndex = uvs.size();

    // Process and transform vertices
    for (size_t i = 0; i < numVertices; ++i) {
        glm::vec3 vertex(
                objAttrib.vertices[3 * i + 0],
                objAttrib.vertices[3 * i + 1],
                objAttrib.vertices[3 * i + 2]
        );

        // Apply transformation and add to the vertices vector
        glm::vec3 transformedVertex = glm::vec3(geom.transform * glm::vec4(vertex, 1.0f));
        vertices.emplace_back(transformedVertex);
    }

    // Process and transform normals
    for (size_t i = 0; i < numNormals; ++i) {
        glm::vec3 normal(
                objAttrib.normals[3 * i + 0],
                objAttrib.normals[3 * i + 1],
                objAttrib.normals[3 * i + 2]
        );

        // Apply normal transformation and normalize
        glm::vec3 transformedNormal = glm::normalize(normalTransform * normal);
        normals.emplace_back(transformedNormal);
    }

    // Process UVs
    if (numTexcoords > 0) {
        for (size_t i = 0; i < numTexcoords; ++i) {
            glm::vec2 uv(
                    objAttrib.texcoords[2 * i + 0],
                    objAttrib.texcoords[2 * i + 1]
            );
            uvs.emplace_back(uv);
        }
    }

    // Load materials from the OBJ file
    std::unordered_map<int, int> materialIDMap;
    for (size_t i = 0; i < objMaterials.size(); ++i) {
        const tinyobj::material_t& objMaterial = objMaterials[i];

        // Initialize with geom's material
        Material material = materials[geom.materialid];

        // Overwrite diffuse color if defined (non-zero)
        glm::vec3 diffuseColor(
                objMaterial.diffuse[0],
                objMaterial.diffuse[1],
                objMaterial.diffuse[2]
        );
        if (glm::length(diffuseColor) > 0.0f) {
            material.color = diffuseColor;
        }

        // Overwrite specular color if defined
        glm::vec3 specularColor(
                objMaterial.specular[0],
                objMaterial.specular[1],
                objMaterial.specular[2]
        );
        if (glm::length(specularColor) > 0.0f) {
            material.specular.color = specularColor;
            material.hasReflective = 1.0f;
        }

        // Overwrite shininess (specular exponent) if defined
        if (objMaterial.shininess > 0.0f) {
            material.specular.exponent = objMaterial.shininess;
            material.roughness = 1.0f - objMaterial.shininess;
        }

        // Overwrite emittance if defined
        glm::vec3 emissionColor(
                objMaterial.emission[0],
                objMaterial.emission[1],
                objMaterial.emission[2]
        );
        if (glm::length(emissionColor) > 0.0f) {
            material.emittance = glm::length(emissionColor);
        }

        // Add the material to the materials vector and map the OBJ material ID to our internal material ID
        int materialID = materials.size();
        materials.push_back(material);
        materialIDMap[i] = materialID;
    }

    geom.meshStart = triangles.size();
    geom.meshCount = 0;

    // Process triangles
    for (const auto& shape : objShapes) {
        size_t indexOffset = 0;

        // Iterate over each face in the shape
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            size_t fv = static_cast<size_t>(shape.mesh.num_face_vertices[f]);

            // Only support triangular faces
            if (fv != 3) {
                std::cerr << "Error: Only triangular faces are supported. "
                          << "Skipping face " << f << " in shape." << std::endl;
                indexOffset += fv;
                continue;
            }

            Triangle triangle{};
            // Assign material ID from the material map
            int objMaterialId = shape.mesh.material_ids[f];
            if (materialIDMap.find(objMaterialId) != materialIDMap.end()) {
                triangle.materialId = materialIDMap[objMaterialId];
            } else {
                triangle.materialId = geom.materialid;
            }

            // Assign vertex, normal, and UV indices
            for (size_t v = 0; v < 3; ++v) {
                const tinyobj::index_t& idx = shape.mesh.indices[indexOffset + v];

                triangle.v[v] = static_cast<int>(baseVertexIndex) + idx.vertex_index;
                triangle.n[v] = (idx.normal_index >= 0)
                                ? (static_cast<int>(baseNormalIndex) + idx.normal_index)
                                : -1;
                triangle.uv[v] = (idx.texcoord_index >= 0)
                                 ? (static_cast<int>(baseUVIndex) + idx.texcoord_index)
                                 : -1;
            }

            glm::vec3 v0 = vertices[triangle.v[0]];
            glm::vec3 v1 = vertices[triangle.v[1]];
            glm::vec3 v2 = vertices[triangle.v[2]];

            // Compute the Axis-Aligned Bounding Box (AABB) for the triangle
            triangle.aabb.minBounds = glm::min(glm::min(v0, v1), v2);
            triangle.aabb.maxBounds = glm::max(glm::max(v0, v1), v2);
            triangle.aabb.centroid = (triangle.aabb.minBounds + triangle.aabb.maxBounds) * 0.5f;

            triangles.emplace_back(triangle);
            indexOffset += fv;
            geom.meshCount++;
        }
    }

#if DEBUG
    std::cout << "Vertices: " << vertices.size() << std::endl;
    std::cout << "Normals: " << normals.size() << std::endl;
    std::cout << "UVs: " << uvs.size() << std::endl;
    std::cout << "Triangles: " << geom.meshCount << std::endl;
#endif

    if (geom.meshCount > 0) {
        geom.bvhRootIndex = buildBVHRecursive(geom.meshStart, geom.meshCount, 0);
    }

    return 0;
}

void Scene::buildBVH() {
    if (!triangles.empty()) {
        // Build the hierarchical BVH for all triangles
        int bvhRoot = buildBVHRecursive(0, triangles.size(), 0);
        std::cout << "Built hierarchical BVH with " << bvh.size() << " nodes." << std::endl;

        linearBVH.resize(bvh.size());

        // Flatten the hierarchical BVH into the linear BVH
        int offset = 0;
        flattenBVHTree(bvhRoot, &offset);
        std::cout << "Flattened BVH into " << linearBVH.size() << " linear nodes." << std::endl;
    }
}

int Scene::buildBVHRecursive(int start, int count, int depth) {
    BVHNode node;

    // compute the bounding box of the current set of triangles
    for (int i = start; i < start + count; ++i) {
        node.aabb.minBounds = glm::min(node.aabb.minBounds, triangles[i].aabb.minBounds);
        node.aabb.maxBounds = glm::max(node.aabb.maxBounds, triangles[i].aabb.maxBounds);
    }
    node.aabb.centroid = (node.aabb.minBounds + node.aabb.maxBounds) * 0.5f;

    // compute bound of primitive centroids, choose split dimension dim
    AABB centroidBounds;
    for (int i = start; i < start + count; ++i) {
        centroidBounds.minBounds = glm::min(centroidBounds.minBounds, triangles[i].aabb.centroid);
        centroidBounds.maxBounds = glm::max(centroidBounds.maxBounds, triangles[i].aabb.centroid);
    }
    // find maximum extent
    glm::vec3 extent = centroidBounds.maxBounds - centroidBounds.minBounds;
    int dim = 0;
    if (extent.y > extent.x) {
        dim = 1;
    }
    if (extent.z > extent.y && extent.z > extent.x) {
        dim = 2;
    }
    node.axis = dim;

    // determine leaf node
    if (count <= 4 || centroidBounds.maxBounds[dim] == centroidBounds.minBounds[dim] || depth >= MAX_DEPTH) {
        node.startIndex = start;
        node.primitiveCount = count;
        node.leftChildIndex = -1;
        node.rightChildIndex = -1;
        bvh.push_back(node);
        return bvh.size() - 1;
    }

    // partition primitives into two sets and build children
    int mid = start + (count / 2);
    std::nth_element(&triangles[start], &triangles[mid], &triangles[start + count],
                     [dim](const Triangle& a, const Triangle& b) {
                         return a.aabb.centroid[dim] < b.aabb.centroid[dim];
                     });

    // Set up the internal node
    node.leftChildIndex =buildBVHRecursive(start, mid - start, depth + 1);
    node.rightChildIndex = buildBVHRecursive(mid, count - (mid - start), depth + 1);
    node.startIndex = -1;

    // Add the internal node to the BVH
    bvh.push_back(node);
    return bvh.size() - 1;
}

int Scene::flattenBVHTree(int nodeIndex, int *offset) {
    const BVHNode& node = bvh[nodeIndex];
    LinearBVHNode& linearNode = linearBVH[*offset];

    linearNode.aabb = node.aabb;
    int currentOffset = (*offset)++;

    if (node.primitiveCount > 0) {
        linearNode.primitivesOffset = node.startIndex;
        linearNode.nPrimitives = node.primitiveCount;
    } else {
        linearNode.axis = node.axis;
        linearNode.nPrimitives = 0;

        flattenBVHTree(node.leftChildIndex, offset);

        linearNode.secondChildOffset = flattenBVHTree(node.rightChildIndex, offset);
    }

    return currentOffset;
}




