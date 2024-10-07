#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename)
    : useDirectLighting(false)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".gltf")
    {
        loadFromGltf(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string &jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto &materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto &item : materialsData.items())
    {
        const auto &name = item.key();
        const auto &p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto &col = p["RGB"];
            newMaterial.albedo.value = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto &col = p["RGB"];
            newMaterial.albedo.value = glm::vec3();
            newMaterial.emittance.value = p["EMITTANCE"].get<float>() * glm::vec3(col[0], col[1], col[2]);
            newMaterial.emissiveStrength = 1.f;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto &col = p["RGB"];
            newMaterial.albedo.value = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = true;
            newMaterial.specular.color = newMaterial.albedo.value;
            newMaterial.roughness = p.contains("ROUGHNESS") ? p["ROUGHNESS"].get<float>() : 0.f;
        }
        else if (p["TYPE"] == "Transmissive")
        {
            const auto &col = p["RGB"];
            newMaterial.albedo.value = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = true;
            newMaterial.indexOfRefraction = p.contains("IOR") ? p["IOR"].get<float>() : 1.55f;
        }
        else if (p["TYPE"] == "Fresnel")
        {
            const auto &col = p["RGB"];
            newMaterial.albedo.value = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.albedo.value;
            newMaterial.hasReflective = true;
            newMaterial.hasRefractive = true;
            newMaterial.indexOfRefraction = p.contains("IOR") ? p["IOR"].get<float>() : 1.55f;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    const auto &objectsData = data["Objects"];
    for (const auto &p : objectsData)
    {
        const auto &type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "square")
        {
            newGeom.type = SQUARE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.meshId = meshes.size();
            Mesh mesh;
            mesh.triCount = p["COUNT"];
            mesh.pointOffset = p["OFFSET"];
            meshes.push_back(mesh);
        }
        else
        {
            throw std::invalid_argument("Bad object type");
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto &trans = p["TRANS"];
        const auto &rotat = p["ROTAT"];
        const auto &scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    for (const auto &point : data["Triangles"])
    {
        positions.push_back(glm::vec3(point[0], point[1], point[2]));
    }

    std::sort(geoms.begin(), geoms.end(), [](const Geom &g1, const Geom &g2)
              { return g1.materialid < g2.materialid; });

    const auto &cameraData = data["Camera"];

    RenderState &state = this->state;
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];

    Camera &camera = state.camera;
    float fovy = cameraData["FOVY"];
    const auto &res = cameraData["RES"];
    const auto &pos = cameraData["EYE"];
    const auto &lookat = cameraData["LOOKAT"];
    const auto &up = cameraData["UP"];
    float lensSize = cameraData.contains("LENSSIZE") ? cameraData["LENSSIZE"].get<float>() : 0.f;
    float focalDist = cameraData.contains("FOCALDIST") ? cameraData["FOCALDIST"].get<float>() : 0.f;

    setupCamera(
        glm::ivec2(res[0], res[1]),
        glm::vec3(pos[0], pos[1], pos[2]),
        glm::vec3(lookat[0], lookat[1], lookat[2]),
        fovy,
        glm::vec3(up[0], up[1], up[2]),
        lensSize,
        focalDist);

    createLightIndices();
}

void Scene::setupCamera(glm::ivec2 resolution, glm::vec3 position, glm::vec3 lookAt, float fovy, glm::vec3 up, float lensSize, float focalDist)
{
    auto &camera = state.camera;
    camera.resolution = resolution;
    camera.position = position;
    camera.lookAt = lookAt;
    camera.up = up;
    camera.lensSize = lensSize;
    camera.focalDist = focalDist;

    // calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::createLightIndices()
{
    lightIndices.clear();
    for (int i = 0; i < geoms.size(); i++) {
        const Material& mat = materials[geoms[i].materialid];
        if (mat.emissiveStrength > 0) {
            lightIndices.push_back(i);
        }
    }
}

void Scene::loadFromGltf(const std::string &gltfName)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;

    std::string err;
    std::string warn;

    if (!loader.LoadASCIIFromFile(&model, &err, &warn, gltfName))
    {
        std::cerr << "Err: " << err << std::endl;
        return;
    }
    if (!warn.empty())
    {
        std::cerr << "Warn: " << warn << std::endl;
    }

    int numMats = model.materials.size();
    const auto& mat = model.materials[0];

    for (size_t i = 0; i < model.textures.size(); i++) {
        loadGltfTexture(model, i);
    }

    for (size_t i = 0; i < model.materials.size(); i++)
    {
        loadGltfMaterial(model, i);
    }

    for (size_t i = 0; i < model.meshes.size(); i++)
    {
        loadGltfMesh(model, i);
    }

    for (int node : model.scenes[model.defaultScene].nodes)
    {
        loadGltfNode(model, node);
    }

    setupCamera();
    createLightIndices();
}

void Scene::loadGltfTexture(const tinygltf::Model &model, int textureId)
{
    assert(textureId == texes.size());
    const auto& image = model.images[model.textures[textureId].source];
    size_t numPixels = image.width * image.height;
    TextureData tex {
        .dimensions = glm::ivec2(image.width, image.height),
        .data = std::vector<glm::vec4>()
    };
    tex.data.reserve(numPixels);
    
    for (size_t i = 0; i < numPixels; i ++) {
        glm::vec4 value;
        for (size_t j = 0; j < image.component; j++) {
            value[j] = image.image[i * image.component + j];
        }
        tex.data.push_back(value);
    }

    size_t texsize = tex.data.size();

    texes.push_back(tex);
}

void Scene::loadGltfMaterial(const tinygltf::Model &model, int materialId)
{
    assert(materials.size() == materialId);
    const auto &material = model.materials[materialId];
    Material newMaterial;
    const auto &materialProperties = material.pbrMetallicRoughness;

    if (materialProperties.baseColorFactor.size() > 0)
    {
        newMaterial.albedo.value = glm::vec3(materialProperties.baseColorFactor[0], materialProperties.baseColorFactor[1], materialProperties.baseColorFactor[2]);
    }
    if (materialProperties.baseColorTexture.index >= 0) {
        newMaterial.albedo.negSuccTexInd = -(1 + materialProperties.baseColorTexture.index);
    }
    
    if (material.emissiveFactor.size() > 0) {
        newMaterial.emittance.value = glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]);
        newMaterial.emissiveStrength = 1;
    }

    if (material.emissiveTexture.index >= 0) {
        newMaterial.emittance.negSuccTexInd = -(1 + material.emissiveTexture.index);
        newMaterial.emissiveStrength = 1;
    }

    auto iter = material.extensions.find("KHR_materials_emissive_strength");
    if (iter != material.extensions.end()) {
        auto emissiveStrength = iter->second.Get("emissiveStrength").GetNumberAsDouble();
        newMaterial.emissiveStrength = emissiveStrength;
    }
    materials.push_back(newMaterial);
}

template<typename SrcT, typename DstT>
void loadBuffer(const tinygltf::Model &model, int accessorIndex, std::vector<DstT>& acc) {
    const auto &accessor = model.accessors[accessorIndex];
    const auto &view = model.bufferViews[accessor.bufferView];
    const auto &buffer = model.buffers[view.buffer];
    const SrcT *data = reinterpret_cast<const SrcT*>(&(buffer.data[accessor.byteOffset + view.byteOffset]));
    for (size_t i = 0; i < accessor.count; i++) {
        assert ((const void*)(data + 1) <= (const void*)(buffer.data.data() + buffer.data.size()));
        acc.push_back(data[i]);
    }
}

void Scene::loadGltfMesh(const tinygltf::Model &model, int meshId)
{
    assert(meshes.size() == meshId);
    const auto &mesh = model.meshes[meshId];
    // TODO: > 1 primitive per mesh? Currently assumes all meshes have one material
    assert(mesh.primitives.size() == 1);
    for (const auto &prim : mesh.primitives)
    {
        Mesh mesh;
        mesh.pointOffset = positions.size();
        loadBuffer<glm::vec3>(model, prim.attributes.at("POSITION"), positions);


        mesh.triCount = model.accessors[prim.indices].count / 3;
        mesh.indOffset = indices.size();
        switch (model.accessors[prim.indices].componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                loadBuffer<unsigned short>(model, prim.indices, indices);
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                loadBuffer<unsigned int>(model, prim.indices, indices);
                break;
            }
            default: {
                throw std::invalid_argument("Bad primitive component type");
            }
        }
        
        auto& material = model.materials[prim.material];
        const std::string texCoordPrefix = "TEXCOORD_";
        for (auto& attribute : prim.attributes) {
            if (attribute.first.find(texCoordPrefix) != 0) {
                continue;
            }
            int index = std::stoi(attribute.first.substr(texCoordPrefix.size()));
            bool textureUsed = false;
            
            if (material.pbrMetallicRoughness.baseColorTexture.texCoord == index) {
                textureUsed = true;
                mesh.albedoUvOffset = uvs.size();
            }    
            if (material.emissiveTexture.texCoord == index) {
                textureUsed = true;
                mesh.emissiveUvOffset = uvs.size();
            }
            if (textureUsed) {
                loadBuffer<glm::vec2>(model, attribute.second, uvs);
            }
        }
        meshes.push_back(mesh);
    }
}

void Scene::loadGltfNode(const tinygltf::Model &model, int node)
{
    for (int nodeIndex : model.scenes[model.defaultScene].nodes)
    {
        const auto &node = model.nodes[nodeIndex];
        Geom newGeom;
        if (node.translation.size() == 3)
        {
            newGeom.translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        }
        if (node.rotation.size() == 4)
        {
            glm::quat rotation;
            rotation.x = node.rotation[0];
            rotation.y = node.rotation[1];
            rotation.z = node.rotation[2];
            rotation.w = node.rotation[3];
            newGeom.rotation = glm::degrees(glm::eulerAngles(rotation));
        }
        if (node.scale.size() == 3)
        {
            newGeom.scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        }
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::transpose(newGeom.inverseTransform);

        if (node.mesh >= 0)
        {
            newGeom.meshId = node.mesh;
            newGeom.materialid = model.meshes[node.mesh].primitives[0].material;
            geoms.push_back(newGeom);
        }

        for (int child : node.children)
        {
            // TODO: Handle children
        }
    }
}