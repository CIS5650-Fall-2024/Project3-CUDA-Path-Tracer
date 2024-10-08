#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_obj_loader.h"
#include "ImGui/imgui.h"

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

    if (data.contains("Environment")) {
        const auto& envMapData = data["Environment"];
        this->environmentMap.brightness = 1;
        if (envMapData.contains("FILENAME")) {
            this->environmentMap.isTextureValid = 1;

            std::size_t pos = jsonName.rfind("/");  // Find the last '/'
            std::string fileName = envMapData["FILENAME"];
            std::string texturePath = jsonName.substr(0, pos + 1) + "Mesh/" + "Textures/" + fileName +".hdr";

            this->environmentMap.imgData = stbi_loadf(texturePath.c_str(), &(this->environmentMap.width), &(this->environmentMap.height), &(this->environmentMap.channel), 0);
            if (!(this->environmentMap.imgData)) {
                std::cerr << "Failed to load HDR map!" << std::endl;
            }

            if (envMapData.contains("BRIGHTNESS")) {
                this->environmentMap.brightness = envMapData["BRIGHTNESS"];
            }
        }
        else {
            this->environmentMap.isTextureValid = 0;
            if (envMapData.contains("COLOR")) {
                const auto& col = envMapData["COLOR"];
                this->environmentMap.color = glm::vec3(col[0], col[1], col[2]);
            }
            else {
                this->environmentMap.color = glm::vec3(0.f);
            }
        }
    }
    else {
        this->environmentMap.isTextureValid = 0;
        this->environmentMap.color = glm::vec3(0.f);
    }

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.baseColorTextIdx = -1;
        newMaterial.bumpMapTextIdx = -1;
        // TODO: handle materials loading differently
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
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0;
        }
        else if (p["TYPE"] == "Transmissive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0;
            newMaterial.indexOfRefraction = 1.55;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];

        if (type == "cube")
        {
            Geom newGeom;

            newGeom.type = CUBE;

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
        else if (type == "sphere")
        {
            Geom newGeom;

            newGeom.type = SPHERE;

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
        else if (type == "custom") 
        {
            tinyobj::ObjReaderConfig reader_config;
            tinyobj::ObjReader reader;

            // Parse to get object name and read object
            std::size_t pos = jsonName.rfind("/");  // Find the last '/'
            std::string objName = jsonName.substr(0, pos + 1) + "Mesh/" + p["PATH"].get<std::string>() + ".obj";
            if (!reader.ParseFromFile(objName)) {
                if (!reader.Error().empty()) {
                    std::cerr << "Error: " << reader.Error() << std::endl;
                }
                exit(1);
            }

            // Get object data
            auto& attrib = reader.GetAttrib();
            auto& shapes = reader.GetShapes();
            auto& materials = reader.GetMaterials();
            std::vector<int> matIdxList;
            
            for (int materialIdx = 0; materialIdx < materials.size(); materialIdx++) {
                Material newMaterial{};
                newMaterial.baseColorTextIdx = -1;
                newMaterial.bumpMapTextIdx = -1;

                // load baseColor texture. If baseColor texture name is not empty, load the texture
                std::string textName = materials[materialIdx].diffuse_texname;                
                if (!textName.empty()) {
                    std::string texturePath = jsonName.substr(0, pos + 1) + "Mesh/" + "Textures/" + textName;
                    Texture diffuseTexture{};
                    diffuseTexture.imgData = stbi_load(texturePath.c_str(), &diffuseTexture.width, &diffuseTexture.height, &diffuseTexture.channel, 0);
                    if (diffuseTexture.imgData == nullptr) {
                        std::cerr << "Failed to load texture: " << texturePath << std::endl;
                    }
                    newMaterial.baseColorTextIdx = this->textures.size();
                    this->textures.push_back(diffuseTexture);
                }

                // load normal map texture. If bump map texture name is not empty, load the texture
                textName = materials[materialIdx].bump_texname;
                if (!textName.empty()) {
                    std::string texturePath = jsonName.substr(0, pos + 1) + "Mesh/" + "Textures/" + textName;
                    Texture bumpTexture{};
                    bumpTexture.imgData = stbi_load(texturePath.c_str(), &bumpTexture.width, &bumpTexture.height, &bumpTexture.channel, 0);
                    if (bumpTexture.imgData == nullptr) {
                        std::cerr << "Failed to load texture: " << texturePath << std::endl;
                    }
                    newMaterial.bumpMapTextIdx = this->textures.size();
                    this->textures.push_back(bumpTexture);
                }

                newMaterial.color = glm::vec3(materials[materialIdx].diffuse[0], materials[materialIdx].diffuse[1], materials[materialIdx].diffuse[2]);
                //newMaterial.hasRefractive = 1;
                newMaterial.indexOfRefraction = materials[materialIdx].ior;
                matIdxList.push_back(this->materials.size());
                this->materials.push_back(newMaterial);
            }
            
            // for each shape:
            //  record index of the mesh in geoms
            //  for each index:
            //      push back each vertex based on its index (record pos, nor, uv)
            for (int shapeIdx = 0; shapeIdx < shapes.size(); shapeIdx++) {
                // record mesh index in geoms                
                // record vertices
                Geom newGeom{};
                newGeom.type = CUSTOM;

                auto& currMesh = shapes[shapeIdx].mesh;
                newGeom.vertex_indices.x = this->vertices.size();

                for (size_t idx = 0; idx < currMesh.indices.size(); idx++) {
                    Vertex v{};
                    for (int i = 0; i < 3; i++) {
                        v.pos[i] = attrib.vertices[currMesh.indices[idx].vertex_index * 3 + i];
                        v.normal[i] = attrib.normals[currMesh.indices[idx].normal_index * 3 + i];
                        if (i < 2) {
                            v.uv[i] = attrib.texcoords[currMesh.indices[idx].texcoord_index * 2 + i];
                        }
                    }
                    this->vertices.push_back(v);
                }
                newGeom.vertex_indices.y = this->vertices.size() - 1;

                // new material for geoms
                bool useExtMat = p["EXTERNAL_MATERIAL"];
                if (useExtMat) {
                    int tmp = currMesh.material_ids[0];
                    if (tmp >= matIdxList.size()) {
                        std::cerr << "Failed to load material: please turn off EXTERNAL_MATERIAL when not needed" << std::endl;
                        exit(1);
                    }
                    newGeom.materialid = matIdxList[currMesh.material_ids[0]];
                }
                else {
                    newGeom.materialid = MatNameToID[p["MATERIAL"]];
                }

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
        }
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
