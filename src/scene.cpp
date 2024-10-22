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
std::vector<BVHNode> Scene::getBvhNode()
{
    if (!jsonLoadedNonCuda)
    {
        std::cout << "loadJSON not called before CUDA load mesh!\n";
        exit(EXIT_FAILURE);
    }
    std::ifstream f(jsonName_str);
    json data = json::parse(f);

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "mesh")
        {
            return bvhNode;
        }
    }
    return std::vector<BVHNode>();
}
std::vector<tinygltf::Image> Scene::getImages()
{
    if (!jsonLoadedNonCuda)
    {
        std::cout << "loadJSON not called before CUDA load mesh!\n";
        exit(EXIT_FAILURE);
    }
    std::ifstream f(jsonName_str);
    json data = json::parse(f);

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "mesh")
        {
            return images;
        }
    }
    return std::vector<tinygltf::Image>();
}

std::vector<MeshTriangle>* Scene::getTriangleBuffer()
{
    if (!jsonLoadedNonCuda)
    {
        std::cout << "loadJSON not called before CUDA load mesh!\n";
        exit(EXIT_FAILURE);
    }
    std::ifstream f(jsonName_str);
    json data = json::parse(f);

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "mesh")
        {
            return triangles;
        }
    }
    return nullptr;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    jsonLoadedNonCuda = true;
    isBVHEmpty = true;
    jsonName_str = jsonName;
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    int idx = 0;
    //materials;
    materials.clear();
    materials.resize(materialsData.size());
    
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        std::cout << "mat name: " << name << "\n";
        Material newMaterial{};

        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        if (p["TYPE"] == LIGHT)
        {
            newMaterial.type = LIGHT;
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == DIFFUSE_REFL)
        {
            newMaterial.type = DIFFUSE_REFL;
        }
        else if (p["TYPE"] == SPEC_REFL)
        {
            newMaterial.type = SPEC_REFL;
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == SPEC_TRANS)
        {
            newMaterial.type = SPEC_TRANS;
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == SPEC_GLASS)
        {
            newMaterial.type = SPEC_GLASS;
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == MICROFACET_REFL)
        {
            newMaterial.type = MICROFACET_REFL;
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == DIAMOND)
        {
            newMaterial.type = DIAMOND;
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == CERAMIC)
        {
            newMaterial.type = CERAMIC;
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.roughness = roughness;
        }
        else if (p["TYPE"] == MATTEBLACK)
        {
            newMaterial.type = MATTEBLACK;
        }
        else {
            std::cout << "UNKNOWN MATERIAL TYPE ERROR\n";
            exit(EXIT_FAILURE);
        }
        MatNameToID[name] = idx;
        materials[idx] = newMaterial;
        idx++;
    }
    const auto& objectsData = data["Objects"];

    int lightIdx = 0;
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "mesh")
        {
            if (loader == nullptr) {
                loader = std::make_unique<glTFLoader>();
            }
            const auto& filePath = p["FILEPATH"];
            bool retLoadModel = loader->loadModel(filePath);
            std::cout << "loaded model!\n";
            if (!retLoadModel) {
                std::cout << "Error loading gltf model!\n";
                exit(EXIT_FAILURE);
            }

            //images:
            std::vector<tinygltf::Image> imgs_tmp = loader->getImages();
            std::cout << "got images\n";
            images = imgs_tmp;

            triangles = loader->getTriangles();
            std::cout << "applying world trans\n";

            if (triangles != nullptr) {
                for (int i = 0; i < triangles->size(); i++) {
                    Geom newGeom;
                    //newGeom.type = TRI;

                    //newGeom.triangle_index = i;

                    newGeom.materialid = MatNameToID[p["MATERIAL"]];
                    const auto& trans = p["TRANS"];
                    const auto& rotat = p["ROTAT"];
                    const auto& scale = p["SCALE"];
                    newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
                    newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                    newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);

                    newGeom.transform = utilityCore::buildTransformationMatrix(
                        newGeom.translation, newGeom.rotation, newGeom.scale);

                    MeshTriangle transformedTri = (*triangles)[i];

                    //MODIFY MESHTRIANGLE HERE!!! (DONT FORGET UV)
                    transformedTri.v0 = glm::vec3(newGeom.transform * glm::vec4((*triangles)[i].v0, 1.0f));
                    transformedTri.v1 = glm::vec3(newGeom.transform * glm::vec4((*triangles)[i].v1, 1.0f));
                    transformedTri.v2 = glm::vec3(newGeom.transform * glm::vec4((*triangles)[i].v2, 1.0f));

                    (*triangles)[i] = transformedTri;

                    //THESE ARE THE FINAL, TRANSFORMED TRIANGLES! We should be forming BVH based on these BABIES!
                }
                bvhNode = loader->getBVHTree();
                isBVHEmpty = false;
            }

            std::cout << "applied world space trans to all triangles!\n";
        }
        else if (type == "arealight") {
            AreaLight newLight;

            const auto& Le = p["LE"];
            const auto& Emittance = p["EMITTANCE"];
            const auto& shapeType = p["SHAPETYPE"];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];

            newLight.Le = glm::vec3(Le[0], Le[1], Le[2]);
            newLight.emittance = Emittance;
            newLight.ID = lightIdx;
            newLight.shapeType = shapeType;

            newLight.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newLight.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newLight.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newLight.transform = utilityCore::buildTransformationMatrix(
                newLight.translation, newLight.rotation, newLight.scale);

            newLight.inverseTransform = glm::inverse(newLight.transform);
            newLight.invTranspose = glm::mat4(glm::inverseTranspose(newLight.transform));

            areaLights.push_back(newLight);
            lightIdx++;
        }
        else if (type == "sphere") {
            Geom newGeom;
            newGeom.type = G_SPHERE;
            newGeom.materialid = p["MATERIAL"];
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
        else {
            std::cout << "Unknown Geom type not mesh or light. Exit fail.\n";
            exit(EXIT_FAILURE);
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
