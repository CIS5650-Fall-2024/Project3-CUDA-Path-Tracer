#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "stb_image.h"
#include <cstdlib>
using json = nlohmann::json;

#define LOAD_GLTF 0

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
#if LOAD_GLTF
        loadFromGLTF(filename);
#endif
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
        // TODO: handle materials loading differently
        if (p["TYPE"] == "SKIN")
        {
            newMaterial.color = glm::vec3(p["RGB"][0], p["RGB"][1], p["RGB"][2]);
            newMaterial.type = SKIN;
            newMaterial.subsurfaceScattering = p["SUBSURFACE_SCATTERING"];
        }
        else if (p["TYPE"] == "GGX")
        {
            newMaterial.color = glm::vec3(p["RGB"][0], p["RGB"][1], p["RGB"][2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasReflective = 1.0f - p["ROUGHNESS"];
            newMaterial.hasRefractive = 0.0f;
            newMaterial.emittance = 0.0f;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.type = GGX;
        }
        else if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.type = LIGHT;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            const float& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - roughness;
            newMaterial.hasRefractive = 0.0f;
            newMaterial.emittance = 0.0f;
            newMaterial.type = SPECULAR;
        }
        else if (p["TYPE"] == "Dielectric") {
            newMaterial.color = glm::vec3(p["RGB"][0], p["RGB"][1], p["RGB"][2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasReflective = 0.0f;
            newMaterial.emittance = 0.0f;
            newMaterial.specular.color = glm::vec3(1, 1, 1);
            newMaterial.type = DIELECTRIC;
        }
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
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        newGeom.matType = materials[newGeom.materialid].type;
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
    camera.aperture = cameraData["APERTURE"];
    camera.focusDistance = cameraData["FOCUS_DISTANCE"];

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

void Scene::LoadMaterialsFromFromGLTF() {
  meshesMaterials = gltfLoader->LoadMaterials(meshesTextures);
}

void Scene::LoadTexturesFromGLTF() {
  meshesTextures = gltfLoader->LoadTextures();
  unsigned int textureCount = meshesTextures.size();
  //load();
}

void Scene::loadFromGLTF(const std::string& gltfFilename)
{
    // gltfLoader = std::make_unique<GLTFLoader>(string("scenes/BoomBox/BoomBox.gltf"));
    gltfLoader = std::make_unique<GLTFLoader>(string("C:/Users/carlo/Code/carlos-lopez-garces/CIS5650/Penn-CIS-5650-Project3-CUDA-Path-Tracer/scenes/FlightHelmet/FlightHelmet.gltf"));
    // gltfLoader = std::make_unique<GLTFLoader>(string("scenes/Sponza/Sponza.gltf"));
    // gltfLoader = std::make_unique<GLTFLoader>(string("scenes/FlightHelmet/FlightHelmet.gltf"));
    gltfLoader->LoadModel();
    LoadTexturesFromGLTF();
    LoadMaterialsFromFromGLTF();
    loadGeometryFromGLTF();
}

__host__ __device__ AABB unionAABB(const AABB& a, const AABB& b) {
    AABB result;
    result.min = glm::min(a.min, b.min);
    result.max = glm::max(a.max, b.max);
    return result;
}

void buildBVH(
    std::vector<BVHNode>& bvhNodes,  
    int nodeIdx,                     
    int* objectIndices,              
    int start,                       
    int end,                         
    const std::vector<AABB>& objectBounds
) {
    bvhNodes[nodeIdx].bounds = objectBounds[objectIndices[start]];
    for (int i = start + 1; i < end; ++i) {
        bvhNodes[nodeIdx].bounds = unionAABB(bvhNodes[nodeIdx].bounds, objectBounds[objectIndices[i]]);
    }

    int numObjects = end - start;

    if (numObjects <= 3) {
        bvhNodes[nodeIdx].isLeaf = true;
        bvhNodes[nodeIdx].start = start;
        bvhNodes[nodeIdx].end = end;
        return;
    }

    glm::vec3 extent = bvhNodes[nodeIdx].bounds.max - bvhNodes[nodeIdx].bounds.min;
    int axis = (extent.x > extent.y) ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);

    std::sort(objectIndices + start, objectIndices + end, [&](int a, int b) {
        return (objectBounds[a].min[axis] + objectBounds[a].max[axis]) / 2.0f <
               (objectBounds[b].min[axis] + objectBounds[b].max[axis]) / 2.0f;
    });

    int mid = (start + end) / 2;

    bvhNodes[nodeIdx].isLeaf = false;
    
    int leftChildIdx = bvhNodes.size();
    bvhNodes.emplace_back();  
    bvhNodes[nodeIdx].left = leftChildIdx;
    buildBVH(bvhNodes, leftChildIdx, objectIndices, start, mid, objectBounds);

    int rightChildIdx = bvhNodes.size();
    bvhNodes.emplace_back();  
    bvhNodes[nodeIdx].right = rightChildIdx;
    buildBVH(bvhNodes, rightChildIdx, objectIndices, mid, end, objectBounds);
}

void Scene::loadGeometryFromGLTF() {
    unsigned int nodeCount = gltfLoader->getNodeCount();

    Material newMaterial{};
    newMaterial.color = glm::vec3(1.0, 0.78, 0.66);
    newMaterial.type = SKIN;
    newMaterial.subsurfaceScattering = 0.8;
    // // newMaterial.type = SPECULAR;
    // newMaterial.roughness = 0.9;
    // newMaterial.indexOfRefraction = 1.5;
    // newMaterial.color = glm::vec3(1, 1, 1);
    // newMaterial.hasRefractive = 1.0f;
    // newMaterial.indexOfRefraction = 1.5;
    // newMaterial.hasReflective = 0.0f;
    // newMaterial.emittance = 0.0f;
    // newMaterial.specular.color = glm::vec3(1, 1, 1);
    // newMaterial.type = DIELECTRIC;
    materials.emplace_back(newMaterial);

    // std::vector<AABB> meshBounds;
    // std::vector<int> meshIndices;
    // int meshIdx = 0;

    for (int nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
        unsigned int primCount = gltfLoader->getPrimitiveCount(nodeIdx);

        for (int primIdx = 0; primIdx < primCount; ++primIdx) {
            Geom newGeom;
            newGeom.type = MESH;

            GLTFPrimitiveData gltfData = gltfLoader->LoadPrimitive(nodeIdx, primIdx);

            newGeom.offset = meshesPositions.size();
            newGeom.count = gltfData.positions.size();
            meshesPositions.insert(meshesPositions.end(), gltfData.positions.begin(), gltfData.positions.end());

            newGeom.indexOffset = meshesIndices.size();
            newGeom.indexCount = gltfData.indices.size();
            meshesIndices.insert(meshesIndices.end(), gltfData.indices.begin(), gltfData.indices.end());

            if (!gltfData.normals.empty()) {
                meshesNormals.insert(meshesNormals.end(), gltfData.normals.begin(), gltfData.normals.end());
            }

            if (!gltfData.uvs.empty()) {
                meshesUVs.insert(meshesUVs.end(), gltfData.uvs.begin(), gltfData.uvs.end());
            }

            newGeom.materialid = materials.size() - 1;
            // newGeom.triangleCount = gltfData.indices.size() / 3;

            newGeom.transform = utilityCore::buildTransformationMatrix(
                glm::vec3(0, 1, 1), glm::vec3(0, 45, 0), glm::vec3(7, 7, 7)
            );

            // std::vector<int> triangleIndices(newGeom.triangleCount);
            // for (size_t i = 0; i < newGeom.triangleCount; ++i) {
            //     triangleIndices[i] = i;  // Index of each triangle
            // }

            // AABB meshAABB;
            // for (int i = 0; i < newGeom.count; ++i) {
            //     glm::vec3 vertex = meshesPositions[newGeom.offset + i];
            //     glm::vec3 transformedVertex = glm::vec3(newGeom.transform * glm::vec4(vertex, 1.0f));
            //     meshAABB.min = glm::min(meshAABB.min, transformedVertex);
            //     meshAABB.max = glm::max(meshAABB.max, transformedVertex);
            // }

            // std::vector<AABB> triangleBounds(newGeom.triangleCount);
            // for (size_t i = 0; i < newGeom.triangleCount; ++i) {
            //     int idx = newGeom.indexOffset + i * 3;
            //     glm::vec3 v0 = gltfData.positions[gltfData.indices[idx]];
            //     glm::vec3 v1 = gltfData.positions[gltfData.indices[idx + 1]];
            //     glm::vec3 v2 = gltfData.positions[gltfData.indices[idx + 2]];

            //     AABB triangleAABB;
            //     triangleAABB.min = glm::min(v0, glm::min(v1, v2));
            //     triangleAABB.max = glm::max(v0, glm::max(v1, v2));
            //     triangleBounds[i] = triangleAABB;
            // }

            // std::vector<BVHNode> meshBVH;
            // meshBVH.emplace_back();
            // buildBVH(meshBVH, 0, triangleIndices.data(), 0, triangleIndices.size(), triangleBounds);

            // newGeom.triangleIndices = new int[newGeom.triangleCount]; 
            // std::memcpy(newGeom.triangleIndices, triangleIndices.data(), newGeom.triangleCount * sizeof(int));

            // newGeom.meshBVH = new BVHNode[meshBVH.size()]; 
            // newGeom.meshBVHCount = meshBVH.size();
            // std::memcpy(newGeom.meshBVH, meshBVH.data(), meshBVH.size() * sizeof(BVHNode));

            // newGeom.bvhRoot = 0;

            // meshBounds.push_back(meshAABB);
            // meshIndices.push_back(meshIdx++);

            geoms.push_back(newGeom);
        }
    }

    // std::vector<BVHNode> topLevelBVHNodes;
    // topLevelBVHNodes.emplace_back();  
    // buildBVH(topLevelBVHNodes, 0, meshIndices.data(), 0, meshIndices.size(), meshBounds);

    // topLevelBVH = new BVHNode[topLevelBVHNodes.size()];  
    // topLevelBVHCount = topLevelBVHNodes.size();
    // std::memcpy(topLevelBVH, topLevelBVHNodes.data(), topLevelBVHNodes.size() * sizeof(BVHNode));  
}
