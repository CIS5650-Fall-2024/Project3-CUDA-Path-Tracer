#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "stb_image.h"
#include "stb_image_write.h"
#include "json.hpp"
#include "scene.h"
#include "bvh.h"
#include "tiny_gltf.h"
#define TINYGLTF_IMPLEMENTATION
using json = nlohmann::json;

Scene::Scene(std::string filename)
{
    sceneDev = new SceneDev();
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        std::cout << "Couldn't read from " << filename << std::endl;
        exit(-1);
    }
}

Scene::~Scene()
{
    if (sceneDev)
    {
        sceneDev->freeCudaMemory();
        delete sceneDev;
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    sceneFileName = jsonName;
    std::ifstream fp_in(sceneFileName);
    json data = json::parse(fp_in);
    
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

    if (cameraData.contains("ENVMAP"))
    {
        skyboxPath = cameraData["ENVMAP"];
        //loadTextureFile(skyboxPath, sceneDev->envMap);
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
    state.albedo.resize(arraylen);
    state.normal.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    std::fill(state.albedo.begin(), state.albedo.end(), glm::vec3());
    std::fill(state.normal.begin(), state.normal.end(), glm::vec3());
}

void Scene::loadSceneModels()
{
    loadEnvMap(skyboxPath);

    std::ifstream fp_in(sceneFileName);
    json data = json::parse(fp_in);
    const auto& materialsData = data["Materials"];
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Lambertian;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.type = Light;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Specular;
        }
        else if (p["TYPE"] == "Microfacet")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Microfacet;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.ior = p["IOR"];
            newMaterial.metallic = p["METALLIC"];
        }
        else if (p["TYPE"] == "MetallicWorkflow")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MetallicWorkflow;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.metallic = p["METALLIC"];
        }
        else if (p["TYPE"] == "Dielectric")
        {
            const auto& col = p["RGB"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = Dielectric;
            newMaterial.ior = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        int currStart = objects.size();

        if (type == "cube")
        {
            objects.push_back(Object());
            Object& newObject = objects.back();
            newObject.type = CUBE;
        }
        else if (type == "sphere")
        {
            objects.push_back(Object());
            Object& newObject = objects.back();
            newObject.type = SPHERE;
        }
        else if (type == "obj")
        {
            loadObj(p["PATH"]);
        }
        for (int i = currStart; i < objects.size(); ++i)
        {
            Object& newObject = objects[i];
            if (newObject.materialid == -1) newObject.materialid = MatNameToID[p["MATERIAL"]];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newObject.transforms.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newObject.transforms.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newObject.transforms.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newObject.transforms.transform = utilityCore::buildTransformationMatrix(
                newObject.transforms.translation, newObject.transforms.rotation, newObject.transforms.scale);
            newObject.transforms.inverseTransform = glm::inverse(newObject.transforms.transform);
            newObject.transforms.invTranspose = glm::inverseTranspose(newObject.transforms.transform);
        }
    }
    fp_in.close();
}

bool Scene::loadObj(const std::string& objPath)
{
    std::cout << "Start loading Obj file: " << objPath << " ..." << std::endl;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string err;
    std::string mtlPath = objPath.substr(0, objPath.find_last_of('/') + 1);
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &mats, &err, objPath.c_str(), mtlPath.c_str());

    if (!err.empty()) std::cerr << err << std::endl;
    if (!ret)  return false;

    if (!mats.empty()) loadObjMaterials(mtlPath, mats);

    bool hasUV = !attrib.texcoords.empty();
    bool hasNor = !attrib.normals.empty();
    for (const auto& shape : shapes)
    {
        objects.push_back(Object());
        Object& newObj = objects.back();
        for (auto idx : shape.mesh.indices)
        {
            newObj.meshData.vertices.push_back(*((glm::vec3*)attrib.vertices.data() + idx.vertex_index));
            newObj.meshData.normals.push_back(hasNor ? *((glm::vec3*)attrib.normals.data() + idx.normal_index) : glm::vec3(0.f));
            newObj.meshData.uvs.push_back(hasUV ?
                *((glm::vec2*)attrib.texcoords.data() + idx.texcoord_index) :
                glm::vec2(0.f)
            );
        }
        int mid = shape.mesh.material_ids[0];
        if (mid != -1)
            newObj.materialid = MatNameToID[mats[mid].name];
        else
            newObj.materialid = -1;
        std::cout << newObj.meshData.vertices.size() << " vertices loaded" << std::endl;
    }


}

void Scene::loadObjMaterials(const std::string& mtlPath, std::vector<tinyobj::material_t>& mats)
{
    if (!mats.empty())
    {
        for (const auto& mat : mats)
        {
            std::string name = mat.name;
            Material newMaterial;
            
            newMaterial.albedo = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
            newMaterial.roughness = mat.illum < 3 ? 1.f : glm::max(1e-9f, mat.roughness);
            newMaterial.metallic = mat.metallic;
            newMaterial.ior = mat.ior;
            newMaterial.emittance = mat.emission[0];

            switch (mat.illum)
            {
            case 7:
                newMaterial.type = Dielectric;
                newMaterial.ior = mat.ior;
                //newMaterial.albedo = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
                break;
            case 4:
                newMaterial.type = Dielectric;
                newMaterial.ior = 1.f;
                break;
            case 11:
                newMaterial.type = Specular;
                break;
                //newMaterial.albedo = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
            default:
                newMaterial.type = Lambertian;
                break;
            }

            if (mat.diffuse_texname != "")
            {
                std::string texPath = mtlPath + mat.diffuse_texname;
                loadTextureFile(texPath, newMaterial.albedoMap);
            }

            if (mat.normal_texname != "")
            {
                std::string texPath = mtlPath + mat.normal_texname;
                loadTextureFile(texPath, newMaterial.normalMap);
            }

            MatNameToID[name] = materials.size();
            materials.emplace_back(newMaterial);

        }
    }
}

void Scene::loadTextureFile(const std::string& texPath, cudaTextureObject_t& texObj)
{
    std::printf("Start loading %s\n", texPath.c_str());
    std::string postfix = texPath.substr(texPath.find_last_of('.') + 1);
    std::string name = texPath.substr(texPath.find_last_of('/') + 1);

    // if already loaded, use it
    if (TextureNameToID.find(name) != TextureNameToID.end())
    {
        texObj = TextureNameToID[name];
        return;
    }

    int width, height, channels;
    if (postfix == "hdr")
    {
        float* data = stbi_loadf(texPath.c_str(), &width, &height, &channels, 4);
        if (data)
        {
            createCudaTexture(data, width, height, texObj, true);
            stbi_image_free(data);
        }
        else
        {
            std::printf("Load %s failed: %s\n", texPath.c_str(), stbi_failure_reason());
        }
    }
    else
    {
        stbi_uc* data = stbi_load(texPath.c_str(), &width, &height, &channels, 4);
        if (data)
        {
            createCudaTexture(data, width, height, texObj, false);
            stbi_image_free(data);
        }
        else
        {
            std::printf("Load %s failed: %s\n", texPath.c_str(), stbi_failure_reason());
        }
    }
    // store in map
    TextureNameToID[name] = texObj;
}

void Scene::loadEnvMap(const std::string& texPath)
{
    std::printf("Start loading environment map %s\n", texPath.c_str());
    std::string postfix = texPath.substr(texPath.find_last_of('.') + 1);
    std::string name = texPath.substr(texPath.find_last_of('/') + 1);

    int width, height, channels;
    if (postfix == "hdr")
    {
        float* data = stbi_loadf(texPath.c_str(), &width, &height, &channels, 4);
        if (data)
        {
            createCudaTexture(data, width, height, sceneDev->envMap, true);

            int imgSize = width * height;
            std::vector<float> pdfs(imgSize);
            std::vector<EnvMapDistrib> distrib(imgSize);

            float sumAll = 0.f;
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    int idx = i * width + j;
                    glm::vec3 col(data[4 * idx], data[4 * idx + 1], data[4 * idx + 2]);
                    pdfs[idx] = math::luminance(col) * glm::sin((0.5f + i) / height * PI);
                    sumAll += pdfs[idx];
                }
            }

            sceneDev->envMapHeight = height;
            sceneDev->envMapWidth = width;
            sceneDev->envMapPdfSumInv = 1.f / sumAll;

            // remap range to [0, imgSize]
            float sumInv = (float)imgSize / sumAll;
            for (int i = 0; i < imgSize; ++i)
            {
                pdfs[i] = pdfs[i] * sumInv;
            }

            uint32_t imgIdx = 0;
            uint32_t cdfIdx = 0;
            float currSum = 0.f;
            while (imgIdx < imgSize)
            {
                while (cdfIdx < imgSize && currSum < (float)imgIdx + 1.f)
                {
                    currSum += pdfs[cdfIdx++];
                }
                distrib[imgIdx++].cdfID = cdfIdx - 1;
            }

            cudaMalloc(&sceneDev->envMapDistrib, distrib.size() * sizeof(EnvMapDistrib));
            cudaMemcpy(sceneDev->envMapDistrib, distrib.data(), distrib.size() * sizeof(EnvMapDistrib), cudaMemcpyHostToDevice);


            stbi_image_free(data);

        }
        else
        {
            std::printf("Load %s failed: %s\n", texPath.c_str(), stbi_failure_reason());
        }
    }
    else
    {
        std::printf("Only support hdr file\n");
        exit(1);
    }
}

void Scene::createCudaTexture(void* data, int width, int height, cudaTextureObject_t& texObj, bool isHDR)
{
    cudaError_t err;
    int bitWidth = isHDR ? 32 : 8;
    cudaChannelFormatKind format = isHDR ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bitWidth, bitWidth, bitWidth, bitWidth, format);

    cudaArray_t cuArray;
    size_t texSizeByte = width * height * (isHDR ? 16 : 4);
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, data, texSizeByte, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    resDesc.res.linear.desc = channelDesc;
    resDesc.res.linear.sizeInBytes = texSizeByte;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = isHDR ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = 1;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// rearrange primitives for better cache locality
void Scene::scatterPrimitives(std::vector<Primitive>& srcPrim,
    std::vector<PrimitiveDev>& dstPrim,
    std::vector<glm::vec3>& dstVec,
    std::vector<glm::vec3>& dstNor,
    std::vector<glm::vec2>& dstUV)
{
    std::vector<glm::vec3> srcVec(sceneDev->triNum * 3);
    std::vector<glm::vec3> srcNor(sceneDev->triNum * 3);
    std::vector<glm::vec2> srcUV(sceneDev->triNum * 3);

    uint32_t offset = 0;
    for (const auto& model : objects)
    {
        if (model.type == TRIANGLE)
        {
            uint32_t num = model.meshData.vertices.size();
            std::copy(model.meshData.vertices.begin(), model.meshData.vertices.end(), srcVec.begin() + offset);
            std::copy(model.meshData.normals.begin(), model.meshData.normals.end(), srcNor.begin() + offset);
            std::copy(model.meshData.uvs.begin(), model.meshData.uvs.end(), srcUV.begin() + offset);
            offset += num;
        }
    }

    dstPrim.resize(srcPrim.size());
    dstVec.resize(sceneDev->triNum * 3);
    dstNor.resize(sceneDev->triNum * 3);
    dstUV.resize(sceneDev->triNum * 3);

    uint32_t curr = 0;
    for (uint32_t i = 0; i < sceneDev->primNum; ++i)
    {
        uint32_t primID = srcPrim[i].primId;
        if (primID < sceneDev->triNum)
        {
            dstVec[3 * curr] = srcVec[3 * primID];
            dstVec[3 * curr + 1] = srcVec[3 * primID + 1];
            dstVec[3 * curr + 2] = srcVec[3 * primID + 2];
            dstNor[3 * curr] = srcNor[3 * primID];
            dstNor[3 * curr + 1] = srcNor[3 * primID + 1];
            dstNor[3 * curr + 2] = srcNor[3 * primID + 2];
            dstUV[3 * curr] = srcUV[3 * primID];
            dstUV[3 * curr + 1] = srcUV[3 * primID + 1];
            dstUV[3 * curr + 2] = srcUV[3 * primID + 2];
            srcPrim[i].primId = curr;
            ++curr;
        }
        dstPrim[i].primId = srcPrim[i].primId;
        dstPrim[i].materialId = srcPrim[i].materialId;
    }

}

void Scene::buildDevSceneData()
{
    uint32_t triNum = 0;
    uint32_t primNum = 0;

    // calculate prim num and apply transformations
    for (auto& model : objects)
    {
        if (model.type == TRIANGLE)
        {
            uint32_t num = model.meshData.vertices.size() / 3;
            triNum += num;
            primNum += num;
            for (uint32_t i = 0; i < model.meshData.vertices.size(); ++i)
            {
                model.meshData.vertices[i] = glm::vec3(model.transforms.transform * glm::vec4(model.meshData.vertices[i], 1.f));
                if (model.meshData.normals[0] == glm::vec3(0))
                    model.meshData.normals[i] = glm::normalize(glm::vec3(model.transforms.invTranspose * glm::vec4(model.meshData.normals[i], 1.f)));
            }
        }
        else
        {
            geoms.push_back(Geom());
            Geom& tmp = geoms.back();
            tmp.type = model.type;
            tmp.materialid = model.materialid;
            tmp.transforms = model.transforms;
            ++primNum;
        }
    }

    sceneDev->primNum = primNum;
    sceneDev->triNum = triNum;
    std::printf("Totally %d primitives loaded", primNum);

    // build primitives
    primitives.resize(primNum);
    uint32_t offset = 0;

    // triangle primitives
    for (auto& model : objects)
    {
        if (model.type == TRIANGLE)
        {
            uint32_t num = model.meshData.vertices.size() / 3;
            for (uint32_t i = 0; i < num; ++i)
            {
                primitives[i + offset].primId = i + offset;
                primitives[i + offset].materialId = model.materialid;
                primitives[i + offset].bbox = AABB::getAABB(model.meshData.vertices[3 * i],
                    model.meshData.vertices[3 * i + 1], model.meshData.vertices[3 * i + 2]);
            }
            offset += num;
        }
    }

    // other primitives
    for (uint32_t i = 0; i < geoms.size(); ++i)
    {
        primitives[triNum + i].primId = triNum + i;
        primitives[triNum + i].materialId = geoms[i].materialid;
        primitives[triNum + i].bbox = AABB::getAABB(geoms[i].type, geoms[i].transforms.transform);
    }

    uint32_t treeSize = 0;
    std::vector<AABB> AABBs;
    std::vector<MTBVHNode> flattenNodes = BVH::buildBVH(*this, AABBs, primNum, triNum, treeSize);
    sceneDev->bvhSize = treeSize;

    std::vector<PrimitiveDev> dstPrim;
    std::vector<glm::vec3> dstVec;
    std::vector<glm::vec3> dstNor;
    std::vector<glm::vec2> dstUV;

    scatterPrimitives(primitives, dstPrim, dstVec, dstNor, dstUV);

    cudaMalloc(&sceneDev->vertices, dstVec.size() * sizeof(glm::vec3));
    cudaMemcpy(sceneDev->vertices, dstVec.data(), dstVec.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->normals, dstNor.size() * sizeof(glm::vec3));
    cudaMemcpy(sceneDev->normals, dstNor.data(), dstNor.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->uvs, dstUV.size() * sizeof(glm::vec2));
    cudaMemcpy(sceneDev->uvs, dstUV.data(), dstUV.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    cudaMalloc(&sceneDev->primitives, dstPrim.size() * sizeof(PrimitiveDev));
    cudaMemcpy(sceneDev->primitives, dstPrim.data(), dstPrim.size() * sizeof(PrimitiveDev), cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->materials, materials.size() * sizeof(Material));
    cudaMemcpy(sceneDev->materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->geoms, geoms.size() * sizeof(Geom));
    cudaMemcpy(sceneDev->geoms, geoms.data(), geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    // copy MTBVH array
    cudaMallocPitch(&sceneDev->bvhNodes, &sceneDev->bvhPitch, treeSize * sizeof(MTBVHNode), 6);
    cudaMemcpy2D(sceneDev->bvhNodes, sceneDev->bvhPitch,
        flattenNodes.data(), treeSize * sizeof(MTBVHNode), treeSize * sizeof(MTBVHNode), 6, cudaMemcpyHostToDevice);
    cudaMalloc(&sceneDev->bvhAABBs, AABBs.size() * sizeof(AABB));
    cudaMemcpy(sceneDev->bvhAABBs, AABBs.data(), AABBs.size() * sizeof(AABB), cudaMemcpyHostToDevice);


    // copy data
    /*offset = 0;
    for (const auto& model : objects)
    {
        if (model.type == TRIANGLE)
        {
            uint32_t num = model.meshData.vertices.size();
            cudaMemcpy(sceneDev->vertices + offset, model.meshData.vertices.data(), num * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            if (!model.meshData.normals.empty())
                cudaMemcpy(sceneDev->normals + offset, model.meshData.normals.data(), num * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            if (!model.meshData.uvs.empty())
                cudaMemcpy(sceneDev->uvs + offset, model.meshData.uvs.data(), num * sizeof(glm::vec2), cudaMemcpyHostToDevice);
            offset += num;
        }
    }*/
}

void SceneDev::freeCudaMemory()
{
    cudaFree(geoms);
    cudaFree(materials);
    cudaFree(primitives);
    cudaFree(bvhNodes);
    cudaFree(bvhAABBs);
    cudaFree(vertices);
    cudaFree(normals);
    cudaFree(uvs);
}

