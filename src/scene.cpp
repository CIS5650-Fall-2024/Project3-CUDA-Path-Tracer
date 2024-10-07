#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include <stb_image.h>
#include <cuda_runtime.h>


#define TINYOBJLOADER_IMPLEMENTATION
#include "./thirdparty/tinyobj_loader/tiny_obj_loader.h"

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
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            //consider as ideal diffuse
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
            //consider as perfect specular
            const auto& col = p["RGB"];
            newMaterial.hasReflective = 1.0f;
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f; 
            newMaterial.indexOfRefraction = p["IOR"]; 
            newMaterial.specular.color = glm::vec3(1.0f);
        }
        else if (p["TYPE"] == "Environment") {
            const std::string envPath = p["HDR_MAP"];
            newMaterial.env_intensity = p["INTENSITY"];
            newMaterial.isEnvironment = true;
            if (!loadTexture_hdr(envPath, newMaterial.envMapData)) {
                std::cerr << "Failed to load env Path for Lighting " << name << "\n";
            }
        }

        if (p.contains("ALBEDO_MAP")) {
            const std::string albedoPath = p["ALBEDO_MAP"];
            if (!loadTexture(albedoPath, newMaterial.albedoMapData)) {
                std::cerr << "Failed to load albedo map for material " << name << "\n";
            }
        }

        if (p.contains("NORMAL_MAP")) {
            const std::string normalPath = p["NORMAL_MAP"];
            if (!loadTexture(normalPath, newMaterial.normalMapData)) {
                std::cerr << "Failed to load normal map for material " << name << "\n";
            }
        }

        /*if (p.contains("SPECULOR_MAP")) {
            const std::string normalPath = p["SPECULOR_MAP"];
            if (!loadTexture(normalPath, newMaterial.normalMapData)) {
                std::cerr << "Failed to load speculor map for material " << name << "\n";
            }
        }*/


        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.materialid = MatNameToID[p["MATERIAL"]];

        // Handle transformations
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

        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;

            // Load mesh data
            const std::string objFilename = p["FILENAME"];
            LoadFromOBJ(objFilename, newGeom);
        }
        
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
    camera.focalDistance = cameraData["FOCAL"];

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

void Scene::LoadFromOBJ(const std::string& filename, Geom& geom){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tinyMaterials;
    std::string err;


    bool ret = tinyobj::LoadObj(
        &attrib,
        &shapes,
        &tinyMaterials,
        &err,
        filename.c_str());     

    if (!ret) {
        std::cerr << "Failed to load OBJ file: " << filename << "\n";
        return;
    }

    std::vector<Triangle> triangles;

    
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;

        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = shapes[s].mesh.num_face_vertices[f];

            // Only process triangles
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            
            tinyobj::index_t idx0 = shapes[s].mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset + 2];

           
            glm::vec3 v0(
                attrib.vertices[3 * idx0.vertex_index + 0],
                attrib.vertices[3 * idx0.vertex_index + 1],
                attrib.vertices[3 * idx0.vertex_index + 2]
            );
            glm::vec3 v1(
                attrib.vertices[3 * idx1.vertex_index + 0],
                attrib.vertices[3 * idx1.vertex_index + 1],
                attrib.vertices[3 * idx1.vertex_index + 2]
            );
            glm::vec3 v2(
                attrib.vertices[3 * idx2.vertex_index + 0],
                attrib.vertices[3 * idx2.vertex_index + 1],
                attrib.vertices[3 * idx2.vertex_index + 2]
            );
            glm::vec2 uv0(0.0f, 0.0f);
            glm::vec2 uv1(0.0f, 0.0f);
            glm::vec2 uv2(0.0f, 0.0f);

            if (idx0.texcoord_index >= 0) {
                uv0 = glm::vec2(
                    attrib.texcoords[2 * idx0.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * idx0.texcoord_index + 1] 
                );
            }
            if (idx1.texcoord_index >= 0) {
                uv1 = glm::vec2(
                    attrib.texcoords[2 * idx1.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * idx1.texcoord_index + 1]
                );
            }
            if (idx2.texcoord_index >= 0) {
                uv2 = glm::vec2(
                    attrib.texcoords[2 * idx2.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * idx2.texcoord_index + 1]
                );
            }
           
            glm::vec4 v0_transformed = geom.transform * glm::vec4(v0, 1.0f);
            glm::vec4 v1_transformed = geom.transform * glm::vec4(v1, 1.0f);
            glm::vec4 v2_transformed = geom.transform * glm::vec4(v2, 1.0f);

            v0 = glm::vec3(v0_transformed) / v0_transformed.w;
            v1 = glm::vec3(v1_transformed) / v1_transformed.w;
            v2 = glm::vec3(v2_transformed) / v2_transformed.w;

           
            glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

           
            Triangle tri;
            tri.v0 = v0;
            tri.v1 = v1;
            tri.v2 = v2;
            tri.normal = normal;
            tri.uv0 = uv0;
            tri.uv1 = uv1;
            tri.uv2 = uv2;

            triangles.push_back(tri);

            index_offset += fv;
        }
    }
    std::vector<BVHNode> bvhNodes;
    buildBVH(bvhNodes, triangles, 0, triangles.size());

    /*cudaMalloc(&geom.bvhNodes, bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(geom.bvhNodes, bvhNodes.data(), bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);*/
    /*geom.numBVHNodes = bvhNodes.size();*/

    geom.numBVHNodes = static_cast<int>(bvhNodes.size());
    geom.bvhNodes = new BVHNode[geom.numBVHNodes];
    std::copy(bvhNodes.begin(), bvhNodes.end(), geom.bvhNodes);


    geom.numTriangles = static_cast<int>(triangles.size());
    geom.triangles = new Triangle[geom.numTriangles];
    std::copy(triangles.begin(), triangles.end(), geom.triangles);
  

    


    std::cout << "Loaded " << geom.numTriangles << " triangles from " << filename << "\n";
}

AABB Scene::calculateAABB(const Triangle& tri) {
    AABB box;
    box.AABBmin = glm::min(glm::min(tri.v0, tri.v1), tri.v2);
    box.AABBmax = glm::max(glm::max(tri.v0, tri.v1), tri.v2);
    return box;

}


int Scene::buildBVH(std::vector<BVHNode>& nodes, std::vector<Triangle>& triangles, int start, int end) {
    BVHNode node;
    node.start = start;
    node.end = end;

    AABB bound;
    bound.AABBmin = glm::vec3(FLT_MAX);
    bound.AABBmax = glm::vec3(-FLT_MAX);

    for (int i = start; i < end; i++) {
        AABB tmp_bound = calculateAABB(triangles[i]);
        bound.AABBmin = glm::min(bound.AABBmin, tmp_bound.AABBmin);
        bound.AABBmax = glm::max(bound.AABBmax, tmp_bound.AABBmax);
    }

    node.bound = bound;

    int numTri = end - start;

    if (numTri <= 1) {
        //leaf
        node.isLeaf = true;
        node.left = -1;
        node.right= -1;
        nodes.push_back(node);
        return nodes.size() - 1;
    }
    else {
        node.isLeaf = false;
        glm::vec3 extent = bound.AABBmax - bound.AABBmin;
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z) {
            axis = 1;
        }
        else if (extent.z > extent.x && extent.z > extent.y) {
            axis = 2;
        }
        

        //Comparator Function (Lambda): [axis](const Triangle& a, const Triangle& b) {
        //float aCentroid = (a.v0[axis] + a.v1[axis] + a.v2[axis]) / 3.0f;
        //float bCentroid = (b.v0[axis] + b.v1[axis] + b.v2[axis]) / 3.0f;
        //return aCentroid < bCentroid;
        //}
        std::sort(triangles.begin() + start, triangles.begin() + end, [axis](const Triangle& a, const Triangle& b) {
            float aCentroid = (a.v0[axis] + a.v1[axis] + a.v2[axis]) / 3.0f;
            float bCentroid = (b.v0[axis] + b.v1[axis] + b.v2[axis]) / 3.0f;
            return aCentroid < bCentroid;
            });

        int mid = start + numTri / 2;

        int leftChild = buildBVH(nodes, triangles, start, mid);
        int rightChild = buildBVH(nodes, triangles, mid, end);

        //post-order
        node.left = leftChild;
        node.right = rightChild;
        nodes.push_back(node);

        return nodes.size() - 1;
    }
}


bool Scene::loadTexture(const std::string& filename, TextureData& textureData) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!data) {
        std::cerr << "Failed to load texture image: " << filename << "\n";
        std::cerr << "stb_image error: " << stbi_failure_reason() << "\n";
        return false;
    }

    textureData.width = width;
    textureData.height = height;
    textureData.channels = STBI_rgb_alpha;

    size_t dataSize = width * height * 4 * sizeof(unsigned char);
    textureData.h_data = new unsigned char[dataSize];
    memcpy(textureData.h_data, data, dataSize);

    stbi_image_free(data);

    return true;
}

//specilized for loading floating point hdr
bool Scene::loadTexture_hdr(const std::string& filename, EnvData_hdr& EnvData_hdr) {
    int width, height, channels;

    float* data = stbi_loadf(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!data) {
        std::cerr << "Failed to load HDR texture image: " << filename << "\n";
        std::cerr << "stb_image error: " << stbi_failure_reason() << "\n";
        return false;
    }
    
    EnvData_hdr.width = width;
    EnvData_hdr.height = height;
    EnvData_hdr.channels = STBI_rgb_alpha;  
    
    size_t numElements = width * height * 4;  
    size_t dataSize = numElements * sizeof(float); 

    EnvData_hdr.h_data = new float[numElements];  
    memcpy(EnvData_hdr.h_data, data, dataSize);
    stbi_image_free(data);

    return true;

}