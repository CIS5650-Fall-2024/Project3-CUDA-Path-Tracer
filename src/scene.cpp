#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "stb_image.h"

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
    size_t lastSlashPos = jsonName.find_last_of('/\\');
    std::string basePath = jsonName.substr(0, lastSlashPos);
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.diffuse_textureId = -1;
        newMaterial.normal_textureId = -1;
        if (p.find("D_PATH") != p.end())
        {
            newMaterial.diffuse_textureId = textures.size();
            std::string textureName = p["D_PATH"];
            std::string texturePath = basePath + textureName;
            loadFromTexture(texturePath);
        }
        if (p.find("N_PATH") != p.end())
        {
            newMaterial.normal_textureId = textures.size();
            std::string textureName = p["N_PATH"];
            std::string texturePath = basePath + textureName;
            loadFromTexture(texturePath);
        }
        if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            newMaterial.hasRefractive = 1.0f;

            newMaterial.indexOfRefraction = p["IOR"];
        }
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
            newMaterial.hasReflective = true;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;

        if (type == "model") newGeom.type = MESH;
        else if (type == "cube") newGeom.type = CUBE;
        else newGeom.type = SPHERE;

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

        if (newGeom.type == MESH)
        {
            std::string objName = p["PATH"];
            std::string objPath = basePath + objName;
            loadFromOBJ(objPath, newGeom.materialid);
            continue;
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

void Scene::loadFromOBJ(const std::string& objName, int materialID) {
    std::string inputfile = objName;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {

                // Define new Vertex
                Vertex newVert;

                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                // Set vertex position
                newVert.pos = glm::vec3(vx, vy, vz);

                // Set vertex materialID
                newVert.materialid = materialID;

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    // Set vertex position
                    newVert.nor = glm::vec3(nx, ny, nz);
                }

                //// Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                    newVert.uv = glm::vec2(tx, ty);
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];

                // push_back new vertex to buffer
                this->vertices.push_back(newVert);
            }

            index_offset += fv;

            // per-face material
            //shapes[s].mesh.material_ids[f];
        }
    }

    for (int v_index = 0; v_index < this->vertices.size(); v_index += 3)
    {
        // Calculate tangents for the triangle
        Vertex& v1 = vertices[v_index];
        Vertex& v2 = vertices[v_index + 1];
        Vertex& v3 = vertices[v_index + 2];

        glm::vec3 edge1 = v2.pos - v1.pos;
        glm::vec3 edge2 = v3.pos - v1.pos;

        glm::vec2 deltaUV1 = v2.uv - v1.uv;
        glm::vec2 deltaUV2 = v3.uv - v1.uv;

        float denom = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

        if (denom == 0) continue;

        denom = 1.0f / denom;

        glm::vec3 tangent;
        tangent = denom * (deltaUV2.y * edge1 - deltaUV1.y * edge2);

        // correct the vertex tangent based on vertex normal
        v1.tangent = tangent;
        v1.tangent -= v1.nor * glm::dot(
            v1.tangent, v1.nor);
        v1.tangent = glm::normalize(v1.tangent);
        
        v2.tangent = tangent;
        v2.tangent -= v2.nor * glm::dot(
            v2.tangent, v2.nor);
        v2.tangent = glm::normalize(v2.tangent);

        v3.tangent = tangent;
        v3.tangent -= v3.nor * glm::dot(
            v3.tangent, v3.nor);
        v3.tangent = glm::normalize(v3.tangent);
    }
}

void Scene::loadFromTexture(const std::string& textureName) {
    int width, height;
    float* data = stbi_loadf(textureName.c_str(), &width, &height, NULL, 4);

    if (!data) {
        std::cerr << "Failed to load texture" << std::endl;
        exit(1);
    }

    Texture newTexture;
    newTexture.width = width;
    newTexture.height = height;
    newTexture.startIdx = (int)this->texturePixels.size();
    this->textures.push_back(newTexture);
    
    for (int i = 0; i < width * height; i++)
    {
        glm::vec4 newPixel;
        for (int j = 0; j < 4; j++)
        {
            newPixel[j] = data[i * 4 + j];
        }
        this->texturePixels.push_back(newPixel);
    }

    stbi_image_free(data);
}