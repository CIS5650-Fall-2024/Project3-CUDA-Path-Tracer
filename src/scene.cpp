#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <stb_image.h>
#include "json.hpp"
#include "scene.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace std;

using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        constructBVHTree();
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

/*Reference: https://github.com/tinyobjloader/tinyobjloader/tree/release */
void Scene::loadFromOBJ(const std::string& objName, std::vector<glm::vec3>& verts, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs, std::vector<std::string>& matNames, std::unordered_map<std::string, uint32_t>& MatNameToID)
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "../scenes/Textures"; // Path to material files
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(objName, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    int id = 0;
    for (auto& mat : materials)
    {
        Material newMaterial{};
        DiffuseMap newDiffuseMap{};
        if (!mat.diffuse_texname.empty())
        {
            newDiffuseMap.index = id++;
            newDiffuseMap.startIdx = this->textures.size();
            std::string path = "../scenes/Textures/" + mat.diffuse_texname;
            float* diffuseTexture = stbi_loadf(path.c_str(), &newDiffuseMap.width, &newDiffuseMap.height, &newDiffuseMap.channel, 0);
            for (int i = 0; i < newDiffuseMap.width * newDiffuseMap.height; ++i) {
                glm::vec3 diffuseColor = glm::vec3(diffuseTexture[newDiffuseMap.channel * i], diffuseTexture[newDiffuseMap.channel * i + 1], diffuseTexture[newDiffuseMap.channel * i + 2]);
                this->textures.emplace_back(diffuseColor);
            }
            newMaterial.diffuseMap = newDiffuseMap;
            newMaterial.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            if (glm::length(newMaterial.specular.color) > EPSILON)
            {
                if (mat.shininess > EPSILON)
                {
                    newMaterial.microfacet.roughness = glm::min(0.8f, 1.f / glm::sqrt(mat.shininess + 1.f));
                    newMaterial.microfacet.isMicrofacet = true;
                }
                else
                {
                    newMaterial.hasReflective = 1.f;
                }

            }
        }
        else {
            newMaterial.color = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
            newMaterial.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            newMaterial.microfacet.isMicrofacet = true;
            newMaterial.microfacet.roughness = 0.5f;
        }
        MatNameToID[mat.name] = this->materials.size();
        this->materials.emplace_back(newMaterial);
    }

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            /* comment: hackery to get rid of an unwanted part in my mesh
            *  not ideal... 
            if (shapes[s].mesh.material_ids[f] == 7) {
                continue;
            }
            */
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                verts.push_back(glm::vec3(vx, vy, vz));

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    normals.push_back(glm::vec3(nx, ny, nz));
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    uvs.push_back(glm::vec2(tx, ty));
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;

            // per-face material
            if (materials.size())
            {
                matNames.push_back(materials[shapes[s].mesh.material_ids[f]].name);
            }
        }
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
            newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 0.f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 1.f;
            newMaterial.indexOfRefraction = 1.55f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Specular_Diffuse")
        {
            const auto& col = p["RGB"];
            const auto& spec_col = p["SPEC_RGB"];
            newMaterial.hasReflective = 1.f;
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
        }
        else if (p["TYPE"] == "Microfacet")
        {
            const auto& col = p["RGB"];
            const auto& spec_col = p["SPEC_RGB"];
            newMaterial.microfacet.isMicrofacet = true;
            newMaterial.microfacet.roughness = p["ROUGHNESS"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
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
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh") {
            //create triangles
            std::vector<glm::vec3> verts;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec2> uvs;
            std::vector<std::string> materialNames;
            std::string filePath = "../scenes/" + std::string(p["NAME"]);
            loadFromOBJ(filePath, verts, normals, uvs, materialNames, MatNameToID);
            int materialID;
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            int f = 0;
            for (int i = 0; i < verts.size() - 2; i+=3)
            {
                Geom geom;
                geom.type = TRIANGLE;
                materialID = materialNames.size() ? MatNameToID[materialNames[f++]] : MatNameToID[p["MATERIAL"]];
                geom.materialid = materialID;
                geom.translation = glm::vec3(trans[0], trans[1], trans[2]);
                geom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                geom.scale = glm::vec3(scale[0], scale[1], scale[2]);
                geom.transform = utilityCore::buildTransformationMatrix(
                  geom.translation, geom.rotation, geom.scale);
                geom.inverseTransform = glm::inverse(geom.transform);
                geom.invTranspose = glm::inverseTranspose(geom.transform);
                geom.triData.verts[0] = verts[i];
                geom.triData.verts[1] = verts[i+1];
                geom.triData.verts[2] = verts[i+2];
                geom.triData.normals[0] = normals[i];
                geom.triData.normals[1] = normals[i + 1];
                geom.triData.normals[2] = normals[i + 2];
                geom.triData.uvs[0] = uvs[i];
                geom.triData.uvs[1] = uvs[i + 1];
                geom.triData.uvs[2] = uvs[i + 2];
                geoms.push_back(geom);
            }
            continue;
        }
        else if (type == "env_map") {
            std::string filePath = "../scenes/Env/" + std::string(p["NAME"]);
            int width, height, channel;
            float* diffuseTexture = stbi_loadf(filePath.c_str(), &width, &height, &channel, 0);
            for (int i = 0; i < width * height; ++i) {
                glm::vec3 diffuseColor = glm::vec3(diffuseTexture[channel * i], diffuseTexture[channel * i + 1], diffuseTexture[channel * i + 2]);
                this->env.emplace_back(diffuseColor);
            }
            env_width = width;
            env_height = height;
            continue;
        }
        else {
            std::cout << "unknown object type" << std::endl;
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
    camera.focalLength = cameraData["FOCALLENGTH"];
    camera.apertureRadius = cameraData["APERTURE"];

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

float getMin(float x, float y, float z)
{
    return fminf(fminf(x, y), fminf(y, z));
}

float getMax(float x, float y, float z)
{
    return fmaxf(fmaxf(x, y), fmaxf(y, z));
}

void getTriangleAABB(Geom& tri, glm::vec3& mins, glm::vec3& maxs)
{
    glm::vec4 vert0 = tri.transform * glm::vec4(tri.triData.verts[0], 1.f);
    glm::vec4 vert1 = tri.transform * glm::vec4(tri.triData.verts[1], 1.f);
    glm::vec4 vert2 = tri.transform * glm::vec4(tri.triData.verts[2], 1.f);
    mins.x = getMin(vert0.x, vert1.x, vert2.x);
    mins.y = getMin(vert0.y, vert1.y, vert2.y);
    mins.z = getMin(vert0.z, vert1.z, vert2.z);
    maxs.x = getMax(vert0.x, vert1.x, vert2.x);
    maxs.y = getMax(vert0.y, vert1.y, vert2.y);
    maxs.z = getMax(vert0.z, vert1.z, vert2.z);
}

void getOtherPrimitiveAABB(Geom& geom, glm::vec3& mins, glm::vec3& maxs)
{
    glm::vec4 corner0 = geom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f);
    glm::vec4 corner1 = geom.transform * glm::vec4(-0.5f, -0.5f, 0.5f, 1.f);
    glm::vec4 corner2 = geom.transform * glm::vec4(-0.5f, 0.5f, -0.5f, 1.f);
    glm::vec4 corner3 = geom.transform * glm::vec4(0.5f, -0.5f, -0.5f, 1.f);
    glm::vec4 corner4 = geom.transform * glm::vec4(-0.5f, 0.5f, 0.5f, 1.f);
    glm::vec4 corner5 = geom.transform * glm::vec4(0.5f, -0.5f, 0.5f, 1.f);
    glm::vec4 corner6 = geom.transform * glm::vec4(0.5f, 0.5f, -0.5f, 1.f);
    glm::vec4 corner7 = geom.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.f);
    mins.x = getMin(getMin(corner0.x, corner1.x, corner2.x), getMin(corner3.x, corner4.x, corner5.x), fmin(corner6.x, corner7.x));
    mins.y = getMin(getMin(corner0.y, corner1.y, corner2.y), getMin(corner3.y, corner4.y, corner5.y), fmin(corner6.y, corner7.y));
    mins.z = getMin(getMin(corner0.z, corner1.z, corner2.z), getMin(corner3.z, corner4.z, corner5.z), fmin(corner6.z, corner7.z));
    maxs.x = getMax(getMax(corner0.x, corner1.x, corner2.x), getMax(corner3.x, corner4.x, corner5.x), fmax(corner6.x, corner7.x));
    maxs.y = getMax(getMax(corner0.y, corner1.y, corner2.y), getMax(corner3.y, corner4.y, corner5.y), fmax(corner6.y, corner7.y));
    maxs.z = getMax(getMax(corner0.z, corner1.z, corner2.z), getMax(corner3.z, corner4.z, corner5.z), fmax(corner6.z, corner7.z));
}

void getPrimitiveAABB(Geom& geom, glm::vec3& mins, glm::vec3& maxs)
{
    if (geom.type == TRIANGLE) {
        getTriangleAABB(geom, mins, maxs);
    }
    else {
        getOtherPrimitiveAABB(geom, mins, maxs);
    }
}

void Scene::expandBounds(int start, int end, glm::vec3& mins, glm::vec3& maxs)
{
    glm::vec3 otherMins;
    glm::vec3 otherMaxs;
    for (int i = start; i < end; ++i) {
        Geom& primitive = geoms[indices[i]];
        //std::cout << " processing: " << indices[i];
        getPrimitiveAABB(primitive, otherMins, otherMaxs);
        //std::cout << " mins: " << glm::to_string(otherMins) << " maxs: " << glm::to_string(otherMaxs) << std::endl;
        mins.x = fminf(mins.x, otherMins.x);
        mins.y = fminf(mins.y, otherMins.y);
        mins.z = fminf(mins.z, otherMins.z);
        maxs.x = fmaxf(maxs.x, otherMaxs.x);
        maxs.y = fmaxf(maxs.y, otherMaxs.y);
        maxs.z = fmaxf(maxs.z, otherMaxs.z);
    }
    //std::cout << "processed mins: " << glm::to_string(mins) << " processed maxs: " << glm::to_string(maxs) << std::endl;
}

glm::vec3 getCentroid(Geom& geom)
{
    glm::vec3 mins;
    glm::vec3 maxs;
    getPrimitiveAABB(geom, mins, maxs);
    return (mins + maxs) * 0.5f;
}

int Scene::getSubtreeSize(int nodeIndex) {
    BVHNode& node = nodes[nodeIndex];
    if (node.isLeaf()) {
        return 1;
    }
    else {
        int leftSubtreeSize = getSubtreeSize(node.leftChild);
        int rightSubtreeSize = getSubtreeSize(node.leftChild + getSubtreeSize(node.leftChild));
        return 1 + leftSubtreeSize + rightSubtreeSize;
    }
}

void Scene::buildBVHTree(int start, int end)
{
    BVHNode& node = nodes[nodes.size() - 1];
    //std::cout << "Current Node: " << nodes.size() - 1 << std::endl;
    expandBounds(start, end, node.mins, node.maxs);

    int numPrimitives = end - start;
    if (numPrimitives <= 2) //stop dividing the tree further down
    {
        node.startIndex = start;
        node.numPrimitives = numPrimitives;
        return;
    }

    // find split axis
    int axis = 0;
    glm::vec3 extent = node.maxs - node.mins;
    if (extent.y > extent.x && extent.y > extent.z)
    {
        axis = 1; //split along y axis
    }
    else if (extent.z > extent.x) {
        axis = 2; //split along z axis
    }
    
    // partition indices array such that everything before mid have
    // centroids less than or equal to the centroid at mid
    int mid = (start + end) / 2;
    std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
        [&](int a, int b) {
            glm::vec3 centroidA = getCentroid(geoms[a]);
            glm::vec3 centroidB = getCentroid(geoms[b]);
            return centroidA[axis] < centroidB[axis];
        });

    node.leftChild = nodes.size();
    BVHNode leftChild = BVHNode();
    nodes.push_back(leftChild);
    buildBVHTree(start, mid);

    node.rightChild = nodes.size();
    BVHNode rightChild = BVHNode();
    nodes.push_back(rightChild);
    buildBVHTree(mid, end);
}

void Scene::constructBVHTree()
{
    indices.resize(geoms.size());
    std::iota(indices.begin(), indices.end(), 0);

    nodes.reserve(geoms.size() * 2 - 1);
    BVHNode root; 
    nodes.push_back(root); 

    buildBVHTree(0, geoms.size());

    /*
    for (int i = 0; i < std::min(50, (int)nodes.size()); ++i) {
        const BVHNode& node = nodes[i];
        std::cout << "BVH Node " << i << "\n";
        std::cout << "  AABB mins: (" << node.mins.x << ", " << node.mins.y << ", " << node.mins.z << ")\n";
        std::cout << "  AABB maxs: (" << node.maxs.x << ", " << node.maxs.y << ", " << node.maxs.z << ")\n";
        std::cout << "  Left Child: " << node.leftChild << "\n";
        std::cout << "  Right Child: " << node.rightChild << "\n";
        std::cout << "  Number of Primitives: " << node.numPrimitives << "\n";
        std::cout << "----------------------------------------\n";
    }
    */
}

