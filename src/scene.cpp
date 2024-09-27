#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"

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
        }
		else if (p["TYPE"] == "Refractive")
		{
			const auto& col = p["RGB"];
            const auto& specCol = p["SPECRGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
			newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.hasRefractive = 1.0f;
		}
        else if (p["TYPE"] == "Glass")
		{
			const auto& col = p["RGB"];
            const auto& specCol = p["SPECRGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
			newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasRefractive = 1.0f;
            newMaterial.hasReflective = 1.0f;
		}
        else if (p["TYPE"] == "Reflective")
        {
            const auto& col = p["RGB"];
			//newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.specular.exponent = p["EXPONENT"];
            newMaterial.hasReflective = 1.0f;
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
        else if (type == "mesh")
        {
            newGeom.type = MESH;
            //Add for mesh
#if 0
            loadFromOBJ(p["OBJ"], newGeom);
            cout << "Loaded mesh from " << p["OBJ"] << endl;
#endif
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
            cout << "Geom push back " << endl;
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

void Scene::loadFromOBJ(const std::string& filename, Geom& newGeom) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

    if (!warn.empty()) {
        std::cout << "WARNING: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
    }
    if (!ret) {
        exit(1);
    }
    newGeom.numTriangles = 0;
    for (const auto& shape : shapes) {
        int numTrianglesInShape = shape.mesh.indices.size() / 3;
        std::cout << "Triangles in shape: " << numTrianglesInShape << std::endl;
        try {
            newGeom.numTriangles += numTrianglesInShape;
        }
        catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
            exit(1);
        }
       // newGeom.numTriangles += numTrianglesInShape;
       // newGeom.numTriangles += shape.mesh.indices.size() / 3;
    }
    try {
        newGeom.triangles = new Triangle[newGeom.numTriangles];
        std::cout << "Successfully allocated memory for " << newGeom.numTriangles << " triangles" << std::endl;
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        exit(1);
    }

    int triangleIndex = 0;
    for (const auto& shape : shapes) {
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            int idx0 = shape.mesh.indices[i].vertex_index;
            int idx1 = shape.mesh.indices[i + 1].vertex_index;
            int idx2 = shape.mesh.indices[i + 2].vertex_index;

            glm::vec3 v0(attrib.vertices[3 * idx0], attrib.vertices[3 * idx0 + 1], attrib.vertices[3 * idx0 + 2]);
            glm::vec3 v1(attrib.vertices[3 * idx1], attrib.vertices[3 * idx1 + 1], attrib.vertices[3 * idx1 + 2]);
            glm::vec3 v2(attrib.vertices[3 * idx2], attrib.vertices[3 * idx2 + 1], attrib.vertices[3 * idx2 + 2]);

            int triangleIndex = i / 3;
            if (triangleIndex >= newGeom.numTriangles) {
                std::cerr << "Error: triangleIndex out of bounds: " << triangleIndex << " >= " << newGeom.numTriangles << std::endl;
                exit(1);
            }
            newGeom.triangles[triangleIndex] = { v0, v1, v2 };
           // std::cout << "Triangles loaded: " << triangleIndex << std::endl;
            triangleIndex++;
        }
    }
    std::cout << "Vertices: " << attrib.vertices.size() / 3 << std::endl;
    std::cout << "Triangles number: " << newGeom.numTriangles << std::endl;

    //// Now convert OBJ data to Geom structure
    //for (const auto& shape : shapes) {
    //    Geom meshGeom;
    //    meshGeom.type = MESH;
    //    meshGeom.numTriangles = shape.mesh.indices.size() / 3;
    //    meshGeom.triangles = new Triangle[meshGeom.numTriangles];

    //    for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
    //        int idx0 = shape.mesh.indices[i].vertex_index;
    //        int idx1 = shape.mesh.indices[i + 1].vertex_index;
    //        int idx2 = shape.mesh.indices[i + 2].vertex_index;

    //        glm::vec3 v0(attrib.vertices[3 * idx0], attrib.vertices[3 * idx0 + 1], attrib.vertices[3 * idx0 + 2]);
    //        glm::vec3 v1(attrib.vertices[3 * idx1], attrib.vertices[3 * idx1 + 1], attrib.vertices[3 * idx1 + 2]);
    //        glm::vec3 v2(attrib.vertices[3 * idx2], attrib.vertices[3 * idx2 + 1], attrib.vertices[3 * idx2 + 2]);

    //        meshGeom.triangles[i / 3] = { v0, v1, v2 };
    //    }

    //  //  geoms.push_back(meshGeom);
    //}

    //for (size_t i = 0; i < attrib.vertices.size() / 3; i++) {
    //    std::cout << "v[" << i << "] = ("
    //        << attrib.vertices[3 * i + 0] << ", "
    //        << attrib.vertices[3 * i + 1] << ", "
    //        << attrib.vertices[3 * i + 2] << ")" << std::endl;
    //}


}
