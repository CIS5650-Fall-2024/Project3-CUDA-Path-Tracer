
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <windows.h>
#include "json.hpp"
#include "scene.h"
#include <cstdlib> 
#include <cstring>
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

void Scene::loadTexture(const std::string& texturePath, Texture* texture) {
    int width, height, nrChannels;
    unsigned char* img = stbi_load(texturePath.c_str(), &width, &height, &nrChannels, 3);
    if (img) {
        texture->width = width;
        texture->height = height;
        glm::vec3* hostData = new glm::vec3[width * height];
        for (int i = 0; i < width * height; i++)
        {
            glm::vec3 textureColor = glm::vec3(img[i * 3], img[i * 3 + 1], img[i * 3 + 2]) / 255.0f;
            hostData[i] = textureColor;
        }
        texture->data = hostData;
    }
    else {
        std::cerr << "Failed to load texture: " << texturePath << std::endl;
    }

   
}

std::string getCurrentPath() {
	char buffer[MAX_PATH];
	DWORD dwRet = GetCurrentDirectory(MAX_PATH, buffer);

	if (dwRet == 0) {
		std::cerr << "Error getting current directory." << std::endl;
	}

	//std::cout << "Current directory: " << buffer << std::endl;
	std::string currentPath(buffer);
	size_t pos = currentPath.find_last_of("\\");

	if (pos != std::string::npos) {
		currentPath = currentPath.substr(0, pos); // Remove the last directory part ('build')
	}

	return currentPath;
}

// write mesh info to Geom
void Scene::writeMeshInfo(tinyobj::attrib_t* attrib, 
    Geom* geom, 
    std::vector<tinyobj::shape_t>& shapes, 
    std::vector<tinyobj::material_t>& materials, 
    std::vector<Material>* scene_materials) 
{

    std::vector<glm::vec3> temp_vertices;
    std::vector<glm::vec3> temp_normals;
    std::vector<glm::vec2> temp_uvs;
    std::vector<int> temp_indices;
	std::vector<Triangle> temp_triangles;

	unsigned int nextTriangleIndex = 0;

	// load materials
    for (unsigned int i = 0; i < materials.size(); i++)
    {
        auto mp = materials[i];
		printf("Material name: %s\n", mp.name.c_str());
		Material mtl_material{};
        if (mp.diffuse_texname.length() > 0) {
			std::cout << "Diffuse texture: " << mp.diffuse_texname << std::endl;
            mtl_material.hasDiffuseTexture = 1;
			mtl_material.diffuseTexture = new Texture();
			std::string workingDir = getCurrentPath();
			std::string texturePath = workingDir + "\\scenes\\objs\\" + mp.diffuse_texname;
            Scene::loadTexture(texturePath, mtl_material.diffuseTexture);
        }
        // normal map
		if (mp.normal_texname.length() > 0) {
			std::cout << "Normal texture: " << mp.normal_texname << std::endl;
            mtl_material.hasNormalTexture = 1;
			mtl_material.normalTexture = new Texture();
			std::string workingDir = getCurrentPath();
			std::string texturePath = workingDir + "\\scenes\\objs\\" + mp.normal_texname;
			Scene::loadTexture(texturePath, mtl_material.normalTexture);
		}
		mtl_material.color = glm::vec3(mp.diffuse[0], mp.diffuse[1], mp.diffuse[2]);
		mtl_material.specular.color = glm::vec3(mp.specular[0], mp.specular[1], mp.specular[2]);
		mtl_material.specular.exponent = 1.0 / mp.shininess * mp.shininess;
		mtl_material.emittance = (mp.emission[0] + mp.emission[1] + mp.emission[2]) / 3.f;
		mtl_material.hasReflective = mp.illum == 2 ? 1 : 0;
		mtl_material.hasRefractive = mp.illum == 3 ? 1 : 0;
		mtl_material.indexOfRefraction = mp.ior;

		// add material to scene materials
		Scene::materials.push_back(mtl_material);
		geom->materialid = Scene::materials.size() - 1;
        

    }

    for (unsigned int i = 0; i < shapes.size(); i++)
    {
        std::vector<float>& positions = attrib->vertices;
        std::vector<float>& normals = attrib->normals;
        std::vector<float>& uvs = attrib->texcoords;
		std::vector<tinyobj::index_t> indices = shapes[i].mesh.indices;
        for (unsigned int j = 0; j < shapes[i].mesh.indices.size(); j += 3) {
            // triangulate
            glm::vec3 p1(positions[indices[j].vertex_index * 3], positions[indices[j].vertex_index * 3 + 1], positions[indices[j].vertex_index * 3 + 2]);
            glm::vec3 p2(positions[indices[j + 1].vertex_index * 3], positions[indices[j + 1].vertex_index * 3 + 1], positions[indices[j + 1].vertex_index * 3 + 2]);
            glm::vec3 p3(positions[indices[j + 2].vertex_index * 3], positions[indices[j + 2].vertex_index * 3 + 1], positions[indices[j + 2].vertex_index * 3 + 2]);

            Triangle t = Triangle(p1, p2, p3, nextTriangleIndex++);

			if (normals.size() > 0)
			{
				glm::vec3 n1(normals[indices[j].normal_index * 3], normals[indices[j].normal_index * 3 + 1], normals[indices[j].normal_index * 3 + 2]);
				glm::vec3 n2(normals[indices[j + 1].normal_index * 3], normals[indices[j + 1].normal_index * 3 + 1], normals[indices[j + 1].normal_index * 3 + 2]);
				glm::vec3 n3(normals[indices[j + 2].normal_index * 3], normals[indices[j + 2].normal_index * 3 + 1], normals[indices[j + 2].normal_index * 3 + 2]);
				t.normals[0] = n1;
				t.normals[1] = n2;
				t.normals[2] = n3;
			}

			if (uvs.size() > 0)
			{
				glm::vec2 uv1(uvs[indices[j].texcoord_index * 2], uvs[indices[j].texcoord_index * 2 + 1]);
				glm::vec2 uv2(uvs[indices[j + 1].texcoord_index * 2], uvs[indices[j + 1].texcoord_index * 2 + 1]);
				glm::vec2 uv3(uvs[indices[j + 2].texcoord_index * 2], uvs[indices[j + 2].texcoord_index * 2 + 1]);
				t.uvs[0] = uv1;
				t.uvs[1] = uv2;
				t.uvs[2] = uv3;
			}


            temp_triangles.push_back(t);
			temp_vertices.push_back(p1);
			temp_vertices.push_back(p2);
			temp_vertices.push_back(p3);
			if (normals.size() > 0)
			{
				temp_normals.push_back(t.normals[0]);
				temp_normals.push_back(t.normals[1]);
				temp_normals.push_back(t.normals[2]);
			}
			if (uvs.size() > 0)
			{
				temp_uvs.push_back(t.uvs[0]);
				temp_uvs.push_back(t.uvs[1]);
				temp_uvs.push_back(t.uvs[2]);
			}
			temp_indices.push_back(temp_indices.size());
			temp_indices.push_back(temp_indices.size());
			temp_indices.push_back(temp_indices.size());

        }
        
    }

    geom->mesh->vertices = new glm::vec3[temp_vertices.size()];
    std::copy(temp_vertices.begin(), temp_vertices.end(), geom->mesh->vertices);

    geom->mesh->normals = new glm::vec3[temp_normals.size()];
    std::copy(temp_normals.begin(), temp_normals.end(), geom->mesh->normals);

    geom->mesh->uvs = new glm::vec2[temp_uvs.size()]; 
    std::copy(temp_uvs.begin(), temp_uvs.end(), geom->mesh->uvs);

    geom->mesh->indices = new int[temp_indices.size()];
    std::copy(temp_indices.begin(), temp_indices.end(), geom->mesh->indices);

    geom->mesh->triangles = new Triangle[temp_triangles.size()];
    std::copy(temp_triangles.begin(), temp_triangles.end(), geom->mesh->triangles);


    // Update sizes in the structure
    geom->mesh->num_vertices = temp_vertices.size();
    geom->mesh->num_normals = temp_normals.size();
    geom->mesh->num_uvs = temp_uvs.size();
    geom->mesh->num_indices = temp_indices.size();
	geom->mesh->num_triangles = temp_triangles.size();
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
#if 1
            // texture
            if (p.find("DIFFUSE_TEXTURE") != p.end() && !p["DIFFUSE_TEXTURE"].is_null())
            {
                newMaterial.diffuseTexture = new Texture();
				std::string workingDir = getCurrentPath();
				std::string texturePath = workingDir + "\\scenes\\" + p["DIFFUSE_TEXTURE"].get<std::string>();
				Scene::loadTexture(texturePath, newMaterial.diffuseTexture);
                newMaterial.hasDiffuseTexture = true;
                std::cout << "Diffuse texture loaded: " << p["DIFFUSE_TEXTURE"] << std::endl;
            }
            else
            {
                newMaterial.hasDiffuseTexture = false;
            }

            // Load normal texture if available
            if (p.find("NORMAL_TEXTURE") != p.end() && !p["NORMAL_TEXTURE"].is_null())
            {
                newMaterial.normalTexture = new Texture();
				std::string workingDir = getCurrentPath();
				std::string texturePath = workingDir + "\\scenes\\" + p["NORMAL_TEXTURE"].get<std::string>();
                Scene::loadTexture(texturePath, newMaterial.normalTexture);
                newMaterial.hasNormalTexture = true;
                std::cout << "Normal texture loaded: " << p["NORMAL_TEXTURE"] << std::endl;
            }
            else
            {
                newMaterial.hasNormalTexture = false;
            }
#endif           
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
            newMaterial.hasReflective = 1;
			newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
			float roughness = p["ROUGHNESS"];
			newMaterial.specular.exponent = 16;
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
		else if (p["TYPE"] == "Reflective")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            const auto& spec_col = p["SPECULAR_COLOR"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);

			newMaterial.hasReflective = 1;
		}
		else if (p["TYPE"] == "Refractive")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasRefractive = 1;
			newMaterial.indexOfRefraction = p["IOR"];
			printf("IOR: %f\n", newMaterial.indexOfRefraction);
		}

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
	const auto& hdrData = data["Skybox"];
    if (!hdrData.is_null())
    {
		skyboxTexture = new Texture();
		std::string workingDir = getCurrentPath();
		std::string texturePath = workingDir + "\\scenes\\" + hdrData["SKYBOX_TEXTURE"].get<std::string>();
		Scene::loadTexture(texturePath, skyboxTexture);
        skyboxTexture->type = TEXTURE_2D;
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
			std::string meshFilePath = p["FILE"];
            char buffer[MAX_PATH];
            DWORD dwRet = GetCurrentDirectory(MAX_PATH, buffer);

            if (dwRet == 0) {
                std::cerr << "Error getting current directory." << std::endl;
            }

            std::cout << "Current directory: " << buffer << std::endl;
            std::string currentPath(buffer);
            size_t pos = currentPath.find_last_of("\\");
            if (pos != std::string::npos) {
                currentPath = currentPath.substr(0, pos); // Remove the last directory part ('build')
            }
            currentPath += "\\scenes";
			std::string mtl_path = currentPath + "\\objs";
            meshFilePath = currentPath + "\\" + meshFilePath;
            //test mesh file path
			std::cout << meshFilePath << std::endl;
            // load obj file
			newGeom.mesh = new Mesh();
            std::vector<tinyobj::shape_t> shapes; 
            std::vector<tinyobj::material_t> materials_obj;
            tinyobj::attrib_t attrib;
            std::string obj_errors;
			std::string obj_warnings;
            bool errors = tinyobj::LoadObj(&attrib, 
                &shapes, 
                &materials_obj, 
                &obj_warnings, 
                &obj_errors, 
                meshFilePath.c_str(), mtl_path.c_str());
            if (!obj_warnings.empty()) {
                std::cout << "WARN: " << obj_warnings << std::endl;
            }
            if (!obj_errors.empty()) {
                std::cerr << obj_errors << std::endl;
            }
            if (obj_warnings.empty() && obj_errors.empty()) {
				writeMeshInfo(&attrib, 
                    &newGeom, 
                    shapes, 
                    materials_obj, 
                    &materials);
			}
            else {
                std::cout << "Error loading mesh: " << errors << std::endl;
            }
			this->num_meshes++;

		}
        if (p.find("MATERIAL") != p.end() && !p["MATERIAL"].is_null())
        {
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
