#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename) : envMap(nullptr), bvh(nullptr)
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

Scene::~Scene()
{
	delete envMap;
	envMap = nullptr;
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
        Material newMaterial;

		// look at each property of material, check if it exist in the json, if not, use default value
		if (p.contains("RGB"))
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
		}
		// load parameters following the order of the struct:
		//::begin parameters
		/*	color baseColor .82 .67 .16
			float metallic 0 1 0
			float subsurface 0 1 0
			float specular 0 1 .5
			float roughness 0 1 .5
			float specularTint 0 1 0
			float anisotropic 0 1 0
			float sheen 0 1 0
			float sheenTint 0 1 .5
			float clearcoat 0 1 0
			float clearcoatGloss 0 1 1
			::end parameters */
		if (p.contains("METALLIC")) newMaterial.metallic = p["METALLIC"];
		if (p.contains("SUBSURFACE")) newMaterial.subsurface = p["SUBSURFACE"];
		if (p.contains("SPECULAR")) newMaterial.specular = p["SPECULAR"];
		if (p.contains("ROUGHNESS")) newMaterial.roughness = p["ROUGHNESS"];
		if (p.contains("SPECULARTINT")) newMaterial.specularTint = p["SPECULARTINT"];
		if (p.contains("ANISOTROPIC")) newMaterial.anisotropic = p["ANISOTROPIC"];
		if (p.contains("SHEEN")) newMaterial.sheen = p["SHEEN"];
		if (p.contains("SHEENTINT")) newMaterial.sheenTint = p["SHEENTINT"];
		if (p.contains("CLEARCOAT")) newMaterial.clearcoat = p["CLEARCOAT"];
		if (p.contains("CLEARCOATGLOSS")) newMaterial.clearcoatGloss = p["CLEARCOATGLOSS"];
		if (p.contains("IOR")) newMaterial.ior = p["IOR"];
		if (p.contains("TYPE"))
		{
			std::string matType(p["TYPE"]);
			if (matType == "Diffuse") newMaterial.type = MaterialType::DIFFUSE;
			else if (matType == "Transmit") newMaterial.type = MaterialType::TRANSMIT;
			else newMaterial.type = MaterialType::MICROFACET;
		}
		else newMaterial.type = MaterialType::DIFFUSE;

        // print material info
		printf("Material %s\n", name.c_str());
		printf("Color: %s\n", glm::to_string(newMaterial.color).c_str());
		printf("Reflective: %f\n", newMaterial.reflective);
		printf("Refractive: %f\n", newMaterial.refractive);
		printf("Emittance: %f\n", newMaterial.emittance);
		printf("Roughness: %f\n", newMaterial.roughness);
		printf("Metallic: %f\n", newMaterial.metallic);
		printf("Sheen: %f\n", newMaterial.sheen);
		printf("Clearcoat: %f\n", newMaterial.clearcoat);
		printf("Anisotropic: %f\n", newMaterial.anisotropic);
		printf("Specular Tint: %f\n", newMaterial.specularTint);
		printf("Subsurface: %f\n", newMaterial.subsurface);
		printf("Type: %d\n", newMaterial.type);
		printf("\n");

        MatNameToID[name] = materials.size();
        addMaterial(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
        glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        glm::vec3 scaling = glm::vec3(scale[0], scale[1], scale[2]);
        if (type == "cube")
        {
            createCube(MatNameToID[p["MATERIAL"]], translation, rotation, scaling);

        }
		else if (type == "sphere")
        {
			createSphere(MatNameToID[p["MATERIAL"]], translation, rotation, scaling);
		}
		else if (type == "mesh")
		{
			const auto& filename = p["FILENAME"];
			loadObj(filename, MatNameToID[p["MATERIAL"]], translation, rotation, scaling);
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

	// set environment 
	if (data.contains("Environment"))
	{
		const auto& env = data["Environment"];
		this->envMapPath = env["FILENAME"];
	}

	// set up lights
	if (data.contains("Lights"))
	{
		const auto& datas = data["Lights"];
		for (const auto& p : datas)
		{
			Geom newLight;
			Light light;

			const auto& trans = p["TRANS"];
			const auto& rotat = p["ROTAT"];
			const auto& scale = p["SCALE"];
			glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
			glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
			glm::vec3 scaling = glm::vec3(scale[0], scale[1], scale[2]);
			updateTransform(newLight, translation, rotation, scaling);

			std::string type(p["TYPE"]);
			Material mat;
			mat.isLight = true;
			if (p.contains("MATERIAL"))
			{
				auto col = p["MATERIAL"]["RGB"];
				mat.color = glm::vec3(col[0], col[1], col[2]);
				mat.emittance = p["MATERIAL"]["EMITTANCE"];
				mat.roughness = p["MATERIAL"]["ROUGHNESS"];
			}
			else
			{
				mat.color = glm::vec3(1.0f);
				mat.emittance = 1.0f;
			}
			newLight.lightid = lights.size();
			addMaterial(mat);
			
			newLight.materialid = mat.materialId;
			newLight.triangleStartIdx = newLight.triangleEndIdx = -1;
			if (type == "Area")
			{
				newLight.triangleStartIdx = triangles.size();
				newLight.triangleEndIdx = triangles.size() + 2;

				// create a square and add to triangles
				Triangle tri1;
				tri1.vertices[0] = glm::vec3(-1, 1, 0);
				tri1.vertices[1] = glm::vec3(-1, -1, 0);
				tri1.vertices[2] = glm::vec3(1, -1, 0);
				tri1.normals[0] = glm::vec3(0, 0, 1);
				tri1.normals[1] = glm::vec3(0, 0, 1);
				tri1.normals[2] = glm::vec3(0, 0, 1);
				tri1.hasNormals = false;

				Triangle tri2;
				tri2.vertices[0] = glm::vec3(-1, 1, 0);
				tri2.vertices[1] = glm::vec3(1, -1, 0);
				tri2.vertices[2] = glm::vec3(1, 1, 0);
				tri2.normals[0] = glm::vec3(0, 0, 1);
				tri2.normals[1] = glm::vec3(0, 0, 1);
				tri2.normals[2] = glm::vec3(0, 0, 1);
				tri2.hasNormals = false;

				triangles.push_back(std::move(tri1));
				triangles.push_back(std::move(tri2));

				updateTriangleTransform(newLight, triangles);

				light.lightType = AREALIGHT;
			}
			else if (type == "Point")
			{
				light.lightType = POINTLIGHT;
			}
			else if (type == "Spot")
			{
				light.lightType = SPOTLIGHT;
			}

			light.transform = newLight.transform;
			light.inverseTransform = newLight.inverseTransform;
			light.emission = mat.color * mat.emittance;
			light.area = (float)scale[0] * (float)scale[1];
			// print light info
			printf("Light %s\n", type.c_str());
			// print light transform
			printf("light transform: %s\n", glm::to_string(light.transform).c_str());

			printf("Emission: %s\n", glm::to_string(light.emission).c_str());
			printf("\n");
			lights.push_back(std::move(light));
		}
	}
}

void Scene::createCube(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
    printf("create cube\n");

	loadObj("D:/Fall2024/CIS5650/Project3-CUDA-Path-Tracer/scenes/objs/cube.obj", materialid, translation, rotation, scale);
}

void Scene::createSphere(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale, int latitudeSegments, int longitudeSegments)
{
    printf("create sphere\n");
	loadObj("D:/Fall2024/CIS5650/Project3-CUDA-Path-Tracer/scenes/objs/sphere.obj", materialid, translation, rotation, scale);
}


void Scene::loadObj(const std::string& filename, uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
	
	printf("load obj\n");
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> meshMaterials;
    //print material info

	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &meshMaterials, &warn, &err, filename.c_str()))
	{
		throw std::runtime_error(warn + err);
	}
    printf("material size: %d\n", meshMaterials.size());
	for (const auto& shape : shapes)
	{
		Geom newMesh;
		newMesh.type = MESH;
		newMesh.materialid = materialid;
        Scene::updateTransform(newMesh, translation, rotation, scale);
		

        if (shape.mesh.num_face_vertices[0] != 3)
        {
            throw std::runtime_error("Only triangles are supported");
        }

		newMesh.triangleStartIdx = triangles.size();
        newMesh.triangleEndIdx = triangles.size();
        // assume only triangles
		for (size_t f = 0; f < shape.mesh.indices.size(); f += 3)
		{
			Triangle tri;
			for (size_t v = 0; v < 3; v++)
			{
				const auto& idx = shape.mesh.indices[f + v];
				tri.vertices[v] = glm::vec3(
					attrib.vertices[3 * idx.vertex_index + 0],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]
				);
				if (attrib.normals.size() > 0)
				{
					tri.normals[v] = glm::vec3(
						attrib.normals[3 * idx.normal_index + 0],
						attrib.normals[3 * idx.normal_index + 1],
						attrib.normals[3 * idx.normal_index + 2]
					);
				}
				if (attrib.texcoords.size() > 0)
				{
					tri.uvs[v] = glm::vec2(
						attrib.texcoords[2 * idx.texcoord_index + 0],
						attrib.texcoords[2 * idx.texcoord_index + 1]
					);
				}
				tri.hasNormals = attrib.normals.size() > 0;

			}
            triangles.push_back(std::move(tri));
			newMesh.triangleEndIdx++;
		}
		printf("Loaded %s with %d triangles\n", filename.c_str(), newMesh.triangleEndIdx - newMesh.triangleStartIdx);

		updateTriangleTransform(newMesh, triangles);
		geoms.push_back(newMesh);
	}
}

void Scene::addMaterial(Material& m)
{
	m.materialId = materials.size();
	materials.push_back(m);
}

void Scene::loadEnvMap() {
	if (envMapPath == "")
	{
		printf("No environment map specified\n");
		return;
	}
	loadEnvMap(envMapPath.c_str());
}

void Scene::loadEnvMap(const char* filename)
{
    if (envMap)
        delete envMap;
    envMap = nullptr;
	envMap = new Texture(filename);
	printf("Loaded environment map %s\n", filename);
    //printf("cuda texture object created: %d\n", envMap->texObj);
}

void Scene::updateTransform(Geom& geom, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
    geom.translation = translation;
    geom.rotation = rotation;
    geom.scale = scale;
    geom.transform = utilityCore::buildTransformationMatrix(
        geom.translation, geom.rotation, geom.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
}

void Scene::updateTriangleTransform(const Geom& geom, std::vector<Triangle>& triangles)
{
    for (int i = geom.triangleStartIdx; i < geom.triangleEndIdx; ++i)
    {
        auto& tri = triangles[i];
        for (int j = 0; j < 3; ++j)
        {
            tri.vertices[j] = glm::vec3(geom.transform * glm::vec4(tri.vertices[j], 1.0f));
			
            tri.normals[j] = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(tri.normals[j], 0.0f)));
        }
		tri.materialid = geom.materialid;
		tri.lightid = geom.lightid;
    }
}


void Scene::createBVH()
{
    if (bvh != nullptr)
    {
        delete bvh;
    }
    bvh = new BVHAccel(this->triangles, this->triangles.size(), 2);
	bvh->build(this->triangles, this->triangles.size());
    printf("BVH created\n");
}

BVHAccel::LinearBVHNode* Scene::getLBVHRoot()
{
	if (bvh == nullptr)
	{
		printf("BVH not created\n");
		return nullptr;
	}
	return bvh->nodes;
}