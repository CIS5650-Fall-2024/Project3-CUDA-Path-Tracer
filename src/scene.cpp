#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "stb_image.h"

// same file from CIS560
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

void Scene::parseObjFileToVertices(const std::string& filepath, Geom& geom) {
	tinyobj::ObjReader reader;
	int current_offset = vertices.size();
	geom.vertex_offset = current_offset;
	geom.vertex_count = 0;

	if (!reader.ParseFromFile(filepath)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	const auto& attrib = reader.GetAttrib();
	const auto& shapes = reader.GetShapes();

	glm::vec3 min_box(FLT_MAX), max_box(-FLT_MAX);

	for (const auto& shape : shapes) {
		size_t index_offset = 0;

		for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); face++) {
			int fv = shape.mesh.num_face_vertices[face];

			std::vector<tinyobj::index_t> face_indices;
			for (int v = 0; v < fv; v++) {
				face_indices.push_back(shape.mesh.indices[index_offset + v]);
			}

			std::vector<int> remaining_indices(fv);
			std::iota(remaining_indices.begin(), remaining_indices.end(), 0);

			while (remaining_indices.size() > 2) {
				int i0 = remaining_indices[0];
				int i1 = remaining_indices[1];
				int i2 = remaining_indices[2];

				for (int v : {i0, i1, i2}) {
					tinyobj::index_t idx = face_indices[v];
					Vertex vert{};

					vert.pos.x = attrib.vertices[3 * idx.vertex_index + 0];
					vert.pos.y = attrib.vertices[3 * idx.vertex_index + 1];
					vert.pos.z = attrib.vertices[3 * idx.vertex_index + 2];

					min_box = glm::min(min_box, vert.pos);
					max_box = glm::max(max_box, vert.pos);

					if (idx.normal_index >= 0) {
						vert.norm.x = attrib.normals[3 * idx.normal_index + 0];
						vert.norm.y = attrib.normals[3 * idx.normal_index + 1];
						vert.norm.z = attrib.normals[3 * idx.normal_index + 2];
					}

					if (idx.texcoord_index >= 0) {
						vert.uv.x = attrib.texcoords[2 * idx.texcoord_index + 0];
						vert.uv.y = attrib.texcoords[2 * idx.texcoord_index + 1];
					}

					vertices.push_back(vert);
				}

				remaining_indices.erase(remaining_indices.begin() + 1);
			}

			index_offset += fv;
		}
	}

	geom.vertex_offset = current_offset;
	geom.vertex_count = vertices.size() - current_offset;
	geom.mesh_aabb_min = min_box;
	geom.mesh_aabb_max = max_box;
	cout << filepath << " loaded" << endl;
}

void Scene::parseImgFileToTextures(const std::string& filepath, Geom& geom, std::vector<Texture>& global_textures) {
	int current_offset = global_textures.size();
	geom.texture_offset = current_offset;
	geom.texture_count = 0;

	int width, height, channels;
	unsigned char* image = stbi_load(filepath.c_str(), &width, &height, &channels, STBI_rgb);

	if (!image) {
		std::cerr << "Failed to load image: " << filepath << std::endl;
		return;
	}


	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 3;

			Texture tex;
			tex.height = height;
			tex.width = width;
			tex.color = glm::vec3(
				image[index] / 255.0f,
				image[index + 1] / 255.0f,
				image[index + 2] / 255.0f
			);

			global_textures.push_back(tex);
		}
	}

	geom.texture_offset = current_offset;
	geom.texture_count = global_textures.size() - current_offset;

	stbi_image_free(image);
	std::cout << "Image loaded and parsed: " << filepath << std::endl;
}

void Scene::parseNormImgFileToTextures(const std::string& filepath, Geom& geom, std::vector<Texture>& global_norm_textures) {
	int current_offset = global_norm_textures.size();
	geom.norm_texture_offset = current_offset;
	geom.norm_texture_count = 0;

	int width, height, channels;
	unsigned char* image = stbi_load(filepath.c_str(), &width, &height, &channels, STBI_rgb);

	if (!image) {
		std::cerr << "Failed to load image: " << filepath << std::endl;
		return;
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 3;

			Texture tex;
			tex.height = height;
			tex.width = width;
			tex.color = glm::vec3(
				image[index] / 255.0f,
				image[index + 1] / 255.0f,
				image[index + 2] / 255.0f
			);

			global_norm_textures.push_back(tex);
		}
	}

	geom.norm_texture_offset = current_offset;
	geom.norm_texture_count = global_norm_textures.size() - current_offset;

	stbi_image_free(image);
	std::cout << "Normal map loaded and parsed: " << filepath << std::endl;
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
			const float& roughness = p["ROUGHNESS"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1.0f - roughness;
		}
		else if (p["TYPE"] == "Transparent")
		{
			const auto& col = p["RGB"];
			const float& roughness = p["ROUGHNESS"];
			const float& transparency = p["TRANSPARENCY"];
			const float& index = p["INDEX"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1.0f - roughness;
			newMaterial.hasRefractive = transparency;
			newMaterial.indexOfRefraction = index;
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

		//// load the mesh data if the geometry type is MESH
		if (newGeom.type == MESH) {

			std::string filepath = jsonName;
			std::size_t dot_pos = filepath.find_last_of('/');

			std::string objpath = filepath;
			if (dot_pos != std::string::npos) {
				objpath.replace(dot_pos + 1, objpath.length() - (dot_pos + 1),
					std::string(p["NAME"]) + std::string(".obj"));
			}
			parseObjFileToVertices(objpath, newGeom);

			std::string imgpath = filepath;
			if (dot_pos != std::string::npos && !p["TEXTURE"].empty()) {
				imgpath.replace(dot_pos + 1, imgpath.length() - (dot_pos + 1),
					std::string(p["TEXTURE"]));
				parseImgFileToTextures(imgpath, newGeom, textures);
			}

			std::string normpath = filepath;
			if (dot_pos != std::string::npos && !p["NORM"].empty()) {
				normpath.replace(dot_pos + 1, normpath.length() - (dot_pos + 1),
					std::string(p["NORM"]));
				parseNormImgFileToTextures(normpath, newGeom, norm_textures);
			}
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
	// used for DOF
	camera.lensRadius = cameraData["LENSRADIUS"];

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
