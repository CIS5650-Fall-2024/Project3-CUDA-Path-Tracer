#pragma once

#include <stb_image.h>  
#include <stb_image_write.h>  
#include <tiny_gltf.h> 
#include <tiny_obj_loader.h>
#include <string>
#include "utilities.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "sceneStructs.h"

namespace mesh {
	typedef struct {
		std::string name;

		float ambient[3];
		float diffuse[3];
		float specular[3];
		float transmittance[3];
		float emission[3];
		float shininess;
		float ior;      // index of refraction
		float dissolve; // 1 == opaque; 0 == fully transparent
		// illumination model (see http://www.fileformat.info/format/material/)
		int illum;

		std::string ambient_texname;
		std::string diffuse_texname;
		std::string specular_texname;
		std::string normal_texname;
		std::map<std::string, std::string> unknown_parameter;
	} material_t;

	typedef struct {
		std::vector<float> positions;
		std::vector<float> normals;
		std::vector<float> texcoords;
		std::vector<unsigned int> indices;
		std::vector<int> material_ids; // per-mesh material ID
	} mesh_t;

	typedef struct {
		std::string name;
		mesh_t mesh;
	} shape_t;

	void loadGLTF(std::string filename);
	void loadOBJ(std::string filename);
	tinygltf::Model LoadModel(std::string& filepath);
}