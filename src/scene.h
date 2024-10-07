#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "TinyObj/tiny_obj_loader.h"
#include <unordered_map>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
	void loadTexture(const std::string& texturePath, Texture* texture);
	void writeMeshInfo(tinyobj::attrib_t* attrib, 
        Geom* geom, 
        std::vector<tinyobj::shape_t>& shapes, 
        std::vector<tinyobj::material_t>& materials, 
        std::vector<Material>* scene_materials);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
	Texture* skyboxTexture = nullptr;
	int num_meshes = 0; // number of meshes in the scene
};
