#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
	ifstream fp_in;
	void loadFromJSON(const std::string& jsonName);
	void parseObjFileToVertices(const std::string& filepath, Geom& geom);
	void Scene::parseImgFileToTextures(const std::string& filepath, Geom& geom, std::vector<Texture>& global_textures);
	void Scene::parseNormImgFileToTextures(const std::string& filepath, Geom& geom, std::vector<Texture>& global_norm_textures);
public:
	Scene(string filename);
	~Scene();

	std::vector<Geom> geoms;
	std::vector<Material> materials;
	std::vector<Vertex> vertices;
	std::vector<Texture> textures;
	std::vector<Texture> norm_textures;
	RenderState state;
};
