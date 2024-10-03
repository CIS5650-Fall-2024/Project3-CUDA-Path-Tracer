#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void parse_gltf_node(
        const tinygltf::Model &model,
        int node,
        const glm::mat4 &base_transform = glm::mat4{}
    );
    void load_from_gltf(const std::string &gltf_filename);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    RenderState state;
};
