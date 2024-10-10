#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

class Scene
{
private:
    std::ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
    void addPlaneAsMesh(const glm::vec3& position, const glm::vec3& normal, float width, float height, int materialId);


public:
    Scene(std::string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    RenderState state;

    // For custom meshes
    std::vector<Triangle> triangles;
};
