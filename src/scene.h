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
    GeomType getGeometryType(const std::string& type);
    void loadFromJSON(const std::string& jsonName);

    glm::vec3 compute_centroid(const std::vector<Mesh_Data>& data, size_t index);
    float compute_bounding_radius(const std::vector<Mesh_Data>& data, std::size_t index, const glm::vec3& centroid);
    void update_bounding_box(glm::vec3& min_corner, glm::vec3& max_corner, const glm::vec3& centroid, float radius);
    int determine_separation_axis(const glm::vec3& dimensions, float max_length);
    int determine_target_index(const glm::vec3& centroid, const glm::vec3& center, float radius, float avg_radius, int axis);

    int loadTexture(const std::string& name);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    

    std::vector<Texture_Data> m_textures;
    std::vector<Mesh_Data> m_data;
    std::vector<glm::vec4> pixels;

    std::vector<BVH_Data> bvh_datas;
    std::vector<BVH_Main_Data> bvh_main_datas;
};
