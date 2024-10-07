#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <stb_image.h>  
#include <stb_image_write.h>  
#include <tiny_gltf.h> 
#include <tiny_obj_loader.h>
#include <unordered_map>


using namespace std;

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName, Geom& newGeom, std::unordered_map<std::string, uint32_t>& MatNameToID, glm::mat4 transformed);
    void loadTexture(const std::string& texName, Geom& newGeom, std::string path);
    void loadNormal(const std::string& texName, Geom& newGeom, std::string path);
    void loadEnv(const std::string& envName, std::string path);
    void UpdateNodeBounds(int& nodeIdx);
    void Subdivide(int& nodeIdx);
    void BuildBVH();
   
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    //add for mesh
    std::vector<Triangle> triangles;
    std::vector<int> triIdx;
    std::vector<Texture> textures;
    std::vector<Texture> normals;
    std::vector<Texture> envs;
    std::vector<BVHNode> bvhNodes;

    RenderState state;
    //int rootNodeIdx = 0
    int nodesUsed;
};
