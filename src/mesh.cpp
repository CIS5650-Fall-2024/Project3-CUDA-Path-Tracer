#include "mesh.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"
#include <string>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp> 
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "mesh.h"


using namespace std;

void mesh::loadGLTF(std::string filename) {

}


tinygltf::Model mesh::LoadModel(std::string& filepath) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    //ascii file
    //bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, argv[1]);
    //glb file
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF\n");
    }

    return model;
}

