#pragma once

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "utilities.h"
#include "sceneStructs.h"

enum class TextureType {
    ALBEDO,
    NORMAL,
    BUMP
};

bool endsWith(const std::string& str, const std::string& suffix);

void loadOBJ(
    const std::string &filepath, 
    std::vector<Triangle> &faces, 
    std::vector<glm::vec3> &verts, 
    std::vector<glm::vec3> &normals, 
    std::vector<glm::vec2> &uvs,
    std::vector<int> &indices);

void loadGLTFOrGLB(
    const std::string &filepath, 
    std::vector<Triangle> &faces, 
    std::vector<glm::vec3> &verts, 
    std::vector<glm::vec3> &normals, 
    std::vector<int> &indices, 
    std::vector<std::tuple<std::string, glm::vec4*, glm::ivec2>> &albedoTextures, 
    std::vector<std::tuple<std::string, glm::vec4*, glm::ivec2>> &normalTextures);

void loadTexture(const std::string& filepath, const std::string& textureType, glm::vec4* &texture, glm::ivec2 &textureSize);