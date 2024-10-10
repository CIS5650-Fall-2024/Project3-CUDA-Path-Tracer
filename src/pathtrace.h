#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

__device__ glm::vec2 barycentricInterpolation(
    const glm::vec2& uv0,
    const glm::vec2& uv1,
    const glm::vec2& uv2,
    const glm::vec3& baryPosition);
__device__ void computeTangentSpace(
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2,
    const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
    const glm::vec3& baryPosition,
    glm::vec3& tangent, glm::vec3& bitangent, glm::vec3& normal);