#pragma once

#include <vector>
#include "scene.h"
#include "sceneStructs.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
bool createCudaTexture(TextureData& textureData, Texture& textureObj);
bool createCudaTexture_hdr(EnvData_hdr& textureData, Texture& textureObj);