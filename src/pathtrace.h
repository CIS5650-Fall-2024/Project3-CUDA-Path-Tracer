#pragma once

#include <vector>
#include "scene.h"
#include "bvh.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, uchar4* pbo_post, int frame, int iteration);
