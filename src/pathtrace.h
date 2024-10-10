#pragma once

#include <vector>

#include <thrust/partition.h>
#include "scene.h"
#include "warp.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
