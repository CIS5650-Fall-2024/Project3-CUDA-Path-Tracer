#pragma once

#include <vector>

#include <thrust/partition.h>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree(Scene *scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
