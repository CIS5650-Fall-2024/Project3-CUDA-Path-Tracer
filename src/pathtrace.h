#pragma once

#include <vector>
#include "scene.h"

#define STREAM_COMPACTION   1
#define MATERIAL_SORT       0
#define ANTI_ALIASING       1
#define DEPTH_OF_FIELD      0
#define RUSSIAN_ROULETTE    1

struct RayHasIntersected {
    __host__ __device__ bool operator()(const PathSegment& path) const {
        return path.remainingBounces != 0;
    }
};

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
