#pragma once

#include <vector>
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

// sort by material
#define SORTMATERIAL 1

// stream compaction
#define STREAM_COMPACTION 1

// Russian roulette
#define RUSSIAN_ROULETTE 1
#define MIN_BOUNCES 3
#define MIN_SURVIVAL_PROB 0.05f

// BVH
#define USE_BVH 1

// OIDA
#define USE_OIDN 1
#define DENOISE_INTERVAL 1000

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
