#pragma once

#include <vector>
#include "scene.h"
#include <OpenImageDenoise/oidn.hpp>

#define DIRECTIONALLIGHT 0
#define sunDir glm::vec3(0.6, -0.3, -1)
#define sunCol glm::vec3(1, 1, 1)

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4* pbo,
		oidn::FilterRef& oidn_filter,
		float& percentD,
		int frame, int iteration);