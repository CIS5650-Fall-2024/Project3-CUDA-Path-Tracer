#pragma once
#include "PTDirectives.h"

extern GLuint pbo;
extern GLuint pbo_post;

std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);



