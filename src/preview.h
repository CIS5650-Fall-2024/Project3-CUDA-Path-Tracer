#pragma once

extern GLuint pbo;
extern bool doMaterialSorting;

std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);