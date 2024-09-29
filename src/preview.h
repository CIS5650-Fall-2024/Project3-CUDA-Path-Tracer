#pragma once

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData, int stratifiedSamples, float focalLength_, float apertureSize_);