#pragma once

extern GLuint pbo;


//static float percentDenoise;
std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);