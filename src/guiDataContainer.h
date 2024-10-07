#include "scene.h"

class GuiDataContainer
{
public:
    GuiDataContainer(Scene* const scene, bool* const camChanged);
    int TracedDepth;
    Scene* const scene;
    bool* const camChanged;
};