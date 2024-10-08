#include "postprocess.h"

namespace postprocess {
    __host__ __device__ glm::vec3 gammaCorrection(glm::vec3 color, float gamma)
    {
        return glm::pow(color, glm::vec3(1.f / gamma));
    }

    // See https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/.
    __host__ __device__ glm::vec3 ACESToneMapping(glm::vec3 color) {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        return glm::clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
    }
}
