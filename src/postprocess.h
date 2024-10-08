#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace postprocess {
    __host__ __device__ glm::vec3 gammaCorrection(glm::vec3 color, float gamma);

    // See https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/.
    __host__ __device__ glm::vec3 ACESToneMapping(glm::vec3 color);
}
