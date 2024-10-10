#pragma once

namespace StreamCompaction {
    namespace Thrust {
        void scan(int n, int *dev_odata, const int *dev_idata);
        
        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        template <typename T>
        __global__ void scatter(int num_paths, T* odata, const T *idata, const int *bools, const int *indices) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_paths) {
                return;
            }

            if (bools[idx]) {
                odata[indices[idx]] = idata[idx];
            }
        }
    }
}
