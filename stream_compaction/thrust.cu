#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *dev_odata, const int *dev_idata) {
            // Wrap the raw device pointer in a thrust::device_ptr
            thrust::device_ptr<const int> dev_ptr_idata = thrust::device_pointer_cast(dev_idata);
            thrust::device_ptr<int> dev_ptr_odata = thrust::device_pointer_cast(dev_odata);

            // Perform exclusive scan directly using thrust pointers
            thrust::exclusive_scan(dev_ptr_idata, dev_ptr_idata + n, dev_ptr_odata);
        }
    }
}
