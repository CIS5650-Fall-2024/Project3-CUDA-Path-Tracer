#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            thrust::host_vector<int> thrust_idata(idata, idata + n);
            thrust::host_vector<int> thrust_odata(odata, odata + n);

            thrust::device_vector<int> thrust_dev_idata(idata,idata+n);
            thrust::device_vector<int> thrust_dev_odata(odata,odata+n);

            thrust::exclusive_scan(thrust_dev_idata.begin(), thrust_dev_idata.end(), thrust_dev_odata.begin());
            thrust::copy(thrust_dev_odata.begin(), thrust_dev_odata.end(), odata);
            timer().endGpuTimer();
        }
    }
}
