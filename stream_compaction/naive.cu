#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define block_size 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scan_global(int n, int *odata, int *idata, int *temp,int offset,int pout)
        {
            int thid = threadIdx.x + (blockIdx.x * blockDim.x);
            // Load input into global memory.
            // This is exclusive scan, so shift right by one
            // and set first element to 0
            int pin = 1 - pout;
            if (thid >= offset)
                temp[pout * n + thid] = temp[pin * n + thid - offset] + temp[pin* n + thid];
            else
                temp[pout * n + thid] = temp[pin * n + thid];
            __syncthreads();
            odata[thid] = temp[pout * n + thid]; // write output
        }

        __global__ void shiftInput(int* idata, int* shifted_input)
        {
            int thid = threadIdx.x + (blockIdx.x * blockDim.x);
            shifted_input[thid] = (thid > 0) ? idata[thid - 1] : 0;
            __syncthreads();
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int *g_odata,*g_idata,*temp;
            int zeropadded_n = pow(2, ilog2ceil(n));
            cudaError_t result = cudaMalloc((void**)&g_idata, zeropadded_n * sizeof(int));
            result = cudaMalloc((void**)&g_odata,zeropadded_n*sizeof(int));
            result = cudaMalloc((void**)&temp,2 * zeropadded_n * sizeof(int));
            cudaMemcpy(g_idata,idata,sizeof(int)*n,cudaMemcpyHostToDevice);
            int threadsPerBlock = 1024;
            int blocksPerGrid = (zeropadded_n + threadsPerBlock - 1) / threadsPerBlock;

            int offset = 1;
            int pout = 0;
            shiftInput<<<blocksPerGrid,threadsPerBlock>>>(g_idata, temp);
            for (int i = 0; i < ilog2ceil(n); i++) {
                pout = 1 - pout;
                scan_global<<<blocksPerGrid,threadsPerBlock>>>(zeropadded_n,g_odata,g_idata,temp,offset,pout);
                offset *= 2;
            }

            cudaMemcpy(odata,g_odata,sizeof(int)*n,cudaMemcpyDeviceToHost);
            for (int i = 0; i < 257; i++)
            {
                //printf("%d %d\n", idata[i], odata[i]);
            }
            timer().endGpuTimer();
        }
    }
}
