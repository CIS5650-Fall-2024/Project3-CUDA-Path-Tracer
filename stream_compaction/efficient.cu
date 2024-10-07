#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define threadsPerBlock 1024
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void scan_upstream(int n, int *idata,int offset)
        {
            int thid = threadIdx.x + (blockIdx.x * blockDim.x);
            //load data in the global memory
            int ai = offset * (2*thid + 1) - 1;
            int bi = offset * (2*thid + 2) - 1;
           
            if (ai < n && bi < n)
            {

                idata[bi] += idata[ai];                           
            }
            if(thid == 0)
                idata[n-1] =0;
            __syncthreads();

        }



        __global__ void scan_downstream(int n, int* idata,int offset)
        {
            int thid = threadIdx.x + (blockIdx.x * blockDim.x);
            //load data in the global memory           
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            if ((ai < n) && (bi < n)) {
                float t = idata[ai];
                idata[ai] = idata[bi];
                idata[bi] += t;
                }
                  
             __syncthreads();

        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int *g_idata;
            int zeropadded_n = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&g_idata,zeropadded_n * sizeof(int));

            cudaMemcpy(g_idata,idata,n*sizeof(int),cudaMemcpyHostToDevice);

            // int threadsPerBlock = 1024;
            int blocksPerGrid = ((zeropadded_n /2) + threadsPerBlock - 1) / threadsPerBlock;
            int offset = 1;
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                scan_upstream << <blocksPerGrid, threadsPerBlock >> > (zeropadded_n, g_idata,offset);
                offset *= 2;
                blocksPerGrid = ((zeropadded_n / (2 * offset)) + threadsPerBlock - 1) / threadsPerBlock;
            }

            cudaDeviceSynchronize();
            
            offset = zeropadded_n / 2;
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                blocksPerGrid = ((zeropadded_n / (2 * offset)) + threadsPerBlock - 1) / threadsPerBlock;
                scan_downstream << <blocksPerGrid, threadsPerBlock >> > (zeropadded_n, g_idata,offset);
                offset /= 2;
                
            }
            cudaMemcpy(odata,g_idata,sizeof(int)*n,cudaMemcpyDeviceToHost);
            cudaFree(g_idata);
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int *g_bools = 0;
            int *bools = 0;
            int* indices;
            int* g_idata;
            int *g_odata;
            int zeropadded_n = pow(2,ilog2ceil(n));
            printf("zeropadded = %d\n", zeropadded_n);
            // int threadsPerBlock = 256;
            int blocksPerGrid = (zeropadded_n + threadsPerBlock - 1) / threadsPerBlock;
            int *temp_array = (int*)malloc(sizeof(int)*zeropadded_n);
            cudaError_t result = cudaMalloc((void**)(&g_bools), zeropadded_n * sizeof(int));
            if (result != cudaSuccess) {
                fprintf(stderr, "Mem alloc failed: %s\n", cudaGetErrorString(result));
                cudaFree(g_bools);
                timer().endGpuTimer();
                return -1;
            }
            result = cudaMalloc((void**)(&g_idata), zeropadded_n * sizeof(int));
            cudaMemset(g_idata, 0, zeropadded_n*sizeof(int));
            if (result != cudaSuccess) {
                fprintf(stderr, "Mem alloc failed: %s\n", cudaGetErrorString(result));
                cudaFree(g_bools);
                cudaFree(g_idata);
                timer().endGpuTimer();
                return -1;
            }
            cudaMemcpy(g_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&g_odata,sizeof(int)* zeropadded_n);
            StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid,threadsPerBlock>>>(zeropadded_n, g_bools, g_idata);
            bools = (int*)malloc(zeropadded_n*sizeof(int));
            cudaDeviceSynchronize();
            cudaMemcpy(bools,g_bools,zeropadded_n*sizeof(int),cudaMemcpyDeviceToHost);
            
            result = cudaMalloc(&indices, zeropadded_n * sizeof(int));
            timer().endGpuTimer();
            scan(zeropadded_n, temp_array, bools);
            
            timer().startGpuTimer();
            cudaMemcpy(indices, temp_array, zeropadded_n * sizeof(int), cudaMemcpyHostToDevice);

            StreamCompaction::Common::kernScatter<<<blocksPerGrid,threadsPerBlock>>>(zeropadded_n, g_odata,g_idata, g_bools, indices);
            cudaMemcpy(odata,g_odata,zeropadded_n*sizeof(int),cudaMemcpyDeviceToHost);
            cudaFree(g_bools);
            cudaFree(indices);
            cudaFree(g_odata);
            cudaFree(g_idata);
            free(bools);
            timer().endGpuTimer();

            return temp_array[zeropadded_n-1];
        }
    }
}
