#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            for(int i = 0; i<n; i++){
                if(i == 0)
                    odata[i] = 0;
                else
                    odata[i] = odata[i-1] + idata[i-1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for(int i = 0;i<n;i++)
            {
                if(idata[i] != 0)
                    odata[j++] = idata[i];
            }
            timer().endCpuTimer();
            return j;
            
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int *temp_array = (int*)malloc(n*sizeof(int));
            int *scan_array = (int*)malloc(n*sizeof(int));
            for(int i = 0; i < n; i++)
                temp_array[i] = (idata[i] != 0);
            for(int i = 0; i<n; i++){
                if(i == 0)
                    scan_array[i] = 0;
                else
                    scan_array[i] = temp_array[i-1] + scan_array[i-1];
            }
            for(int i = 0; i < n;i++)
            {
                if(temp_array[i] == 1)
                {
                    odata[scan_array[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return scan_array[n-1];
        }
    }
}
