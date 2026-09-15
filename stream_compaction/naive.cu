#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void simpleKernScan(int n, int d, int *odata, const int *idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n ) return;
            if (i < d){
                odata[i] = idata[i];;
            }else{
                odata[i] = idata[i] + idata[i-d];
            }
        }


            
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            const int blockSize = 256;
            int blocks = (n + blockSize - 1) / blockSize;
            int *A = nullptr;
            int *B = nullptr;
            cudaMalloc(&A, n * sizeof(int));
            cudaMalloc(&B, n * sizeof(int));
            cudaMemcpy(A, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int d = 1; d < n; d *= 2) {
                simpleKernScan<<<blocks, blockSize>>>(n, d, B, A);
                
                int *temp = A;
                A = B;
                B = temp;
            }
            timer().endGpuTimer();
            
            
            odata[0] = 0;
            if (n > 1) {
                cudaMemcpy(odata + 1, A, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            }
            cudaFree(A);
            cudaFree(B);
        }
    }
}
