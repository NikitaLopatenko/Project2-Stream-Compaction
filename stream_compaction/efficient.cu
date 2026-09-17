#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int offset = 2 * d;

            if (i >= n / offset) return;

            int ai = i * offset + d - 1;
            int bi = i * offset + offset - 1;
            data[bi] += data[ai];
        }

        __global__ void kernDownSweep(int n, int d, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int offset = 2 * d;

            if (i >= n / offset) return;

            int ai = i * offset + d - 1;
            int bi = i * offset + offset - 1;
            int t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
         void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            int nPow2 = 1;
            while (nPow2 < n) {
                nPow2 *= 2;
            }

            const int blockSize = 256;
            int *dev = nullptr;

            cudaMalloc(&dev, nPow2 * sizeof(int));
            cudaMemset(dev, 0, nPow2 * sizeof(int));
            cudaMemcpy(dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // upsweep
            for (int d = 1; d < nPow2; d *= 2) {
                int usedThreads = nPow2 / (2 * d);
                int blocks = (usedThreads + blockSize - 1) / blockSize;
                if (blocks > 0) {
                    kernUpSweep<<<blocks, blockSize>>>(nPow2, d, dev);
                }
            }

            // we make it exclusive
            cudaMemset(dev + nPow2 - 1, 0, sizeof(int));

            // downsweep
            for (int d = nPow2 / 2; d >= 1; d /= 2) {
                int usedThreads = nPow2 / (2 * d);
                int blocks = (usedThreads + blockSize - 1) / blockSize;
                if (blocks > 0) {
                    kernDownSweep<<<blocks, blockSize>>>(nPow2, d, dev);
                }
            }
            timer().endGpuTimer();


            cudaMemcpy(odata, dev, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev);
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
            if (n <= 0){
                return 0;
            }

            const int blockSize = 256;
            int blocks = (n + blockSize - 1)/ blockSize;

            int nPow2 = 1;
            while (nPow2 < n){
                nPow2 *= 2;
            }

            int *devInput = nullptr;
            int *devOutput = nullptr;
            int *devBools = nullptr;
            int *devIndices = nullptr;

            cudaMalloc(&devInput, n * sizeof(int));
            cudaMalloc(&devOutput, n * sizeof(int));
            cudaMalloc(&devBools, n * sizeof(int));
            cudaMalloc(&devIndices, nPow2 * sizeof(int));
            cudaMemcpy(devInput, idata, n * sizeof(int), cudaMemcpyHostToDevice);



            timer().startGpuTimer();
            // map input to 1 or 0
            StreamCompaction::Common::kernMapToBoolean<<<blocks, blockSize>>>(n, devBools, devInput);

            // exclusive scan to get indices
            cudaMemset(devIndices, 0, nPow2 * sizeof(int));
            cudaMemcpy(devIndices, devBools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int d = 1; d < nPow2; d *= 2) {
                int usedThreads = nPow2 / (2 * d);
                int b = (usedThreads + blockSize - 1) / blockSize;
                if (b > 0) {
                    kernUpSweep<<<b, blockSize>>>(nPow2, d, devIndices);
                }
            }

            cudaMemset(devIndices + nPow2 - 1, 0, sizeof(int));
            for (int d = nPow2 / 2; d >= 1; d /= 2) {
                int usedThreads = nPow2 / (2 * d);
                int b = (usedThreads + blockSize - 1) / blockSize;
                if (b > 0) {
                    kernDownSweep<<<b, blockSize>>>(nPow2, d, devIndices);
                }
            }

            // scatter
            StreamCompaction::Common::kernScatter<<<blocks, blockSize>>>(n, devOutput, devInput, devBools, devIndices);
            timer().endGpuTimer();


            
            int lastIndex = 0;
            int lastBool = 0;
            cudaMemcpy(&lastIndex, devIndices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, devBools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int count = lastIndex + lastBool;
            if (count > 0) {
                cudaMemcpy(odata, devOutput, count * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(devInput);
            cudaFree(devBools);
            cudaFree(devIndices);
            cudaFree(devOutput);
            return count;
        }
    }
}
