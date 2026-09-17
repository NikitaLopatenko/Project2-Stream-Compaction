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
            if (n <= 0) {
                return;
            }

            // prepare everything beforehand
            thrust::host_vector<int> hv_input(idata, idata + n);
            thrust::device_vector<int> dv_input = hv_input;
            thrust::device_vector<int> dv_output(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(dv_input.begin(), dv_input.end(), dv_output.begin());
            timer().endGpuTimer();

            thrust::copy(dv_output.begin(), dv_output.end(), odata);
        }
    }
}
