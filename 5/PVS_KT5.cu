#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

#define CUDA_CHECK(api_call) \
    do { \
        cudaError_t error_status = (api_call); \
        if (error_status != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " for call '" << #api_call << "': " \
                      << cudaGetErrorString(error_status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ double device_f(double x) {
    return std::cos(x) / (std::log(1 + std::sin(x)) * std::sin(1 + std::sin(x)));
}

__global__ void integrate_kernel(double a, double delta_x, int n_steps, double* partial_sums_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_steps) {
        double x_i = a + (idx + 0.5) * delta_x;
        partial_sums_d[idx] = device_f(x_i) * delta_x;
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    double a = 1e-9;
    double b = M_PI - 1e-9;
    const int total_computational_units = 1000000;
    const int threads_per_block = 1024;

    double delta_x = (b - a) / total_computational_units;

    std::cout << "Integrating f(x) = cos(x) / (ln(1+sin(x)) * sin(1+sin(x)))" << std::endl;
    std::cout << "Interval: [" << a << ", " << b << "]" << std::endl;
    std::cout << "Total computational units (N): " << total_computational_units << std::endl;
    std::cout << "Delta x: " << delta_x << std::endl;

    size_t size = total_computational_units * sizeof(double);
    double* partial_sums_d;
    std::vector<double> partial_sums_h(total_computational_units);

    CUDA_CHECK(cudaMalloc(&partial_sums_d, size));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    int num_blocks = (total_computational_units + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaEventRecord(start_event));

    integrate_kernel << <num_blocks, threads_per_block >> > (a, delta_x, total_computational_units, partial_sums_d);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(partial_sums_h.data(), partial_sums_d, size, cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    double total_sum = 0.0;
    for (int i = 0; i < total_computational_units; ++i) {
        total_sum += partial_sums_h[i];
    }

    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Integral result: " << total_sum << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    CUDA_CHECK(cudaFree(partial_sums_d));

    return 0;
}