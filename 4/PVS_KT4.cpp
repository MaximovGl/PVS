#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void fillMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = rand() % 100;
        }
    }
}

int main() {
    const int N = 2000;
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    fillMatrix(A, N, N);
    fillMatrix(B, N, N);

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time: " << duration.count() << " sec" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}