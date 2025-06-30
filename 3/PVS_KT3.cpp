#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

void fillMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = rand() % 100;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 2000;
    double* A = nullptr, * B = new double[N * N], * C = nullptr;

    if (rank == 0) {
        A = new double[N * N];
        C = new double[N * N];
        fillMatrix(A, N, N);
        fillMatrix(B, N, N);
    }


    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    int rows_per_proc = N / size;
    double* local_A = new double[rows_per_proc * N];
    double* local_C = new double[rows_per_proc * N];

    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
        local_A, rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);


    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            local_C[i * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();


    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
        C, rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time (" << size << " processes): " << duration.count() << " sec" << std::endl;
        delete[] A;
        delete[] C;
    }

    delete[] B;
    delete[] local_A;
    delete[] local_C;

    MPI_Finalize();
    return 0;
}

