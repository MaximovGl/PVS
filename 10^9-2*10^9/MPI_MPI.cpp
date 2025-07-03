#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    using namespace std;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long a = 1000000000LL;
    const long long b = 2000000000LL;
    const long long segment_size = 500000LL;
    if (rank == 0) {
        cout << "Interval [" << a << ", " << b << "]" << endl;
    }

    double start_time = MPI_Wtime();
    const int limit = static_cast<int>(sqrt(b)) + 1;


    vector<char> sieve(limit + 1, 1);
    vector<long long> primes;

    if (limit >= 2) {
        sieve[0] = sieve[1] = 0;
        for (int i = 2; i * i <= limit; i++) {
            if (sieve[i]) {
                for (int j = i * i; j <= limit; j += i) {
                    sieve[j] = 0;
                }
            }
        }

        for (int i = 2; i <= limit; i++) {
            if (sieve[i]) primes.push_back(i);
        }
    }

    const long long total_numbers = b - a + 1;
    const long long base_chunk_size = total_numbers / size;
    const long long remainder = total_numbers % size;

    long long local_a = a + rank * base_chunk_size;
    long long local_b = local_a + base_chunk_size - 1;

    if (rank < remainder) {
        local_a += rank;
        local_b += rank + 1;
    }
    else {
        local_a += remainder;
        local_b += remainder;
    }

    if (rank == size - 1) {
        local_b = b;
    }

    long long local_count = 0;
    const long long local_segments = (local_b - local_a + segment_size) / segment_size;

    for (long long seg = 0; seg < local_segments; seg++) {
        const long long low = local_a + seg * segment_size;
        const long long high = min(low + segment_size - 1, local_b);

        long long start_num = low;
        if (start_num % 2 == 0) start_num++;

        const long long n_odd = (high < start_num) ? 0 : ((high - start_num) / 2 + 1);
        if (n_odd == 0) continue;

        vector<char> segment(n_odd, 1);

        for (size_t k = 1; k < primes.size(); ++k) {
            const long long p = primes[k];
            long long start = (low + p - 1) / p * p;
            if (start < low) start += p;
            if (start % 2 == 0) start += p;
            if (start > high) continue;

            for (long long j = start; j <= high; j += 2 * p) {
                const long long idx = (j - start_num) / 2;
                if (idx < n_odd) segment[idx] = 0;
            }
        }

        for (long long i = 0; i < n_odd; ++i) {
            local_count += segment[i];
        }
    }

    if (local_a <= 2 && 2 <= local_b) local_count++;

    vector<long long> all_counts;
    if (rank == 0) {
        all_counts.resize(size);
    }

    MPI_Gather(&local_count, 1, MPI_LONG_LONG,
        rank == 0 ? all_counts.data() : nullptr, 1, MPI_LONG_LONG,
        0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        long long total_count = 0;
        for (int i = 0; i < size; ++i) {
            total_count += all_counts[i];
        }
        cout << "\nPrime count between " << a << " and " << b << ": "
            << total_count << endl;
        cout << "Total time: " << elapsed_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
