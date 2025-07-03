#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

int main() {
    const long long a = 1000000000LL;
    const long long b = 2000000000LL;
    const long long segment_size = 500000LL;

    std::cout << "Interval [" << a << ", " << b << "]" << std::endl;
    const int limit = static_cast<int>(std::sqrt(b)) + 1;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<bool> sieve(limit + 1, true);
    std::vector<long long> primes;
    sieve[0] = sieve[1] = false;

    for (int i = 2; i * i <= limit; i++) {
        if (sieve[i]) {
            for (int j = i * i; j <= limit; j += i) {
                sieve[j] = false;
            }
        }
    }

    for (int i = 2; i <= limit; i++) {
        if (sieve[i]) primes.push_back(i);
    }

    std::cout << "Found " << primes.size() << " primes up to " << limit << std::endl;

    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads" << std::endl;

    const long long total_segments = (b - a + segment_size - 1) / segment_size;
    long long total_count = 0;
    long long processed_segments = 0;
    double last_report_time = omp_get_wtime();

#pragma omp parallel reduction(+:total_count)
    {
        long long local_count = 0;
        double last_local_report = omp_get_wtime();
        const int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic, 1)
        for (long long seg = 0; seg < total_segments; seg++) {
            const long long low = a + seg * segment_size;
            const long long high = std::min(low + segment_size - 1, b);

            long long start_num = low;
            if (start_num % 2 == 0) start_num++;

            const long long n_odd = (high < start_num) ? 0 : ((high - start_num) / 2 + 1);
            if (n_odd == 0) continue;

            std::vector<char> segment(n_odd, 1);

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
                if (segment[i]) local_count++;
            }

#pragma omp atomic
            processed_segments++;

            if (thread_id == 0) {
                const double current_time = omp_get_wtime();
                if (current_time - last_local_report > 10.0) {
                    const double progress = 100.0 * processed_segments / total_segments;
#pragma omp critical
                    {
                        std::cout << "Progress: " << progress << "% ("
                            << processed_segments << "/" << total_segments << " segments)"
                            << std::endl;
                    }
                    last_local_report = current_time;
                }
            }
        }

        total_count += local_count;
    }

    if (a <= 2 && 2 <= b) total_count++;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Prime count: " << total_count << std::endl;
    std::cout << "Total time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
