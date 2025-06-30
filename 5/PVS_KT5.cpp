#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>

double f(double x) {
    return std::cos(x) / (std::log(1 + std::sin(x)) * std::sin(1 + std::sin(x)));
}

int main() {
    double a = 1e-9;
    double b = M_PI - 1e-9;
    const int total_computational_units = 16;
    double delta_x = (b - a) / total_computational_units;

    std::cout << "Integrating f(x) = cos(x) / (ln(1+sin(x)) * sin(1+sin(x)))" << std::endl;
    std::cout << "Interval: [" << a << ", " << b << "]" << std::endl;
    std::cout << "Total computational units (N): " << total_computational_units << std::endl;
    std::cout << "Delta x: " << delta_x << std::endl;

    std::vector<double> partial_sums(total_computational_units);

    double start_time = omp_get_wtime();

#pragma omp parallel for
    for (int idx = 0; idx < total_computational_units; ++idx) {
        double x_i = a + (idx + 0.5) * delta_x;
        partial_sums[idx] = f(x_i) * delta_x;
    }

    double end_time = omp_get_wtime();

    double total_sum = 0.0;
    for (int i = 0; i < total_computational_units; ++i) {
        total_sum += partial_sums[i];
    }

    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Integral result: " << total_sum << std::endl;
    std::cout << "Execution time: " << (end_time - start_time) * 1000 << " ms" << std::endl;

    return 0;
}