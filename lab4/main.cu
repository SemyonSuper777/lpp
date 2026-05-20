#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;

using Matrix1D = vector<int>;

inline void check_cuda(cudaError_t code, const char* filename, int lineno) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%d) at %s:%d\n",
            cudaGetErrorString(code), code, filename, lineno);
        exit(code);
    }
}

#define CUDA_CHECK(call) check_cuda(call, __FILE__, __LINE__)

Matrix1D generate_sq_matrix(int n, int minValue, int maxValue) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(minValue, maxValue);

    Matrix1D matrix(n * n);
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

void write_sq_matrix_to_file(const Matrix1D& matrix, int n, const string& filename) {
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Error opening file: " + filename);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << matrix[i * n + j];
            if (j + 1 < n) out << ' ';
        }
        out << '\n';
    }
}

void print_matrix_fragment(const Matrix1D& matrix, int n, int limit = 5) {
    int rows = min(n, limit);
    int cols = min(n, limit);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i * n + j] << "\t";
        }
        cout << endl;
    }
}

__global__ void mul_sq_matrix_kernel(const int* A, const int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            cerr << "Usage: " << argv[0]
                << " N blockX blockY [minValue maxValue]" << endl;
            cerr << "Example: " << argv[0] << " 512 16 16 0 9" << endl;
            return 1;
        }

        int n = stoi(argv[1]);
        int blockX = stoi(argv[2]);
        int blockY = stoi(argv[3]);

        int minValue = 0;
        int maxValue = 9;

        if (argc >= 6) {
            minValue = stoi(argv[4]);
            maxValue = stoi(argv[5]);
        }

        if (n <= 0) throw runtime_error("N must be > 0");
        if (blockX <= 0 || blockY <= 0) throw runtime_error("Block dimensions must be > 0");
        if (minValue > maxValue) throw runtime_error("minValue must be <= maxValue");
        if (blockX * blockY > 1024) throw runtime_error("Too many threads per block for CUDA");

        Matrix1D matrix_a = generate_sq_matrix(n, minValue, maxValue);
        Matrix1D matrix_b = generate_sq_matrix(n, minValue, maxValue);
        Matrix1D result_matrix(n * n, 0);

        size_t bytes = n * n * sizeof(int);
        long long operations_counts = 2LL * n * n * n;

        int* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, matrix_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, matrix_b.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(blockX, blockY);
        dim3 grid((n + block.x - 1) / block.x,
            (n + block.y - 1) / block.y);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        mul_sq_matrix_kernel << <grid, block >> > (d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaMemcpy(result_matrix.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Random values range: [" << minValue << ", " << maxValue << "]" << endl;
        cout << "Block config: (" << blockX << ", " << blockY << ")" << endl;
        cout << "Grid config: (" << grid.x << ", " << grid.y << ")" << endl;
        cout << "Kernel execution time: " << milliseconds << " ms" << endl;
        cout << "Count of operation = " << operations_counts << endl;

        if (n <= 10) {
            cout << "\nMatrix A fragment:\n";
            print_matrix_fragment(matrix_a, n);
            cout << "\nMatrix B fragment:\n";
            print_matrix_fragment(matrix_b, n);
            cout << "\nResult matrix fragment:\n";
            print_matrix_fragment(result_matrix, n);
        }

        write_sq_matrix_to_file(result_matrix, n, "result_matrix");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}