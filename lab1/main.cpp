#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include<chrono>
using namespace std;


using Matrix = vector<vector<int>>;



Matrix read_sq_matrix_from_file(const string& filename) {
    ifstream in(filename);
    if (!in) {
        cerr << "Error opening file: " << filename << endl;
        return {};
    }

    Matrix matrix;
    string line;

    while (getline(in, line)) {
        if (line.empty()) continue;            

        istringstream iss(line);
        vector<int> row;
        int x;

        while (iss >> x) {
            row.push_back(x);
        }

        if (!row.empty())
            matrix.push_back(row);
    }
    return matrix;
}

void write_sq_matrix_to_file(const Matrix& matrix, const string& filename) {
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Error opening file: " + filename);
    }

    int n = static_cast<int>(matrix.size());
    if (n == 0) return;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << matrix[i][j];
            if (j + 1 < n) out << ' ';
        }
        out << '\n';
    }
}

Matrix mul_sq_matrix(const Matrix& matrix_a, const Matrix& matrix_b){
    int n = static_cast<int>(matrix_a.size());
    Matrix result_matrix(n, vector<int>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += matrix_a[i][k] * matrix_b[k][j];
            }
            result_matrix[i][j] = sum;
        }
    }

    return result_matrix;
}

int main(){
    try{
        Matrix matrix_a = read_sq_matrix_from_file("matrix_a");
        Matrix matrix_b = read_sq_matrix_from_file("matrix_b");

        int n = static_cast<int>(matrix_a.size());

        long long operations_counts = 2LL * n * n * n;

        auto start = chrono::high_resolution_clock::now();

        Matrix result_mul_matrix = mul_sq_matrix(matrix_a, matrix_b);

        auto end = chrono::high_resolution_clock::now();

        write_sq_matrix_to_file(result_mul_matrix, "result_matrix");
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        cout << "Matrix multiplication completed in " << duration << " milliseconds." << endl;
        cout<<"Count of operation = "<<operations_counts<<endl;

        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}