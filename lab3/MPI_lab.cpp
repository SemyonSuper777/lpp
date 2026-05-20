#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    Matrix A, B;

    auto start = chrono::high_resolution_clock::now();

    if (rank == 0) {
        A = read_sq_matrix_from_file("matrix_a");
        B = read_sq_matrix_from_file("matrix_b");

        n = static_cast<int>(A.size());
        if (n == 0 || (int)A[0].size() != n || (int)B.size() != n || (int)B[0].size() != n) {
            cerr << "Matrices must be non-empty and square\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Рассылаем всем размер n
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Вычисляем количество строк для каждого процесса
    vector<int> rows_per_proc(size);
    vector<int> displs_rows(size);  // смещения по строкам в A/C

    int base = n / size;
    int rem = n % size;

    for (int p = 0; p < size; ++p) {
        rows_per_proc[p] = base + (p < rem ? 1 : 0);
    }
    displs_rows[0] = 0;
    for (int p = 1; p < size; ++p) {
        displs_rows[p] = displs_rows[p - 1] + rows_per_proc[p - 1];
    }

    // Линейные представления A, B, C
    vector<int> A_flat;
    vector<int> B_flat(n * n);
    vector<int> C_flat;  // только на ранге 0

    if (rank == 0) {
        A_flat.resize(n * n);
        C_flat.resize(n * n);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A_flat[i * n + j] = A[i][j];

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                B_flat[i * n + j] = B[i][j];
    }

    // Рассылаем B целиком всем
    MPI_Bcast(B_flat.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // Подготавливаем counts/disp для Scatterv/Gatherv в элементах (int)
    vector<int> sendcounts(size), displs(size);
    vector<int> recvcounts(size), rdispls(size);

    for (int p = 0; p < size; ++p) {
        sendcounts[p] = rows_per_proc[p] * n;
        displs[p] = displs_rows[p] * n;
        recvcounts[p] = rows_per_proc[p] * n;
        rdispls[p] = displs_rows[p] * n;
    }

    int local_rows = rows_per_proc[rank];
    int local_elems = local_rows * n;

    vector<int> local_A(local_elems);
    vector<int> local_C(local_elems, 0);

    // Разбрасываем части A
    MPI_Scatterv(
        rank == 0 ? A_flat.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_INT,
        local_A.data(), local_elems, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // Локальное умножение: local_C = local_A * B
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += local_A[i * n + k] * B_flat[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }

    // Собираем части C
    MPI_Gatherv(
        local_C.data(), local_elems, MPI_INT,
        rank == 0 ? C_flat.data() : nullptr,
        recvcounts.data(), rdispls.data(), MPI_INT,
        0, MPI_COMM_WORLD
    );

    auto end = chrono::high_resolution_clock::now();

    if (rank == 0) {
        Matrix C(n, vector<int>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] = C_flat[i * n + j];

        write_sq_matrix_to_file(C, "result_matrix");

        chrono::duration<double> diff = end - start;
        double seconds = diff.count();


        cout.setf(ios::scientific);
        cout.precision(3);
        cout << "Time = " << seconds << " s\n";
        cout << "Processes (MPI) = " << size << endl;
    }

    MPI_Finalize();
    return 0;
}
