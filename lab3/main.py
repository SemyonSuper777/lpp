import re
import subprocess
import numpy as np
from pathlib import Path

MATRIX_SIZE = 100
PROCS = [1, 2, 4, 8, 10, 16, 20]
EXE_NAME = "MPI_Lab.exe"


def random_matrix(n, low=0, high=10):
    return np.random.randint(low, high, size=(n, n))


def save_matrix_to_file(matrix: np.ndarray, filename: str) -> None:
    np.savetxt(filename, matrix, fmt="%d", delimiter=" ")


def read_matrix_from_file(filename: str) -> np.ndarray:
    return np.loadtxt(filename, dtype=int)


def run_mpi_and_get_time(exe_path: str, n_procs: int) -> str:
    """Запускает mpiexec -n n_procs exe и возвращает stdout как строку."""
    cmd = ["mpiexec", "-n", str(n_procs), exe_path]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"MPI program failed with code {res.returncode}")
    return res.stdout


def parse_time_from_stdout(stdout: str) -> float:
    """
    Парсит строку вида 'Time = 1.234e-02 s' и возвращает число секунд.
    Предполагается, что такую строку печатает только rank 0.
    """
    m = re.search(r"Time\s*=\s*([0-9.eE+-]+)\s*s", stdout)
    if not m:
        raise ValueError("Cannot find 'Time = ... s' in program output")
    return float(m.group(1))


def main():
    n = MATRIX_SIZE
    exe = Path(EXE_NAME)
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found")

    results = {}

    for p in PROCS:
        print(f"\n=== {p} MPI процессов ===")

        # 1. Генерация матриц
        A = random_matrix(n)
        B = random_matrix(n)
        save_matrix_to_file(A, "matrix_a")
        save_matrix_to_file(B, "matrix_b")
        print(f"Matrices {n}x{n} generated and saved.")

        # 2. Запуск MPI-программы и чтение времени из C++
        stdout = run_mpi_and_get_time(str(exe), p)
        t_cpp = parse_time_from_stdout(stdout)
        results[p] = t_cpp
        print(f"Time from C++: {t_cpp:.6e} s")

        # 3. Верификация результата
        # имя файла должно совпадать с тем, что пишет твой C++-код
        #C_cpp = read_matrix_from_file("result_matrix")  # или "result_matrix"
        #C_np = A @ B

        #if np.array_equal(C_cpp, C_np):
        #    print("Verification: OK (MPI == NumPy)")
        #else:
        #    diff = C_np - C_cpp
        #    max_err = np.max(np.abs(diff))
        #    print("Verification: ERROR, max |diff| =", max_err)
        #    # при желании можно здесь делать break```

    print("\nИтоговые времена (из C++):")
    for p in PROCS:
        print(f"{p} процессов: {results[p]:.6e} s")


if __name__ == "__main__":
    main()
