import subprocess
import re
import statistics
from pathlib import Path

EXE_PATH = Path(r"C:\Users\Liza\source\repos\CUDALab\x64\Release\CUDALab.exe")

N = 512

BLOCK_CONFIGS = [
    (8, 8), (16, 8), (16, 16),
    (32, 8), (32, 16), (32, 32)
]

REPEATS = 5

TIME_PATTERN = re.compile(r"Kernel execution time:\s*([0-9.]+)\s*ms")


def run_exe(n, bx, by):
    cmd = [str(EXE_PATH), str(n), str(bx), str(by)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"Ошибка: {result.stderr}")

    match = TIME_PATTERN.search(result.stdout)
    if match:
        return float(match.group(1))
    raise RuntimeError("Время не найдено")


print(f"CUDA: {N}x{N}")
print("=" * 50)
print(f"Повторов: {REPEATS}")
print()

print("Прогрев...")
run_exe(N, 16, 16)

print("Блок\tПотоков\tСреднее\tМедиана")
print("-" * 35)

for bx, by in BLOCK_CONFIGS:
    times = [run_exe(N, bx, by) for _ in range(REPEATS)]

    avg = statistics.mean(times)
    med = statistics.median(times)

    print(f"{bx}x{by}\t{bx * by}\t{avg:.4f}\t{med:.4f}")

print("\nГотово!")
