import time
import threading
import psutil
import tracemalloc
import numpy as np


def measure_time_and_memory(func, *args, **kwargs):
    proc = psutil.Process()
    cpu_readings = []
    monitoring = True

    def monitor_cpu():
        while monitoring:
            cpu_readings.append(psutil.cpu_percent(interval=0.5))

    cpu_thread = threading.Thread(target=monitor_cpu)
    cpu_thread.start()

    tracemalloc.start()
    start_cpu = time.process_time()
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    end_cpu = time.process_time()

    monitoring = False
    cpu_thread.join()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall = end_time - start_time
    cpu_time = end_cpu - start_cpu
    avg_cpu = float(np.mean(cpu_readings)) if cpu_readings else 0.0
    max_cpu = float(np.max(cpu_readings)) if cpu_readings else 0.0
    peak_mem_mb = peak_mem / (1024 * 1024)

    return {
        "result": result,
        "WallTime(s)": wall,
        "CPUTime(s)": cpu_time,
        "CPUUtil_Avg(%)": avg_cpu,
        "CPUUtil_Max(%)": max_cpu,
        "PeakMem(MB)": peak_mem_mb,
    }
