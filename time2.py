import matplotlib.pyplot as plt
import numpy as np

# Data for image processing effects
effects = ["Oil Painting", "Pencil Sketch", "Sepia", "Negative", "Cartoon"]
cpu_times = [1521, 0.68608, 12, 7, 13]  # Convert seconds to ms where needed
gpu_times = [205.27, 76.99, 0.085696, 0.084512, 0.094944]

# Convert times to NumPy arrays
cpu_times_np = np.array(cpu_times)
gpu_times_np = np.array(gpu_times)

# 1. Bar Chart
plt.figure(figsize=(10, 5))
plt.bar(effects, cpu_times, color='red', alpha=0.6, label='CPU Time (ms)')
plt.bar(effects, gpu_times, color='blue', alpha=0.6, label='GPU Time (ms)')
plt.ylabel("Execution Time (ms)")
plt.title("CPU vs GPU Execution Time")
plt.legend()
plt.show()

# 2. Logarithmic Bar Chart
plt.figure(figsize=(10, 5))
plt.bar(effects, cpu_times, color='red', alpha=0.6, label='CPU Time (ms)')
plt.bar(effects, gpu_times, color='blue', alpha=0.6, label='GPU Time (ms)')
plt.ylabel("Execution Time (ms) [Log Scale]")
plt.yscale("log")  # Log scale for better visibility
plt.title("CPU vs GPU Execution Time (Log Scale)")
plt.legend()
plt.show()

# 3. Scatter Plot (CPU vs GPU)
plt.figure(figsize=(10, 5))
plt.scatter(cpu_times, gpu_times, color='purple', label="Processing Effects")
for i, effect in enumerate(effects):
    plt.annotate(effect, (cpu_times[i], gpu_times[i]))
plt.xlabel("CPU Time (ms)")
plt.ylabel("GPU Time (ms)")
plt.title("CPU vs GPU Execution Time Scatter Plot")
plt.legend()
plt.grid(True)
plt.show()

# 4. Line Chart (Trend Analysis)
plt.figure(figsize=(10, 5))
plt.plot(effects, cpu_times, marker="o", linestyle="--", color="red", label="CPU Time (ms)")
plt.plot(effects, gpu_times, marker="s", linestyle="--", color="blue", label="GPU Time (ms)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time Trend Analysis")
plt.legend()
plt.grid(True)
plt.show()
