# Plot file for CUDA GEMM Optimization

# Add your plotting code or data here
import re
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Parse the TXT input into structured dictionaries
# ------------------------------------------------------------

def parse_file(path):
    with open(path, "r") as f:
        text = f.read()

    blocks = text.split("Running")
    fp32_block = None
    fp16_block = None

    for b in blocks:
        if "FP32 GEMM profiling" in b:
            fp32_block = b
        if "FP16 GEMM profiling" in b:
            fp16_block = b

    def extract(block):
        if block is None:
            return {}

        pattern = r"Custom GEMM Kernel (.+?)\ncuBLAS GEMM Kernel Performance\s+Latency: ([0-9.]+).*?Effective TFLOPS: ([0-9.]+).*?Custom GEMM Kernel Performance\s+Latency: ([0-9.]+).*?Effective TFLOPS: ([0-9.]+)"
        entries = re.findall(pattern, block, re.S)

        data = {}
        for (name, cublas_lat, cublas_tflops, custom_lat, custom_tflops) in entries:
            data[name.strip()] = {
                "cuBLAS_TFLOPS": float(cublas_tflops),
                "custom_TFLOPS": float(custom_tflops)
            }
        return data

    return extract(fp32_block), extract(fp16_block)


# ------------------------------------------------------------
# 2. Plotting function
# ------------------------------------------------------------

def plot_results(data, title, outname):
    names = list(data.keys())
    cuBLAS_vals = [data[n]["cuBLAS_TFLOPS"] for n in names]
    custom_vals = [data[n]["custom_TFLOPS"] for n in names]

    plt.figure(figsize=(12, 6))
    x = range(len(names))

    # Custom kernel bars
    plt.bar(x, custom_vals, label="Custom Kernel TFLOPS", alpha=0.8)

    # cuBLAS horizontal line
    avg_cublas = sum(cuBLAS_vals) / len(cuBLAS_vals)
    plt.axhline(avg_cublas, color="red", linestyle="--", label=f"cuBLAS TFLOPS ≈ {avg_cublas:.2f}")

    plt.xticks(x, names, rotation=75, ha="right")
    plt.ylabel("TFLOPS")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outname)
    print(f"[Saved] {outname}")


# ------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------

fp32, fp16 = parse_file("input.txt")   # <--- put your filename here

plot_results(fp32, "FP32 Custom GEMM Performance vs cuBLAS", "fp32_vs_cublas.png")
plot_results(fp16, "FP16 Custom GEMM Performance vs cuBLAS", "fp16_vs_cublas.png")
