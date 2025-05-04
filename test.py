import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.interpolate import interp1d
import sofar as sf

# ====== 读取 HRTF 并做 FFT ======
def load_hrtf_fft_binaural(directory):
    left_list, right_list = [], []
    for fname in os.listdir(directory):
        if not fname.endswith(".sofa"):
            continue
        try:
            sofa = sf.read_sofa(os.path.join(directory, fname), verify=False)
            ir = sofa.Data_IR
            pos = sofa.SourcePosition
            idx = np.argmin(np.linalg.norm(pos - np.array([0, 0, 1]), axis=1))
            left = np.abs(np.fft.rfft(ir[idx, 0, :]))
            right = np.abs(np.fft.rfft(ir[idx, 1, :]))
            left_list.append(left)
            right_list.append(right)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    return np.array(left_list), np.array(right_list)

# ====== 提取显著区域（p < 0.05，log空间插值）======
def extract_regions_logspace(freqs, p_vals, threshold=0.05, resolution=10000):
    log_p = np.log10(p_vals)
    f_interp = np.linspace(freqs.min(), freqs.max(), resolution)
    log_interp_func = interp1d(freqs, log_p, kind='linear', fill_value='extrapolate')
    log_vals_interp = log_interp_func(f_interp)
    below = log_vals_interp < np.log10(threshold)

    regions = []
    in_region = False

    for i in range(1, len(f_interp)):
        if i == 1 and below[0]:
            start_f = f_interp[0]
            in_region = True
        elif below[i] and not below[i - 1]:
            start_f = f_interp[i - 1] + (np.log10(threshold) - log_vals_interp[i - 1]) * (f_interp[i] - f_interp[i - 1]) / (log_vals_interp[i] - log_vals_interp[i - 1])
            in_region = True
        elif not below[i] and below[i - 1] and in_region:
            end_f = f_interp[i - 1] + (np.log10(threshold) - log_vals_interp[i - 1]) * (f_interp[i] - f_interp[i - 1]) / (log_vals_interp[i] - log_vals_interp[i - 1])
            regions.append((start_f, end_f))
            in_region = False
    if in_region:
        regions.append((start_f, f_interp[-1]))
    return regions

# ====== 绘图函数 ======
def plot_with_regions(freqs, pL, pR, regions_L, regions_R, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, pL, label="Left Ear p-value", color="blue")
    plt.plot(freqs, pR, label="Right Ear p-value", color="orange")
    plt.axhline(0.05, color='red', linestyle='--', label="p = 0.05")
    for start, end in regions_L:
        plt.axvspan(start, end, color='blue', alpha=0.2)
    for start, end in regions_R:
        plt.axvspan(start, end, color='orange', alpha=0.2)
    plt.yscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("p-value (log scale)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"✔ Plot saved: {filename}")

# ====== 保存 CSV 为图片（带序号） ======
def save_csv_as_image(csv_path, output_image_path):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.4 + 1))
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.show()
    print(f"✔ Table image saved to: {output_image_path}")

# ====== 主流程 ======
# 1. 加载数据
female_left, female_right = load_hrtf_fft_binaural('./female')
male_left,   male_right   = load_hrtf_fft_binaural('./male')

# 2. Welch's t-test
tL, pL = ttest_ind(male_left, female_left, axis=0, equal_var=False)
tR, pR = ttest_ind(male_right, female_right, axis=0, equal_var=False)

# 3. 构建频率轴
n, fs = female_left.shape[1], 44100
freqs = np.fft.rfftfreq(n * 2 - 1, d=1 / fs)
low_mask = freqs <= 20000
high_mask = freqs > 20000

freqs_low, freqs_high = freqs[low_mask], freqs[high_mask]
pL_low, pR_low = pL[low_mask], pR[low_mask]
pL_high, pR_high = pL[high_mask], pR[high_mask]

# 4. 提取显著区域（左右耳）
regions_L_low = extract_regions_logspace(freqs_low, pL_low)
regions_L_high = extract_regions_logspace(freqs_high, pL_high)
regions_R_low = extract_regions_logspace(freqs_low, pR_low)
regions_R_high = extract_regions_logspace(freqs_high, pR_high)

# 5. 保存 CSV（包含序号）
all_L = regions_L_low + regions_L_high
all_R = regions_R_low + regions_R_high
max_len = max(len(all_L), len(all_R))

df = pd.DataFrame({
    "Left Ear Start (Hz)":  [r[0] for r in all_L] + [None] * (max_len - len(all_L)),
    "Left Ear End (Hz)":    [r[1] for r in all_L] + [None] * (max_len - len(all_L)),
    "Right Ear Start (Hz)": [r[0] for r in all_R] + [None] * (max_len - len(all_R)),
    "Right Ear End (Hz)":   [r[1] for r in all_R] + [None] * (max_len - len(all_R))
})
df.insert(0, "Index", pd.Series(range(1, len(df) + 1), dtype="Int64"))
df.to_csv("pvalue_regions_fixed.csv", index=False)
print("✔ CSV saved to: pvalue_regions_fixed.csv")

# 6. 绘图
plot_with_regions(freqs_low, pL_low, pR_low, regions_L_low, regions_R_low,
                  "Welch's t-test on HRTF: Male vs Female (≤20kHz, fixed)",
                  "pvalue_plot_low_fixed.png")

plot_with_regions(freqs_high, pL_high, pR_high, regions_L_high, regions_R_high,
                  "Welch's t-test on HRTF: Male vs Female (>20kHz, fixed)",
                  "pvalue_plot_high_fixed.png")

# 7. CSV 表格转图片
save_csv_as_image("pvalue_regions_fixed.csv", "pvalue_regions_fixed_table.png")
