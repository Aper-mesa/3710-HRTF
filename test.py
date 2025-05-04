# ====== 1. 导入库 ======
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import sofar as sf  # 用于读取 SOFA 文件

# ====== 2. 加载 SOFA 中的 HRTF 并取目标方向的 FFT ======
def load_hrtf_fft_binaural(directory):
    left_list, right_list = [], []
    for fname in os.listdir(directory):
        if not fname.endswith(".sofa"):
            continue
        try:
            sofa = sf.read_sofa(os.path.join(directory, fname), verify=False)
            ir = sofa.Data_IR
            pos = sofa.SourcePosition
            idx = np.argmin(np.linalg.norm(pos - np.array([0, 0, 1]), axis=1))  # 找正前方方向
            left = np.abs(np.fft.rfft(ir[idx, 0, :]))
            right = np.abs(np.fft.rfft(ir[idx, 1, :]))
            left_list.append(left)
            right_list.append(right)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    return np.array(left_list), np.array(right_list)

# ====== 3. 提取 p < threshold 的频率区间，包含边界插值 ======
def extract_regions_all_fixed(freqs, p_vals, threshold=0.05):
    below = p_vals < threshold
    regions = []
    in_region = False
    start_freq = None

    for i in range(1, len(freqs)):
        if i == 1 and below[0]:
            start_freq = freqs[0]
            in_region = True
        elif below[i] and not below[i-1]:
            f0, f1 = freqs[i-1], freqs[i]
            p0, p1 = p_vals[i-1], p_vals[i]
            start_freq = f0 + (threshold - p0) * (f1 - f0) / (p1 - p0)
            in_region = True
        elif not below[i] and below[i-1] and in_region:
            f0, f1 = freqs[i-1], freqs[i]
            p0, p1 = p_vals[i-1], p_vals[i]
            end_freq = f0 + (threshold - p0) * (f1 - f0) / (p1 - p0)
            regions.append((start_freq, end_freq))
            in_region = False

    if in_region and start_freq is not None:
        regions.append((start_freq, freqs[-1]))
    return regions

# ====== 4. 加载男女 HRTF 数据 ======
female_left, female_right = load_hrtf_fft_binaural('./female')
male_left,   male_right   = load_hrtf_fft_binaural('./male')

# ====== 5. Welch's t-test 比较男女之间的 HRTF 差异 ======
tL, pL = ttest_ind(male_left, female_left, axis=0, equal_var=False)
tR, pR = ttest_ind(male_right, female_right, axis=0, equal_var=False)

# ====== 6. 构建频率轴，并区分低频（≤20kHz）与高频（>20kHz） ======
n, fs = female_left.shape[1], 44100
freqs = np.fft.rfftfreq(n * 2 - 1, d=1/fs)
low_mask = freqs <= 20000
high_mask = freqs > 20000

freqs_low, freqs_high = freqs[low_mask], freqs[high_mask]
pL_low, pR_low = pL[low_mask], pR[low_mask]
pL_high, pR_high = pL[high_mask], pR[high_mask]

# ====== 7. 提取显著性频率区间 ======
regions_L_low = extract_regions_all_fixed(freqs_low, pL_low)
regions_R_low = extract_regions_all_fixed(freqs_low, pR_low)
regions_L_high = extract_regions_all_fixed(freqs_high, pL_high)
regions_R_high = extract_regions_all_fixed(freqs_high, pR_high)

# ====== 8. 存储所有区间到 CSV 文件 ======
all_L = regions_L_low + regions_L_high
all_R = regions_R_low + regions_R_high
max_len = max(len(all_L), len(all_R))

df = pd.DataFrame({
    "Left Ear Start (Hz)": [r[0] for r in all_L] + [None] * (max_len - len(all_L)),
    "Left Ear End (Hz)":   [r[1] for r in all_L] + [None] * (max_len - len(all_L)),
    "Right Ear Start (Hz)": [r[0] for r in all_R] + [None] * (max_len - len(all_R)),
    "Right Ear End (Hz)":   [r[1] for r in all_R] + [None] * (max_len - len(all_R))
})
df.to_csv("pvalue_regions_fixed.csv", index=False)
print("✔ CSV saved to: pvalue_regions_fixed.csv")

# ====== 9. 绘制低频区间 p 值曲线并高亮显著频段 ======
plt.figure(figsize=(12, 6))
plt.plot(freqs_low, pL_low, label="Left Ear p-value", color="blue")
plt.plot(freqs_low, pR_low, label="Right Ear p-value", color="orange")
plt.axhline(0.05, color='red', linestyle='--', label="p = 0.05")
for start, end in regions_L_low:
    plt.axvspan(start, end, color='blue', alpha=0.2)
for start, end in regions_R_low:
    plt.axvspan(start, end, color='orange', alpha=0.2)
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("p-value (log scale)")
plt.title("Welch's t-test on HRTF: Male vs Female (≤20kHz, fixed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pvalue_plot_low_fixed.png")
plt.show()

# ====== 10. 高频段绘图 ======
plt.figure(figsize=(12, 5))
plt.plot(freqs_high, pL_high, label="Left Ear p-value", color="blue")
plt.plot(freqs_high, pR_high, label="Right Ear p-value", color="orange")
plt.axhline(0.05, color='red', linestyle='--', label="p = 0.05")
for start, end in regions_L_high:
    plt.axvspan(start, end, color='blue', alpha=0.2)
for start, end in regions_R_high:
    plt.axvspan(start, end, color='orange', alpha=0.2)
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("p-value (log scale)")
plt.title("Welch's t-test on HRTF: Male vs Female (>20kHz, fixed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pvalue_plot_high_fixed.png")
plt.show()

# ====== 11. CSV 表格转为图片输出 ======
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
    print(f"✔ Table image saved to: {output_image_path}")
    plt.show()

# 保存为图片
save_csv_as_image("pvalue_regions_fixed.csv", "pvalue_regions_fixed_table.png")
