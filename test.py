import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sofar as sf
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind

def load_fft_four_directions_avg(directory, skip=0):
    import os
    left_all, right_all = [], []
    filenames = sorted([f for f in os.listdir(directory) if f.endswith(".sofa")])

    for fname in filenames[skip:]:  # 跳过前 skip 个文件
        try:
            sofa = sf.read_sofa(os.path.join(directory, fname), verify=False)
            ir = sofa.Data_IR
            pos = sofa.SourcePosition

            left_dir = []
            right_dir = []

            for target in [
                np.array([0, 0, 1]),  # front
                np.array([180, 0, -1]),  # back
                np.array([-90, 0, 0]),  # left
                np.array([90, 0, 0])  # right
            ]:
                idx = np.argmin(np.linalg.norm(pos - target, axis=1))
                left_fft = np.abs(np.fft.rfft(ir[idx, 0, :]))
                right_fft = np.abs(np.fft.rfft(ir[idx, 1, :]))
                left_dir.append(left_fft)
                right_dir.append(right_fft)

            left_all.append(np.mean(left_dir, axis=0))
            right_all.append(np.mean(right_dir, axis=0))

        except Exception as e:
            print(f"Error reading {fname}: {e}")

    return np.array(left_all), np.array(right_all)


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
            start_f = f_interp[i - 1] + (np.log10(threshold) - log_vals_interp[i - 1]) * (
                        f_interp[i] - f_interp[i - 1]) / (log_vals_interp[i] - log_vals_interp[i - 1])
            in_region = True
        elif not below[i] and below[i - 1] and in_region:
            end_f = f_interp[i - 1] + (np.log10(threshold) - log_vals_interp[i - 1]) * (
                        f_interp[i] - f_interp[i - 1]) / (log_vals_interp[i] - log_vals_interp[i - 1])
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
import random

# 0. 配置
repeat_times = 5
sample_size = 11

# 1. 加载女性数据
female_left, female_right = load_fft_four_directions_avg('./female')
n, fs = female_left.shape[1], 44100
freqs_all = np.fft.rfftfreq(n * 2 - 1, d=1 / fs)
valid_mask = (freqs_all >= 20) & (freqs_all <= 20000)
freqs = freqs_all[valid_mask]

# 2. 加载全部男性数据
male_left_all, male_right_all = load_fft_four_directions_avg('./male')

# 3. 多次随机抽样
for i in range(1, repeat_times + 1):
    print(f"=== Repeat {i}: Randomly sampling {sample_size} male subjects ===")
    total_male = male_left_all.shape[0]
    indices = random.sample(range(total_male), sample_size)
    male_left = male_left_all[indices]
    male_right = male_right_all[indices]

    # 4. Welch's t-test
    tL, pL = ttest_ind(male_left, female_left, axis=0, equal_var=False)
    tR, pR = ttest_ind(male_right, female_right, axis=0, equal_var=False)
    pL, pR = pL[valid_mask], pR[valid_mask]

    # 5. 提取显著区域（左右耳）
    regions_L = extract_regions_logspace(freqs, pL)
    regions_R = extract_regions_logspace(freqs, pR)

    # 6. 保存 CSV（包含序号）
    max_len = max(len(regions_L), len(regions_R))
    df = pd.DataFrame({
        "Left Ear Start (Hz)": [r[0] for r in regions_L] + [None] * (max_len - len(regions_L)),
        "Left Ear End (Hz)":   [r[1] for r in regions_L] + [None] * (max_len - len(regions_L)),
        "Right Ear Start (Hz)": [r[0] for r in regions_R] + [None] * (max_len - len(regions_R)),
        "Right Ear End (Hz)":   [r[1] for r in regions_R] + [None] * (max_len - len(regions_R))
    })
    df.insert(0, "Index", pd.Series(range(1, len(df) + 1), dtype="Int64"))
    csv_name = f"pvalue_regions_repeat{i}.csv"
    df.to_csv(csv_name, index=False)
    print(f"✔ CSV saved to: {csv_name}")

    # 7. 绘图（只画 20~20000Hz 范围）
    plot_with_regions(freqs, pL, pR, regions_L, regions_R,
                      f"Welch's t-test on HRTF: Random Male (n=11) vs Female (20-20000Hz)",
                      f"pvalue_plot_repeat{i}.png")

    # 8. CSV 表格转图片
    save_csv_as_image(csv_name, f"pvalue_regions_table_repeat{i}.png")
