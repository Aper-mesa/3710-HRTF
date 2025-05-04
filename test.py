import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import sofar as sf

def load_hrtf_fft_binaural(directory):
    left_list = []
    right_list = []
    for fname in os.listdir(directory):
        if fname.endswith(".sofa"):
            try:
                sofa = sf.read_sofa(os.path.join(directory, fname), verify=False)
                ir = sofa.Data_IR
                pos = sofa.SourcePosition
                idx = np.argmin(np.linalg.norm(pos - np.array([0, 0, 1]), axis=1))
                left_fft = np.abs(np.fft.rfft(ir[idx, 0, :]))
                right_fft = np.abs(np.fft.rfft(ir[idx, 1, :]))
                left_list.append(left_fft)
                right_list.append(right_fft)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
    return np.array(left_list), np.array(right_list)

def get_significant_regions(freqs, p_values, threshold=0.05, min_width=300):
    regions = []
    in_region = False
    start = 0
    for i, p in enumerate(p_values):
        if p < threshold and not in_region:
            in_region = True
            start = i
        elif p >= threshold and in_region:
            in_region = False
            if freqs[i-1] - freqs[start] > min_width:
                regions.append((freqs[start], freqs[i-1]))
    if in_region and freqs[-1] - freqs[start] > min_width:
        regions.append((freqs[start], freqs[-1]))
    return regions

# 加载数据
female_left, female_right = load_hrtf_fft_binaural('./female')
male_left, male_right = load_hrtf_fft_binaural('./male')

# Welch's t-test
t_left, p_left = ttest_ind(male_left, female_left, axis=0, equal_var=False)
t_right, p_right = ttest_ind(male_right, female_right, axis=0, equal_var=False)

# 频率轴
n = female_left.shape[1]
fs = 44100
freqs = np.fft.rfftfreq(n * 2 - 1, d=1/fs)

# 拆分低频/高频
low_mask = freqs <= 20000
high_mask = freqs > 20000
freqs_low = freqs[low_mask]
freqs_high = freqs[high_mask]
p_left_low = p_left[low_mask]
p_right_low = p_right[low_mask]
p_left_high = p_left[high_mask]
p_right_high = p_right[high_mask]

# 显著频段
regions_left_low = get_significant_regions(freqs_low, p_left_low)
regions_right_low = get_significant_regions(freqs_low, p_right_low)
regions_left_high = get_significant_regions(freqs_high, p_left_high)
regions_right_high = get_significant_regions(freqs_high, p_right_high)

# ----------- 主图：0–20kHz ----------
plt.figure(figsize=(12, 6))
plt.plot(freqs_low, p_left_low, label="Left Ear p-value", color="blue")
plt.plot(freqs_low, p_right_low, label="Right Ear p-value", color="orange")
plt.axhline(0.05, color='red', linestyle='--', label="p=0.05")

top = 10  # 用于标注高度（log scale）
for start, end in regions_left_low:
    mid = (start + end) / 2
    plt.axvspan(start, end, color='blue', alpha=0.2)
    plt.text(mid, top, f"{int(start)}–{int(end)}Hz", color='blue', ha='center', va='top', fontsize=9)

for start, end in regions_right_low:
    mid = (start + end) / 2
    plt.axvspan(start, end, color='orange', alpha=0.2)
    plt.text(mid, top / 2, f"{int(start)}–{int(end)}Hz", color='orange', ha='center', va='top', fontsize=9)

plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("p-value (log scale)")
plt.title("Welch's t-test on HRTF: Male vs Female (≤20kHz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------- 高频图：>20kHz ----------
plt.figure(figsize=(10, 5))
plt.plot(freqs_high, p_left_high, label="Left Ear p-value", color="blue")
plt.plot(freqs_high, p_right_high, label="Right Ear p-value", color="orange")
plt.axhline(0.05, color='red', linestyle='--', label="p=0.05")

top = 10  # 用于高频图标注位置
for start, end in regions_left_high:
    mid = (start + end) / 2
    plt.axvspan(start, end, color='blue', alpha=0.2)
    plt.text(mid, top, f"{int(start)}–{int(end)}Hz", color='blue', ha='center', va='top', fontsize=9)

for start, end in regions_right_high:
    mid = (start + end) / 2
    plt.axvspan(start, end, color='orange', alpha=0.2)
    plt.text(mid, top / 2, f"{int(start)}–{int(end)}Hz", color='orange', ha='center', va='top', fontsize=9)

plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("p-value (log scale)")
plt.title("Welch's t-test on HRTF: Male vs Female (>20kHz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
