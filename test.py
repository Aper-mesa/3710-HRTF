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

def get_significant_regions(freqs, p_values, threshold=0.05):
    regions = []
    in_region = False
    start = 0
    for i, p in enumerate(p_values):
        if p < threshold and not in_region:
            in_region = True
            start = i
        elif p >= threshold and in_region:
            in_region = False
            regions.append((freqs[start], freqs[i-1]))
    if in_region:
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

# 获取显著频段
regions_left = get_significant_regions(freqs, p_left)
regions_right = get_significant_regions(freqs, p_right)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(freqs, p_left, label="Left Ear p-value", color="blue")
plt.plot(freqs, p_right, label="Right Ear p-value", color="orange")
plt.axhline(0.05, color='red', linestyle='--', label="p=0.05")

# 标注显著区域 + 频率值
for start, end in regions_left:
    plt.axvspan(start, end, color='blue', alpha=0.2)
    mid = (start + end) / 2
    label = f"{int(start)}–{int(end)}Hz"
    plt.text(mid, 1.5, label, color='blue',
             rotation=0, ha='center', va='bottom', fontsize=8)

for start, end in regions_right:
    plt.axvspan(start, end, color='orange', alpha=0.2)
    mid = (start + end) / 2
    label = f"{int(start)}–{int(end)}Hz"
    plt.text(mid, 1.1, label, color='orange',
             rotation=0, ha='center', va='bottom', fontsize=8)

plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("p-value (log scale)")
plt.title("Welch's t-test on HRTF: Male vs Female")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
