import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import sofar as sf

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

def extract_significant_regions(freqs, p_values, threshold=0.05):
    """提取连续 p < threshold 的频率区间 (start, end)，自动修复单点段为短区间"""
    below = p_values < threshold
    regions = []
    start_idx = None
    for i in range(len(freqs)):
        if below[i] and start_idx is None:
            start_idx = i
        elif not below[i] and start_idx is not None:
            start_freq = freqs[start_idx]
            end_freq = freqs[i - 1]
            if start_freq == end_freq and 0 < start_idx < len(freqs) - 1:
                delta = freqs[1] - freqs[0]
                start_freq -= delta / 2
                end_freq += delta / 2
            regions.append((start_freq, end_freq))
            start_idx = None
    if start_idx is not None:
        start_freq = freqs[start_idx]
        end_freq = freqs[-1]
        if start_freq == end_freq and start_idx > 0:
            delta = freqs[1] - freqs[0]
            start_freq -= delta / 2
            end_freq += delta / 2
        regions.append((start_freq, end_freq))
    return regions

# ————— 1. 加载数据并计算 p-value ————— #
female_left, female_right = load_hrtf_fft_binaural('./female')
male_left,   male_right   = load_hrtf_fft_binaural('./male')

tL, pL = ttest_ind(male_left, female_left, axis=0, equal_var=False)
tR, pR = ttest_ind(male_right, female_right, axis=0, equal_var=False)

n, fs = female_left.shape[1], 44100
freqs = np.fft.rfftfreq(n * 2 - 1, d=1/fs)

# 拆分频段
low_mask = freqs <= 20000
high_mask = freqs > 20000

freqs_low = freqs[low_mask]
pL_low = pL[low_mask]
pR_low = pR[low_mask]

freqs_high = freqs[high_mask]
pL_high = pL[high_mask]
pR_high = pR[high_mask]

# ————— 2. 提取显著频率区间（修复单点段） ————— #
regions_left_low  = extract_significant_regions(freqs_low, pL_low)
regions_right_low = extract_significant_regions(freqs_low, pR_low)

regions_left_high  = extract_significant_regions(freqs_high, pL_high)
regions_right_high = extract_significant_regions(freqs_high, pR_high)

# 合并所有区段
all_left = regions_left_low + regions_left_high
all_right = regions_right_low + regions_right_high

# ————— 3. 保存为 CSV ————— #
max_len = max(len(all_left), len(all_right))
left_starts = [r[0] for r in all_left] + [None] * (max_len - len(all_left))
left_ends   = [r[1] for r in all_left] + [None] * (max_len - len(all_left))
right_starts = [r[0] for r in all_right] + [None] * (max_len - len(all_right))
right_ends   = [r[1] for r in all_right] + [None] * (max_len - len(all_right))

df = pd.DataFrame({
    "Left Ear Start (Hz)": left_starts,
    "Left Ear End (Hz)": left_ends,
    "Right Ear Start (Hz)": right_starts,
    "Right Ear End (Hz)": right_ends
})

df.to_csv("pvalue_regions.csv", index=False)
print("✔ Saved to: pvalue_regions.csv")