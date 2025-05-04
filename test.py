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
                ir = sofa.Data_IR  # (M, R, N)
                pos = sofa.SourcePosition  # (M, 3)

                # 找到最接近前方 (0,0,1)
                idx = np.argmin(np.linalg.norm(pos - np.array([0, 0, 1]), axis=1))

                left_fft = np.abs(np.fft.rfft(ir[idx, 0, :]))
                right_fft = np.abs(np.fft.rfft(ir[idx, 1, :]))
                left_list.append(left_fft)
                right_list.append(right_fft)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
    return np.array(left_list), np.array(right_list)

# 实际数据读取
female_left, female_right = load_hrtf_fft_binaural('./female')
male_left, male_right = load_hrtf_fft_binaural('./male')

# Welch's t-test
t_left, p_left = ttest_ind(male_left, female_left, axis=0, equal_var=False)
t_right, p_right = ttest_ind(male_right, female_right, axis=0, equal_var=False)

# 频率轴
n = female_left.shape[1]
fs = 44100
freqs = np.fft.rfftfreq(n * 2 - 1, d=1/fs)

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(freqs, p_left, label="Left Ear p-value")
plt.plot(freqs, p_right, label="Right Ear p-value")
plt.axhline(0.05, color='red', linestyle='--', label="p=0.05")
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("p-value (log scale)")
plt.title("Welch's t-test on HRTF: Male vs Female")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
