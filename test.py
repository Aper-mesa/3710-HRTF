import os
import pysofa2 as sofa

# 设定路径
hrtf_path = './hrtf/'

# 找到第一个 .sofa 文件
sofa_files = [f for f in os.listdir(hrtf_path) if f.endswith('.sofa')]

if not sofa_files:
    print("No SOFA files found in the directory.")
else:
    sofa_file_path = os.path.join(hrtf_path, sofa_files[0])
    print(f"Loading SOFA file: {sofa_file_path}")

    # 用 pysofa2 正确打开
    sofa_data = sofa.open(sofa_file_path)

    # 打印基础信息
    print("\n--- Basic SOFA Information ---")
    # Global attributes
    print(f"Title        : {sofa_data.get('GLOBAL_Title', 'N/A')}")
    print(f"Author       : {sofa_data.get('GLOBAL_Author', 'N/A')}")
    print(f"Organization : {sofa_data.get('GLOBAL_Organization', 'N/A')}")
    print(f"Comment      : {sofa_data.get('GLOBAL_Comment', 'N/A')}")
    print(f"Version      : {sofa_data.get('GLOBAL_Version', 'N/A')}")

    # Sampling rate
    if 'Data_SamplingRate' in sofa_data:
        print(f"Sampling Rate: {sofa_data['Data_SamplingRate'][0]} Hz")
    else:
        print("Sampling Rate: N/A")

    # Data dimensions
    print("\n--- Data Shapes ---")
    for key in ['ListenerPosition', 'ReceiverPosition', 'SourcePosition', 'Data_IR', 'Data_SamplingRate']:
        if key in sofa_data:
            print(f"{key:20}: {sofa_data[key].shape}")
        else:
            print(f"{key:20}: Not available")
