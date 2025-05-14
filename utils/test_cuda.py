import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"找到 {num_gpus} 个CUDA设备:")
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"  cuda:{i} - {device_name}")
else:
    print("未找到CUDA设备。")

# 查看当前默认选择的设备
# current_device_index = torch.cuda.current_device()
# print(f"当前PyTorch默认CUDA设备: cuda:{current_device_index} - {torch.cuda.get_device_name(current_device_index)}")