import torch
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenet_s_ssd_lite import create_mobilenet_s_ssd_lite

iterations = 300   # 重复计算的轮次

model = create_mobilenet_s_ssd_lite(num_classes=21, width_mult=1.)
device = torch.device("cuda:0")
model.to(device)

random_input = torch.randn(2, 3, 300, 300).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time/2., 1000/(mean_time/2.)))

