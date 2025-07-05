import torch
import torch.nn as nn
x = torch.randn(1, 3, 224, 224).cuda()
model = nn.Conv2d(3, 64, 3).cuda()
with torch.no_grad():
    model(x)
print("CUDNN warmup complete")
