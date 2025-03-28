# loading the meta mlp model from the paper's pretrained weights
# https://github.com/hminle/gamut-mlp/blob/main/pretrained_models/meta_tinycudnn64_metaep3_innersteps10k.pt

# TODO: change 9000 iter to 1200
# 
import torch
from model import GamutMLP

model = GamutMLP()
# checkpoint = torch.load('meta_tinycudnn64_metaep3_innersteps10k.pt')

total_params = sum(p.numel() for p in model.parameters())
model_size_bytes = total_params * 4  # 4 bytes for each float32 parameter

model_size_kB = model_size_bytes / 1024
print(f"Model size: {model_size_kB:.2f} kB")