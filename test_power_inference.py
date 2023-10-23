import torch
# torch.backends.cuda.matmul.allow_tf32 = True
import timm
import subprocess
from torch import autocast

model = timm.create_model("vit_base_patch16_224")
model.head = torch.nn.Linear(768,1)
model=model.to("cuda") #.half()
model.eval()

# import torch_tensorrt
# model = torch.jit.load("./trt_exp/vit_base_patch16_224_best.ts")
# model.eval().half().cuda()


images = torch.rand((512,3,224,224)).to("cuda") #.half()

results = []
for pwr in range(300,490,10):
    subprocess.run(["nvidia-smi","-pl",str(pwr)])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(64):
        with torch.no_grad():
            # with autocast(device_type='cuda', dtype=torch.float16):
            model(images)
    end.record()
    torch.cuda.synchronize()
    print((pwr,start.elapsed_time(end)/64))
    results.append((pwr,start.elapsed_time(end)/64))
print(results)