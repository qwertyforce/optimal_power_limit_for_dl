import torch
# torch.backends.cuda.matmul.allow_tf32 = True
import timm
import subprocess

model = timm.create_model("vit_base_patch16_224")
model.head = torch.nn.Linear(768,1)
batch_size = 160
model=model.to("cuda") #.half()

criterion = torch.nn.BCEWithLogitsLoss()
images = torch.rand((batch_size,3,224,224)).to("cuda") #.half()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 
from torch.cuda.amp import GradScaler
from torch import autocast
scaler = GradScaler()

results = []
for pwr in range(150,490,10):
    subprocess.run(["nvidia-smi","-pl",str(pwr)])
    model.train()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(64):
        labels = torch.randint(0,2,(batch_size,1),dtype=torch.float32).to("cuda")
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.forward(images)
            loss = criterion(outputs,labels) 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        old_scale = scaler.get_scale()
        scaler.update()

        # outputs = model.forward(images)
        # loss = criterion(outputs,labels)
        # loss.backward()
        # optimizer.step()
    end.record()
    torch.cuda.synchronize()
    print((pwr,start.elapsed_time(end)/64))
    results.append((pwr,start.elapsed_time(end)/64))
print(results)