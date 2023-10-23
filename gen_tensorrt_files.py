# docker pull nvcr.io/nvidia/pytorch:23.09-py3
# docker run --gpus all --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,source="$(pwd)",target=/app nvcr.io/nvidia/pytorch:23.09-py3 /bin/bash -c "pip install --pre timm onnx onnxruntime onnx-simplifier && python /app/gen_tensorrt_files.py"
# By @SimJeg https://github.com/pytorch/TensorRT/issues/1860

import os
import warnings
import time
from tempfile import TemporaryDirectory

import torch
import timm
import onnx
from onnxsim import simplify

# Parameters
opset_version = 18 

# Load model
model_name = 'vit_base_patch16_224'
shape = (512, 3, 224, 224)
model = timm.create_model(model_name, exportable=True)
model.eval().cuda().half()

tmpdir="/app"
name = lambda ext: f'{tmpdir}/{model_name}_fp16.{ext}'


# 1. Compile model using ONNX export + TensorRT

# Export to ONNX
with torch.inference_mode(), torch.autocast("cuda"):
    inputs = torch.randn(*shape, dtype=torch.float16, device='cuda')
    torch.onnx.export(model, inputs, name('onnx'), export_params=True, opset_version=opset_version,
                    do_constant_folding=True, input_names = ['input_0'], output_names = ['output_0'])

# Simplify using onnx-simplifier
model = onnx.load(name('onnx'))
simplified_model, check = simplify(model)
if not check:
    warnings.warn('Simplified ONNX model could not be validated, using original ONNX model')
else:
    onnx.save(simplified_model, name('onnx'))

# Convert to TensorRT using default settings
os.system(f'trtexec --onnx={name("onnx")} --saveEngine={name("trt")} --fp16')
os.system(f'torchtrtc {name("trt")} {name("ts")} --embed-engine --device-type=gpu')
