#!/bin/bash
/home/zmy/anaconda3/envs/sam2/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --force-reinstall 2>&1 | tail -20
echo "=== DONE ==="
/home/zmy/anaconda3/envs/sam2/bin/python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('GPU: N/A')
"
