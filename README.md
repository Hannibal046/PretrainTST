# Pretrained PatchTST

## Requirements
```bash
# install pytorch according to the cuda version
pip install transformers accelerate wandb gpustat pandas matplotlib scikit-learn ipywidgets
conda install ipykernel
```

## TODO
Finish the `##TODO` tag in `pretrain.py`(mainly about how to prepare data and how to calculate loss).

## Get Started
``` bash
wandb login
accelerate config
## multi GPU training
accelerate launch --num_processes 4 --gpu_ids 0,1,3,4 pretrain.py
```