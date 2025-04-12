import torch
import torch.nn as nn
import os


def init_gpu(gpu_ids):
    if torch.cuda.is_available():
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
            device = torch.device("cuda")
            num_gpus = len(gpu_ids)
            print(f"Using {num_gpus} GPUs: {gpu_ids}")
        else:
            device = torch.device("cuda")
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPUs")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def distribute_model(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    device = init_gpu()
    model = model.to(device)
    return model
    