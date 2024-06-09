import torch

def transfer_to_cuda_and_float32(tensor_dict):
    for key in tensor_dict:
        tensor_dict[key] = tensor_dict[key].to(torch.device('cuda')).float()
    return tensor_dict