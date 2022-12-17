import torch
import time

def KL_divergence(tensor1, tensor2, batch_size):
    #tensor* : probability
    return torch.sum(tensor1*(tensor1/tensor2).log()) / batch_size
