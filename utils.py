import torch
import numpy as np

def unnormalize(image, mean, std):
    image = ((image * torch.as_tensor(std).reshape(1, image.size(1), 1, 1).to(image.device)) + torch.as_tensor(mean).reshape(1, image.size(1), 1, 1).to(image.device))
    return image

def renorm(image):
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-4)
    image = (255 * image).astype(np.uint8)
    return image

def freeze(net):
    for name, param in net.named_parameters():
        if 'bn' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

def compute_FPR(gt, pred):
    FP = ((pred | ~gt)).sum()
    TN = (~pred & ~gt).sum()
    FPR = (FP / (FP + TN)).item()
    return FPR