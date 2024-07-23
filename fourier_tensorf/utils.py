import math
import torch
from torch import Tensor, nn
from typing import Type
from nerfstudio.engine.optimizers import OptimizerConfig
from dataclasses import dataclass
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

def off_center_gaussian(kernel_size,sigma=1):
    '''
    Return off centered gaussian to much the DC of the 2D fft. The kernel returned is normalized as to obtain values between [0,1]
    '''
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )[:kernel_size-1,:kernel_size-1]
    # Make sure sum of values in gaussian kernel equals 1.
    # gaussian_kernel = gaussian_kernel/ torch.sum(gaussian_kernel)
    gaussian_kernel /= torch.max(gaussian_kernel)
    return gaussian_kernel

@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.AdamW
    weight_decay: float = 0
    betas: tuple = (0.9,0.98)
    """The weight decay to use."""
    
class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        exp_x = torch.exp(x)
        ctx.save_for_backward(exp_x)
        return exp_x

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        exp_x = ctx.saved_tensors[0]
        return g * exp_x.clamp(min=1e-6, max=1e6)


trunc_exp = _trunc_exp.apply


class TruncExp(nn.Module):

    @staticmethod
    def forward(x):
        return _trunc_exp.apply(x)