import torch
from torch import Tensor
import torch.nn.functional as F
from fourier_tensorf.utils import off_center_gaussian
from nerfstudio.field_components.encodings import Encoding
from jaxtyping import Float


class FTensorCPEncoding(Encoding):
    """CP Decomposition inferred from Fourier coefficients

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.2, frequency_cap: int = 1) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.register_buffer('frequency_cap',torch.tensor(frequency_cap))
        self.axis=2
        self.fourier_line_coef = torch.nn.Parameter(
            init_scale * torch.ones((3, num_components, resolution, 1)) # initialize as a constant so that fourier coefs have this shape -> [mean, 0, 0, ..., 0]
            )

    def set_frequency_cap(self,value):
        self.frequency_cap.fill_(value)

    def get_frequency_cap(self):
        return self.frequency_cap.item()

    @torch.no_grad()
    def increase_frequency_cap(self):
        '''Function to increase the frequency cap of the fourier coefficients preserved after clipping.'''
        if self.frequency_cap.item() == (self.resolution//2 + 1): return
        old_line = self.get_line_coef()
        self.frequency_cap += 1
        with torch.no_grad():
            self.fourier_line_coef.copy_(old_line)
    def get_line_coef(self):
        f_space = torch.fft.rfft(self.fourier_line_coef,dim=self.axis)
        padded_tensor = torch.zeros_like(f_space)
        _, _, resolution, _ = f_space.shape
        
        f_cap = self.get_frequency_cap()
        padded_tensor[:,:,:f_cap,:]  = f_space[:,:,:f_cap,:]
        return torch.fft.irfft(padded_tensor,dim=self.axis)
    
    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2).detach()
        line_features = F.grid_sample(self.get_line_coef(), line_coord, align_corners=True)  # [3, Components, -1, 1]
        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., Components]

class FTensorVMEncoding(Encoding):
    plane_coef: Float[Tensor, "3 num_components resolution resolution"]
    line_coef: Float[Tensor, "3 num_components resolution 1"]

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
        frequency_cap: int = 100,
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.register_buffer('frequency_cap',torch.tensor(frequency_cap)) # in percentage
        self.register_buffer('filtering_kernel',off_center_gaussian(resolution+1))
        self.axis = 2

        self.plane_coef = torch.nn.Parameter(init_scale * torch.ones((3, num_components, resolution, resolution)))
        self.line_coef = torch.nn.Parameter(init_scale * torch.ones((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def set_frequency_cap(self,value):
        self.frequency_cap.fill_(value)

    def get_frequency_cap(self):
        return self.frequency_cap.item()

    @torch.no_grad()
    def increase_frequency_cap(self):
        '''Function to increase the frequency cap of the fourier coefficients preserved after clipping.'''
        if self.frequency_cap == 100: return
        old_plane = self.get_plane_coef()
        old_line = self.get_line_coef()
        self.frequency_cap += 1
        with torch.no_grad():
            self.line_coef.copy_(old_line)
            self.plane_coef.copy_(old_plane)

    def get_plane_coef(self):
        f_space = torch.fft.fft2(self.plane_coef) #rfft2 applies the fourier transform to the last 2 dimensions
        f_space_shifted = torch.fft.fftshift(f_space)
        
        padded_tensor = torch.zeros_like(f_space_shifted)
        mask = self.filtering_kernel>=(1-self.frequency_cap/100.0)
        padded_tensor.view(padded_tensor.shape[0],padded_tensor.shape[1],-1)[:,:,mask.flatten()] = f_space_shifted.view(padded_tensor.shape[0],padded_tensor.shape[1],-1)[:,:,mask.flatten()]

        padded_tensorf_unshifted = torch.fft.ifftshift(padded_tensor)
        return torch.real(torch.fft.ifft2(padded_tensorf_unshifted))

    def get_line_coef(self):
        f_space = torch.fft.rfft(self.line_coef,dim=self.axis)
        padded_tensor = torch.zeros_like(f_space)
        f_cap = int(f_space.shape[2] * self.frequency_cap/100.0)
        padded_tensor[:,:,:f_cap,:]  = f_space[:,:,:f_cap,:] 
        return torch.fft.irfft(padded_tensor,dim=self.axis)

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2).detach()
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        plane_features = F.grid_sample(self.get_plane_coef(), plane_coord, align_corners=True)  # [3, Components, -1, 1]
        line_features = F.grid_sample(self.get_line_coef(), line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., 3 * Components]