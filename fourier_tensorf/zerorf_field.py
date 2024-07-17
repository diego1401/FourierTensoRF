# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ZeroRF Field"""

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from typing import Dict, Optional
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import Encoding, Identity, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field

from fourier_tensorf.utils import TruncExp


class CommonDecoder(nn.Module):

    def __init__(self, point_channels, rgb_mode='zerorf'):
        super().__init__()
        assert rgb_mode in ['zerorf', 'zerorf_no_direction', 'zerorf_dedicated_networks']
        self.rgb_mode = rgb_mode
        self.dir_encoder = SHEncoding(levels=3)
        self.base_net = nn.Linear(point_channels, 64)
        self.base_activation = nn.SiLU()
        self.density_net = nn.Sequential(
            nn.Linear(64, 1),
            TruncExp()
        )
        self.dir_net = nn.Linear(9, 64)
        
        self.sigmoid_saturation = 0.001
        self.interp_mode = 'bilinear'
        
        if self.rgb_mode == 'zerorf_dedicated_networks':
            self.color_net = nn.Sequential(
                nn.Linear(64, 3)
                # nn.ReLu()
            )
            self.color_net_specular = nn.Sequential(
                nn.Linear(64, 3)
                # nn.SiLU()
            )
            self.rgb_activation = nn.Sigmoid()
        else:
            self.color_net = nn.Sequential(
                nn.Linear(64, 3),
                nn.Sigmoid()
            )
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        if self.dir_net is not None:
            torch.nn.init.constant_(self.dir_net.weight, 0)

    def get_rgbs(self,feature_x, dirs):
        
        if self.rgb_mode == 'zerorf':
            sh_enc = self.dir_encoder(dirs).to(feature_x.dtype)
            encoded_directions = self.dir_net(sh_enc)
            color_in = self.base_activation(feature_x + encoded_directions)
            rgbs = self.color_net(color_in)
        elif self.rgb_mode == 'zerorf_no_direction': 
            color_in = self.base_activation(feature_x)
            rgbs = self.color_net(color_in)
        elif self.rgb_mode == 'zerorf_dedicated_networks':
            sh_enc = self.dir_encoder(dirs).to(feature_x.dtype)
            encoded_directions = self.dir_net(sh_enc)
            
            color_in_diffuse = self.base_activation(feature_x)
            rgbs_diffuse = self.color_net(color_in_diffuse) # In linear space
            
            color_in_specular = self.base_activation(feature_x + encoded_directions)
            rgbs_specular = self.color_net_specular(color_in_specular) # In linear space
            # Drop out 1% of specular batches to force a high quality diffuse reconstruction
            if self.train:
                drop_out = (torch.rand((rgbs_specular.shape[0],1))<0.99).float().to(rgbs_specular.device).unsqueeze(-1)
                rgbs =  self.rgb_activation(rgbs_diffuse + drop_out * rgbs_specular)
            else:
                rgbs =  self.rgb_activation(rgbs_diffuse + rgbs_specular)
            
        else:
            raise ValueError(f"rgb mode {self.rgb_mode} is not implemented")
        return rgbs

    def forward(self, point_code, dirs, out_sdf=False):
        base_x = self.base_net(point_code)
        base_x_act = self.base_activation(base_x)
        sigmas = self.density_net(base_x_act).squeeze(-1)
        
        if dirs is None:
            rgbs = None
        else:
            rgbs = self.get_rgbs(base_x,dirs)
            
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        return sigmas.unsqueeze(-1), rgbs


class ZeroRFField(Field):
    """ZeroRF Field"""

    def __init__(
        self,
        aabb: Tensor,
        rgb_mode: str,
        # the tensor encoding method used for scene density and appereance
        encoding: Encoding = Identity(in_dim=3)
        
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.encoding = encoding
        self.decoder = CommonDecoder(self.encoding.get_out_dim(), rgb_mode=rgb_mode)

    def get_both_outputs(self, ray_samples: RaySamples):
        d = ray_samples.frustums.directions
        # d = torch.zeros_like(d).cuda()
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        features = self.encoding(positions)
        density, rgb = self.decoder(features, d)

        return density, rgb
    
    def get_density(self, ray_samples: RaySamples):
        d = ray_samples.frustums.directions
        # d = torch.zeros_like(d).cuda()
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        features = self.encoding(positions)
        density, _ = self.decoder(features, d)
        return density

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                density, rgb = self.get_both_outputs(input_rays)

                base_density[mask] = density
                base_rgb[mask] = rgb

                base_density.requires_grad_()
                base_rgb.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            density, rgb = self.get_both_outputs(input_rays)

        output = {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}
        
        #TODO: When tackling view dependence implement a way to visualize each component
        # if self.decoder.rgb_mode == 'zerorf_dedicated_networks'
        return output