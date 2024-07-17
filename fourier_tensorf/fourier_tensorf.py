"""
Fourier TensoRF Model file
"""
from __future__ import annotations

import torch
from torch.nn import Parameter
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, cast


from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.encodings import  TensorCPEncoding, TensorVMEncoding, TriplaneEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared, tv_loss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.viewer.viewer_elements import ViewerSlider

from fourier_tensorf.zerorf_field import ZeroRFField
from fourier_tensorf.fourier_tensorf_encoding import FTensorCPEncoding, FTensorVMEncoding

@dataclass
class FourierTensorfModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: FourierTensorfModel)
    """TensoRF model config"""
    resolution: int = 320
    """Render resolution"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_loss": 1.0,
            "tv_reg": 1e-4,
            "l1_reg": 5e-5,
        }
    )
    """Loss specific weights."""
    num_samples: int = 50
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 200
    """Number of samples in density evaluation"""
    num_components: int = 96
    """Number of channels for encoding"""
    tensorf_encoding: Literal["triplane", "vm", "cp","fcp","fvm"] = "cp"
    """tensorf encoding used"""
    regularization: Literal["none", "l1", "tv","fourier_l1"] = "l1"
    """Regularization method used in tensorf paper"""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""
    frequency_cap: int = 1
    """Maximum number of fourier coeffcients to keep after clipping"""
    increase_frequency_cap_every: int = 500
    """Increase the frequency cap every certain number of iterations"""
    rgb_mode: Literal['zerorf', 'zerorf_no_direction', 'zerorf_dedicated_networks'] = "zerorf_dedicated_networks" 
    """Type of network used to infer radiance"""

class FourierTensorfModel(Model):
    """Fourier Tensorf Model Model

    Args:
        config: Fourier Tensorf Model configuration to instantiate model
    """

    config: FourierTensorfModelConfig

    def __init__(
        self,
        config: FourierTensorfModelConfig,
        **kwargs,
    ) -> None:
        self.init_resolution = config.resolution
        self.num_components = config.num_components
        

        super().__init__(config=config, **kwargs)

        if 'f' in self.config.tensorf_encoding:
            def on_change_callback(handle: ViewerSlider) -> None:
                self.field.encoding.set_frequency_cap(int(handle.value))

            max_freq = self.init_resolution//2 + 1
            self.frequency_cap = ViewerSlider(name="Frequency Cap", default_value=self.config.frequency_cap, 
                                              min_value=1, max_value=max_freq, step=1,
                                              cb_hook=on_change_callback)

        # self.b = ViewerNumber(name="Number", default_value=1.0)
        # self.c = ViewerCheckbox(name="Checkbox", default_value=False)
        # self.d = ViewerDropdown(name="Dropdown", default_value="A", options=["A", "B"])
        # self.e = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        # self.f = ViewerText(name="Text", default_value="Hello World")
        # self.g = ViewerVec3(name="3D Vector", default_value=(0.1, 0.7, 0.1))

        self.fourier_l1_weights = None

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        
        def increase_frequency_cap(self, training_callback_attributes: TrainingCallbackAttributes, step:int) -> None:
            if self.config.tensorf_encoding not in ['fcp','fvm']: return 
            if step == 0 or self.config.increase_frequency_cap_every < 0:
                pass
            elif (step%self.config.increase_frequency_cap_every)==0:
                self.field.encoding.increase_frequency_cap()
                self.frequency_cap.value = self.field.encoding.get_frequency_cap()

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                func=increase_frequency_cap,
                args=[self, training_callback_attributes],
            )
        ]
        return callbacks
    
    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        if self.config.tensorf_encoding == "vm":
            encoding = TensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_components,
            )
        elif self.config.tensorf_encoding == "cp":
            encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_components,
            )
        elif self.config.tensorf_encoding == "triplane":
            encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_components
            )
        elif self.config.tensorf_encoding == 'fcp':
            encoding = FTensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_components,
                frequency_cap=self.config.frequency_cap
            )

        elif self.config.tensorf_encoding == 'fvm':
            encoding = FTensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_components,
                frequency_cap=self.config.frequency_cap
            )
        else:
            raise ValueError(f"Encoding {self.config.tensorf_encoding} not supported")

        self.field = ZeroRFField(
            self.scene_box.aabb,
            encoding=encoding,
            rgb_mode=self.config.rgb_mode,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

        # regularizations
        if self.config.tensorf_encoding == "cp" and self.config.regularization == "tv":
            raise RuntimeError("TV reg not supported for CP decomposition")

        # (optional) camera optimizer
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["fields"] = (
            list(self.field.decoder.parameters())
        )
        param_groups["encodings"] = list(self.field.encoding.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)
        pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        rgb_loss = self.rgb_loss(image, pred_image)

        loss_dict = {"rgb_loss": rgb_loss}

        if self.config.regularization == "fourier_l1":
            l1_parameters = []
            for parameter in self.field.encoding.parameters():
                f_space_parameter = torch.fft.rfft(parameter,dim=2)
                
                if self.fourier_l1_weights is None:
                    weights = torch.linspace(0, 1, steps=f_space_parameter.shape[-2])**3
                    self.fourier_l1_weights = (weights.repeat(3,f_space_parameter.shape[1],1).unsqueeze(-1).cuda())
                    
                l1_soft = torch.real(f_space_parameter* self.fourier_l1_weights)
                l1_parameters.append(l1_soft.view(-1))
            loss_dict["l1_reg"] = torch.abs(torch.cat(l1_parameters)).mean()

        elif self.config.regularization == "l1":
            l1_parameters = []
            for parameter in self.field.encoding.parameters():
                l1_parameters.append(parameter.view(-1))
            loss_dict["l1_reg"] = torch.abs(torch.cat(l1_parameters)).mean()
        elif self.config.regularization == "tv":
            assert hasattr(self.field.encoding, "plane_coef") and \
                   isinstance(self.field.encoding.plane_coef, torch.Tensor),\
                "TV reg only supported for TensoRF encoding types with plane_coef attribute"

            loss_dict["tv_reg"] = tv_loss(self.field.encoding.plane_coef)
        elif self.config.regularization == "none":
            pass
        else:
            raise ValueError(f"Regularization {self.config.regularization} not supported")

        self.camera_optimizer.get_loss_dict(loss_dict)

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = cast(torch.Tensor, self.ssim(image, rgb))
        lpips = self.lpips(image, torch.clamp(rgb,0.0,1.0))

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        self.camera_optimizer.get_metrics_dict(metrics_dict)

        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict


