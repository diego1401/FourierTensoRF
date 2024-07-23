"""
Installing Tensorf with zerorf's decoder as a package.
We register the method with Nerfstudio CLI.
"""

from __future__ import annotations
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipelineConfig,
)

from fourier_tensorf.fourier_tensorf import FourierTensorfModelConfig
from fourier_tensorf.utils import AdamWOptimizerConfig

# Including method 
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig


fourier_tensorf = MethodSpecification(
    config=TrainerConfig(
        method_name="fourier_tensorf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager,
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=FourierTensorfModelConfig(
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=0.002,weight_decay=0.2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=30000),
            },
            "encodings": {
                "optimizer": AdamWOptimizerConfig(lr=0.02),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
            },
            # "camera_opt": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=5000),
            # }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Fourier TensoRF Method",
)
