import torch
import imageio
import numpy as np

from typing import Dict
from pathlib import Path
from dataclasses import dataclass,field
from typing import Type

from nerfstudio.data.dataparsers.blender_dataparser import Blender, BlenderDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.cameras.cameras import Cameras, CameraType

from sklearn.cluster import KMeans

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()
    
@dataclass
class BlenderZeroRFDataParserConfig(BlenderDataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: BlenderZeroRF)
    """target class to instantiate"""

    number_of_views: int = 3
    """Number of views chosen for the tas"""

@dataclass
class BlenderZeroRF(Blender):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderZeroRFDataParserConfig

    def __init__(self, config: BlenderZeroRFDataParserConfig):
        super().__init__(config=config)
        

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []

        # Choosing views like in ZeroRF
        if split != 'train' or self.config.number_of_views < 0:
            selected_idxs = list(range(len(meta["frames"])))
        else:
            all_poses = []
            for i,frame in enumerate(meta["frames"]):
                all_poses.append(np.array(frame["transform_matrix"]))
            all_poses = np.array(all_poses).astype(np.float32)
            selected_idxs = kmeans_downsample(all_poses[..., :3, 3], self.config.number_of_views)

        for i,frame in enumerate(meta["frames"]):
            if i not in selected_idxs: continue
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))

        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {}
        if self.config.ply_path is not None:
            metadata.update(self._load_3D_points(self.config.data / self.config.ply_path))
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata,
        )

        return dataparser_outputs