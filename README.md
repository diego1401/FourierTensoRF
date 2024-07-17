# FourierTensoRF
Parametrizing TensoRF feature maps with the Fourier Transform.

```bash
#TODO put teaser.
```

# Installation

We follow the installation instructions given by nerfstudio. First, the prerequesites must be already be installed.

<details>

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name fourier_tensorf -y python=3.8
conda activate fourier_tensorf
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.

</details>

### Installing nerfstudio and our method

```bash
pip install nerfstudio
pip install -e .
```

If installation was successful the following command should not give an error. You 

```bash
ns-train fourier_tensorf -h
```

# Example command

```bash
ns-train fourier_tensorf --data <path/to/data> blender-free-nerf
```