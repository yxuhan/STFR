# Environment
First, install the basic environment as follows: 

```
# create a conda environment and activate it
conda create -n stfr python=3.10
conda activate stfr

# install pytorch
pip install https://download.pytorch.org/whl/cu121/torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu121/torchvision-0.18.1%2Bcu121-cp310-cp310-linux_x86_64.whl

# install other libs
pip install --no-build-isolation \
    onnx==1.18.0 \
    onnxruntime-gpu==1.22.0 \
    opencv-python==4.10.0.84 \
    kornia==0.6.12 \
    tqdm \
    trimesh \
    plyfile \
    matplotlib==3.7.0 \
    lpips \
    tensorboard==2.14.1 \
    mediapy \
    open3d \
    scikit-image==0.25.2 \
    iopath \
    yacs \
    face_alignment

# Optional eye-cap UV island packing (task 1)
pip install shapely==2.1.2

# install chumpy
pip install git+https://github.com/mattloper/chumpy.git@master --no-build-isolation

# install mediapipe and overwrite the numpy version to 1.26.4 (Although there may be some version conflicts on protobuf, the code can still run normally.)
pip install \
    numpy==1.26.4 \
    jaxlib==0.4.20 \
    scipy==1.11.4 \
    opencv-contrib-python==4.10.0.84 \
    mediapipe==0.10.11

# install tinycudann
pip install setuptools==69.5.1
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# install pytorch3d
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2

# install other libs
apt-get update && apt-get install -y xvfb ffmpeg
```

Some steps require special handling (e.g., download weights and install other libs); please refer to the details below.


## Matting
Download the [weights](https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/foreground-segmentation-model-vitl16_384.onnx) of the Soft Foreground Segmentation model from [DAViD](https://github.com/microsoft/DAViD), then put it to `matting/model`.

## Geometry Reconstruction
Download the 2DGS repository and install the required libraries for 2DGS:

```
cd reconstruction
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
cd 2d-gaussian-splatting
pip install submodules/diff-surfel-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
```

Then, install [COLMAP](https://github.com/colmap/colmap) following the [instructions](https://colmap.github.io/install.html). 

## Registration
First, install ibug face-raleted libs:
```
cd registration

git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..

git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```

Then, download [weights](https://drive.google.com/drive/folders/1u_COcmxh--hNjKpdP0162HKGyDLDP12t?usp=drive_link) of my custom landmark detector and put it into `registration/pretrained`.

Next, download `generic_model.pkl` from [FLAME](https://flame.is.tue.mpg.de/) and put it into `registration/align/AlbedoMMFitting/data/FLAME2020`.

Lastly, download the linux version of [Wrap](https://faceform.com/download-wrap/) from [here](https://downloads.faceform.com/file/faceform/Wrap/2025.11.14/9f99598ca4cd3b373d0b26a450d9a47d/Faceform_Wrap_2025.11.14_Linux.tar.xz).
Then, unzip and put it into `registration/wrap`. 
The file structure should be:
```
|-registration
    |- wrap
        |- Fonts
        |- Gallery
        |- lib
        |- plugins
        ...
```
