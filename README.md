# DeepDeblur

## Make sure to clone with submodules
This is a repository for the 10701 project "Blur Kernel Estimation and Tracking via GANs". The

## Make sure to clone with submodules
```
git clone --recurse-submodules https://github.com/Harsharma2308/DeepDeblur.git
```

## Anaconda environment
```
conda create --name py37 python=3.7
```


### Prerequisites
- NVIDIA GPU + CUDA CuDNN
- Pytorch

### Dependencies
```
pip install -r requirements.txt
```


## Training Generators
```
cd DeepDeblur/Blind-Image-Deconvolution-using-Deep-Generative-Priors/scripts
python svhn_train.py
python blur_train.py
python DCGAN_train.py
```
## Inference
```
cd gan_train
Run `deblurring_SVHN_wo_init.py`  for running the algorithm without the modified initialisation of latent space.
Run `deblurring_SVHN_with_init.py`  for running the algorithm with initialisation of latent space.
```


### Install dependencies for training generators
```
pip install -r requirements.txt
```

## Datasets

### Download SVHN train dataset 
```
cd gan_train/data
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
```

### Download Blurryvideo dataset
[blurVideoSVHN](https://drive.google.com/drive/folders/15aGZ9PlWYYpyENXTiInfvIZACP89-f8E?usp=sharing)

### Create Blurkernel dataset
```
cd Blind-Image-Deconvolution-using-Deep-Generative-Priors/blur_data_generation
matlab -nodisplay -nodesktop -r "run blur_data_generate.m"
```

### Hyper parameters used

REGULARIZORS = [0.01 , 0.01]
alpha 		= 1.0 (for Algorithm 2)
NOISE_STD       = 0.01
STEPS           = 6000
