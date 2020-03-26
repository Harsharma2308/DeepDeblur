# DeepDeblur

## Make sure to clone with submodules
```
git clone --recurse-submodules -j8 https://github.com/Harsharma2308/DeepDeblur.git
```

## Anaconda environment
```
conda create --name py37 python=3.7
```

### Prerequisites
- NVIDIA GPU + CUDA CuDNN
- Pytorch


## Training GANs
```
cd gan_train
```
### Install dependencies for training generators
```
pip install -r requirements.txt
```

### Download SVHN train dataset 
```
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
tar -zxvf train.tar.gz
rm -rf train.tar.gz
```
