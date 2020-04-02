# DeepDeblur

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

###Dependencies
```
pip install -r requirements.txt
```


## Training Generators
```
cd gan_train
```
### Install dependencies for training generators
```
pip install -r requirements.txt
```

### Download SVHN train dataset 
```
cd data
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
```
