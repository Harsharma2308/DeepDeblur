'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
 

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from PIL import Image
from keras.optimizers import Adam, RMSprop
import os
from keras.utils import plot_model
from glob import glob
from sklearn.model_selection import train_test_split
from keras import initializers


import matplotlib.pyplot as plt

def preprocess(x):
    return (x/255)*2-1

def deprocess(x):
    return np.uint8((x+1)/2*255)

def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable



class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):

    def __init__(self, img_rows=32, img_cols=32, channel=3):


        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        # if self.D:
        #     return self.D
        # self.D = Sequential()
        # depth = 64
        # dropout = 0.4
        # # In: 28 x 28 x 1, depth = 1
        # # Out: 14 x 14 x 1, depth=64
        # input_shape = (self.img_rows, self.img_cols, self.channel)
        # print(input_shape)
        # self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
        #     padding='same'))
        # self.D.add(LeakyReLU(alpha=0.2))
        # self.D.add(Dropout(dropout))

        # self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        # self.D.add(LeakyReLU(alpha=0.2))
        # self.D.add(Dropout(dropout))

        # self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        # self.D.add(LeakyReLU(alpha=0.2))
        # self.D.add(Dropout(dropout))

        # self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        # self.D.add(LeakyReLU(alpha=0.2))
        # self.D.add(Dropout(dropout))

        # # Out: 1-dim probability
        # self.D.add(Flatten())
        # self.D.add(Dense(1))
        # self.D.add(Activation('sigmoid'))
        # self.D.summary()
        # return self.D

        if self.D:
            return self.D
        self.D = Sequential()
      
        leaky_alpha = .2
      
 
        self.D.add(Conv2D(64, kernel_size=5, strides=2, padding='same',     # 16,16,64
               input_shape=(32,32,3)))
        self.D.add(LeakyReLU(alpha=leaky_alpha))

        self.D.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
        self.D.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=leaky_alpha))

        self.D.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
        self.D.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=leaky_alpha))

        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D
        
       
        # Conv2D(64, kernel_size=5, strides=2, padding='same',     # 16,16,64
        #        input_shape=(32,32,3)),
        # LeakyReLU(alpha=leaky_alpha),
        # Conv2D(128, kernel_size=5, strides=2, padding='same'),   # 8,8,128
        # BatchNormalization(),
        # LeakyReLU(alpha=leaky_alpha),
        # Conv2D(256, kernel_size=5, strides=2, padding='same'),   # 4,4,256
        # BatchNormalization(),
        # LeakyReLU(alpha=leaky_alpha),
        # Flatten(),
        # Dense(1),
        # Activation('sigmoid')        
 


        

    def generator(self):
        # if self.G:
        #     return self.G
        # self.G = Sequential()
        # dropout = 0.4
        # depth = (64+64+64+64)*2
        # dim = 8
        # # In: 100
        # # Out: dim x dim x depth
        # self.G.add(Dense(dim*dim*depth, input_dim=100))
        # self.G.add(BatchNormalization(momentum=0.5))
        # self.G.add(Activation('relu'))
        # self.G.add(Reshape((dim, dim, depth)))
        # self.G.add(Dropout(dropout))

        # # In: dim x dim x depth
        # # Out: 2*dim x 2*dim x depth/2
        # self.G.add(UpSampling2D())
        # self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.5))
        # self.G.add(Activation('relu'))

        # self.G.add(UpSampling2D())
        # self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.5))
        # self.G.add(Activation('relu'))

        # self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.5))
        # self.G.add(Activation('relu'))

        # # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        # self.G.add(Conv2DTranspose(3, 5, padding='same'))
        # self.G.add(Activation('sigmoid'))
        # self.G.summary()

        # if self.G:
        #     return self.G
        leaky_alpha = .2

        self.G = Sequential()
        self.G.add(Dense(4*4*512, input_shape=(100,)))
        self.G.add(Reshape((4, 4, 512)))

        self.G.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=leaky_alpha))
        self.G.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))

        self.G.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=leaky_alpha))
        self.G.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))

        self.G.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=leaky_alpha))
        self.G.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'))

        self.G.add(Activation('tanh'))
        self.G.summary()

        

        # return Sequential([
        #     Dense(4*4*512, input_shape=(input_size,)),
        #     Reshape(target_shape=(4, 4, 512)),                              # 4,4,512
        #     BatchNormalization(),
        #     LeakyReLU(alpha=leaky_alpha),
        #     Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'), # 8,8,256
        #     BatchNormalization(),
        #     LeakyReLU(alpha=leaky_alpha),
        #     Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'), # 16,16,128
        #     BatchNormalization(),
        #     LeakyReLU(alpha=leaky_alpha),
        #     Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'),   # 32,32,3
        #     Activation('tanh')
        # ])


        # gf_dim = 64
        # gan = Sequential()
        # gan.add(Dense(8192, use_bias = True, bias_initializer='zeros', input_dim=100))
        # gan.add(Reshape([4,4,gf_dim*8]))
        # gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        # gan.add(Activation('relu'))
        # gan.add(Conv2DTranspose(gf_dim*4, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        # gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        # gan.add(Activation('relu'))
        # gan.add(Conv2DTranspose(gf_dim*2, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        # gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        # gan.add(Activation('relu'))
        # gan.add(Conv2DTranspose(gf_dim*1, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        # gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        # gan.add(Activation('relu'))
        # gan.add(Conv2DTranspose(3, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        # gan.add(Activation('tanh')) 

        # self.G = gan


        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        #optimizer = RMSprop(lr=0.0001, decay=6e-8  )
        optimizer=Adam(lr=0.01, beta_1=.5)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        #optimizer = RMSprop(lr=0.0001, decay=3e-8)
        optimizer=Adam(lr=0.0001, beta_1=.5)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class MNIST_DCGAN(object):

    def preprocess_image(self,im):
        
        im_resized = im.resize((self.img_rows,self.img_rows))
        img_arr=np.array(im_resized.copy())
        # if(not img_arr.shape==self.input_shape):  #Input images should have 32x32x3 size
        #     return None
        
        img_arr = preprocess(img_arr)
        
        return img_arr


    def data_loader(self): 
        data_imgs=[]
        training_images_path='train' 
        images_path=glob(os.path.join(training_images_path, '*.png')) 
        for idx,img_path in enumerate(images_path): 
            loaded_img = Image.open(img_path)             
            preprocessed_img=self.preprocess_image(loaded_img)
            if(preprocessed_img is not None):
                data_imgs.append(preprocessed_img)
        x_train = data_imgs
        x_train =np.asarray(x_train) 
        return x_train 


    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channel = 3

        self.x_train = self.data_loader()
        
        print(self.x_train.shape)
       
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 3).astype(np.float32)
 

        print(self.x_train.shape)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=128, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            make_trainable(self.discriminator, True)
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y*.9)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            make_trainable(self.discriminator, False)
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols,3])
            image = deprocess(image)
            print(image)
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=1000, batch_size=128, save_interval=500)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)