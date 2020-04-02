import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Input, Lambda, Reshape, Flatten, UpSampling2D, MaxPooling2D
from keras.models import Model
import keras.backend as K
from keras import initializers
from tensorflow.python.keras.callbacks import TensorBoard
from keras.losses import mse, binary_crossentropy
from IPython import embed
from PIL import Image
import os
from keras.utils import plot_model
from glob import glob
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


class SVHNGenerator():
    def preprocess_image(self,im):
        im_resized = im.resize((self.image_size,self.image_size))
        img_arr=np.array(im_resized.copy())
        if(not img_arr.shape==self.input_shape):  #Input images should have 32x32x3 size
            return None
        return img_arr

    def data_loader_mat(self):
        data='train_32x32.mat'
        data_dict=loadmat(data)
        x_train=data_dict['X']
        x_train=np.moveaxis(x_train, -1, 0)
        # x_train,x_test=train_test_split(data_imgs,test_size=0.001)
        return x_train

    def data_loader(self): 
        data_imgs=[]
        training_images_path='extra' 
        images_path=glob(os.path.join(training_images_path, '*.png')) 
        for idx,img_path in enumerate(images_path): 
            loaded_img = Image.open(img_path)             
            preprocessed_img=self.preprocess_image(loaded_img)
            if(preprocessed_img is not None):
                data_imgs.append(preprocessed_img)
        x_train,x_test=train_test_split(data_imgs,test_size=0.001)
        x_train,x_test=np.asarray(x_train),np.asarray(x_test)
        return (x_train,x_test) # np array of size (batch,) 

    def __init__(self):
        self.latent_dim = 100        # Dimension of Latent Representation
        self.Encoder = None
        self.Decoder = None
        self.model = None
        self.weights_path = './model weights/svhn.h5'
        
        self.epochs=1000 #TODO answer
        self.batch_size=128 #TODO answer

    # Sampling z from P(z)
    def sample_z(self,args):
        mu, std_sq_log = args
        epsilon = K.random_normal(shape=(K.shape(mu)[0], self.latent_dim), mean=0., stddev=1.)
        z = mu + epsilon * K.sqrt( K.exp(std_sq_log)) 
        return z
            
    def GenerateModel(self):
        b_f = 128
        # ENCODER
        input_ = Input(shape=(32,32,3))

        encoder_hidden1 = Conv2D(filters = b_f, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(input_)
        encoder_hidden1 = BatchNormalization()(encoder_hidden1)
        encoder_hidden1 = Activation('relu')(encoder_hidden1)

        encoder_hidden2 = Conv2D(filters = b_f*2, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden1)
        encoder_hidden2 = BatchNormalization()(encoder_hidden2)
        encoder_hidden2 = Activation('relu')(encoder_hidden2)

        encoder_hidden3 = Conv2D(filters = b_f*4, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden2)
        encoder_hidden3 = BatchNormalization()(encoder_hidden3)
        encoder_hidden3 = Activation('relu')(encoder_hidden3)

        encoder_hidden4 = Flatten()(encoder_hidden3)

        # Latent Represenatation Distribution, P(z)
        z_mean = Dense(self.latent_dim, activation='linear', 
                                  kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden4)
        z_std_sq_log = Dense(self.latent_dim, activation='linear', 
                                  kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden4)

        

        z = Lambda(self.sample_z)([z_mean, z_std_sq_log])


        # DECODER
        decoder_hidden0 = Dense(K.int_shape(encoder_hidden4)[1], activation='relu', kernel_initializer= initializers.he_normal(seed=None))(z)
        decoder_hidden0 = Reshape(K.int_shape(encoder_hidden3)[1:])(decoder_hidden0)

        decoder_hidden1 = Conv2DTranspose(filters = b_f*4, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden0)
        decoder_hidden1 = BatchNormalization()(decoder_hidden1)
        decoder_hidden1 = Activation('relu')(decoder_hidden1)

        decoder_hidden2 = Conv2DTranspose(filters = b_f*2, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden1)
        decoder_hidden2 = BatchNormalization()(decoder_hidden2)
        decoder_hidden2 = Activation('relu')(decoder_hidden2)

        decoder_hidden3 = Conv2DTranspose(filters = b_f, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden2)
        decoder_hidden3 = BatchNormalization()(decoder_hidden3)
        decoder_hidden3 = Activation('relu')(decoder_hidden3)

        decoder_hidden4 = Conv2D(filters = 3, kernel_size= 1, strides = (1,1), padding='valid', kernel_initializer = 'he_normal')(decoder_hidden3)
        decoder_hidden4 = Activation('sigmoid')(decoder_hidden4)
        # MODEL
        vae = Model(input_, decoder_hidden4)

        # Encoder Model
        encoder = Model(inputs = input_, outputs = [z_mean, z_std_sq_log])
        
        # Decoder Model
        no_of_encoder_layers = len(encoder.layers)
        no_of_vae_layers = len(vae.layers)

        decoder_input = Input(shape=(self.latent_dim,))
        decoder_hidden = vae.layers[no_of_encoder_layers+1](decoder_input)

        for i in np.arange(no_of_encoder_layers+2 , no_of_vae_layers-1):
            decoder_hidden = vae.layers[i](decoder_hidden)
        decoder_hidden = vae.layers[no_of_vae_layers-1](decoder_hidden)
        decoder = Model(decoder_input,decoder_hidden )

        self.VAE = vae
        self.Encoder = encoder
        self.Decoder = decoder


        ##Training
        # network parameters
        self.image_size=32
        self.input_shape=(32,32,3)
        self.batch_size = 128
        self.epochs = 100

        # VAE loss = mse_loss or xent_loss + kl_loss
        # if args.mse:
        # else:
        #     reconstruction_loss = binary_crossentropy(K.flatten(inputs),
        #K.flatten(outputs))
        x_train=self.data_loader_mat()
        # x_train = np.reshape(x_train, [-1, image_size, image_size, 3])
        # x_test = np.reshape(x_test, [-1, image_size, image_size, 3])
        x_train = x_train.astype('float32') / 255  #normalising
        # x_test = x_test.astype('float32') / 255
        
        # embed()
        #loss
        # reconstruction_loss = mse(K.flatten(input_), K.flatten(decoder_hidden4))
        reconstruction_loss = binary_crossentropy(K.flatten(input_),K.flatten(decoder_hidden4))
        reconstruction_loss *= self.image_size * self.image_size * self.input_shape[2]

        kl_loss = 1 + z_std_sq_log - K.square(z_mean) - K.exp(z_std_sq_log)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.VAE.add_loss(vae_loss)
        self.VAE.compile(optimizer='rmsprop')
        
        
        # train the autoencoder
        embed()
        vae.fit(x_train,epochs=self.epochs,batch_size=self.batch_size)#,validation_data=(x_test, None))
        plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
        vae.save_weights('vae_customtrained_svhn_extra_im.h5')
        vae.summary()


    # def train(self):
    #     # network parameters
    #     input_shape=(32,32,3)
    #     batch_size = 128
    #     filters = 16
    #     latent_dim = 100
    #     epochs = 30

    #     #Forward pass
    #     # z_mean,z_log_var=self.Encoder.predict(x_train)
    #     # z=sample_z([z_mean,z_log_var])
    #     # output=self.Decoder.predict(z)
        
    #     # VAE loss = mse_loss or xent_loss + kl_loss
    #     # if args.mse:
    #     reconstruction_loss = mse(K.flatten(input), K.flatten(output))
    #     # else:
    #     #     reconstruction_loss = binary_crossentropy(K.flatten(inputs),
    #     #                                             K.flatten(outputs))

    #     reconstruction_loss *= image_size * image_size
    #     kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    #     kl_loss = K.sum(kl_loss, axis=-1)
    #     kl_loss *= -0.5
    #     vae_loss = K.mean(reconstruction_loss + kl_loss)
    #     self.VAE.add_loss(vae_loss)
    #     self.VAE.compile(optimizer='rmsprop')


    #     # train the autoencoder
    #     vae.fit(x_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test, None))
    #     vae.save_weights('vae_customtrained_svhn.h5')
    
    def GetModels(self):
        return self.VAE, self.Encoder, self.Decoder





if __name__ == "__main__":
    Gen = SVHNGenerator()
    Gen.GenerateModel()
    
