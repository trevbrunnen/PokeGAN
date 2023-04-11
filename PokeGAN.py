# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:34:36 2020

@author: trevb

Adapted from the tutorial at 
https://towardsdatascience.com/writing-your-first-generative-adversarial-network-with-keras-2d16fd8d4889

"""


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np


class PokeGAN():
    def __init__(self):
        self.img_rows = 30
        self.img_cols = 42
        self.channels = 4
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.latent_dim = 100
        
        optimizer = Adam(0.0002,0.5)
        
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy',
                                   optimizer = optimizer, metrics = ['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape = (self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z,validity)
        self.combined.compile(loss = 'binary_crossentropy',optimizer = optimizer)
        
        
    def build_generator(self):
        
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))
        model.add(Reshape(self.img_shape))
        
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)
    
    def build_discriminator (self):
        
        model = Sequential()
        
        model.add(Flatten(input_shape = self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()
        
        img = Input(shape = self.img_shape)
        validity = model(img)
        return Model(img, validity)
    
    def train(self, epochs, batch_size=128, sample_interval = 50):
        
        X_train = np.load('cropped.npy')
        
        # X_train = X_train / 127.5 - 1
        # X_train = np.expand_dims(X_train,axis=3)
        
        valid = np.ones((batch_size, 1)) - np.abs(np.random.normal(0,.05,(batch_size,1)))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            
            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]
            
            noise = np.random.normal(0,1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0,1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            
            print("%d [Dloss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0],100*d_loss[1],g_loss))
            
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
    
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r*c,self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        gen_imgs = 0.5 * gen_imgs + .5
        
        fig, axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, :])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
        
        
if __name__== '__main__':
    gan = PokeGAN()
    gan.train(epochs=50001,batch_size=135, sample_interval=1000)