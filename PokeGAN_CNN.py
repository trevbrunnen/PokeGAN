# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:34:12 2020

@author: trevb
"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import numpy as np


class PokeGAN_CNN():
    def __init__(self):
        self.img_rows = 30
        self.img_cols = 42
        self.channels = 4
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.latent_dim = 100
        
        #TODO: Use different learning rates for discriminator and generator
        
        optimizer = Adam(0.0004,0.5)
        
        self.discriminator = self.build_discriminator()
        loss = BinaryCrossentropy(label_smoothing=.1)
        self.discriminator.compile(loss = loss,
                                   optimizer = optimizer, metrics = ['accuracy'])
        self.generator = self.build_generator()
        
        optimizer = Adam(0.0001, 0.5)
        
        z = Input(shape = (self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z,validity)
        self.combined.compile(loss = loss, optimizer = optimizer)
        
        
    def build_generator(self):
        
        model = Sequential()
        
        # model.add(Dense(252, input_dim=self.latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Reshape((7,9,4)))
        # model.add(Conv2DTranspose(30,(5,5),strides=2,padding='valid'))
        # model.add(BatchNormalization(momentum=0.8))

        #model.add(Conv2D(5,(5,5), strides = 2))
        #model.add(BatchNormalization(momentum=0.8))
        # model.add(Flatten())
        # model.add(Dense(np.prod(self.img_shape),activation='tanh'))        
        # model.add(Reshape(self.img_shape))
        model.add(Dense(128,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Dense(216))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((6,9,4)))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(60,(3,3),strides=1,padding='valid'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(120,(3,3),strides=1,padding='valid'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(4,(5,5),strides = 1, padding = 'same',
                                  activation = 'tanh'))
        
        
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)
    
    def build_discriminator (self):
        
        model = Sequential()
        
        model.add(Conv2D(64, (5,5),padding='valid', activation = 'relu',
                        input_shape=[self.img_rows,self.img_cols,self.channels]))
        #model.add(Conv2D(64, (5,5),padding='valid', activation = 'relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(30, (3,3),padding='valid', activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()
        
        img = Input(shape = self.img_shape)
        validity = model(img)
        return Model(img, validity)
    
    def train(self, epochs, batch_size=128, sample_interval = 50):
        
        X_train = np.load('cropped.npy')
        
        X_train = X_train*2 - 1
        
        # X_train = X_train / 127.5 - 1
        # X_train = np.expand_dims(X_train,axis=3)
        
        #valid = np.ones((batch_size, 1)) - np.abs(np.random.normal(0,.05,(batch_size,1)))
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        #fake = np.ones((batch_size,1)) *0.05
        
        summary_stats = np.zeros((epochs,6))
        
        for epoch in range(epochs):
            
            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]
            
            #TODO: What happens if you don't use normally distributed random numbers?
            
            noise = np.random.normal(0,1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            for update in range(3):
                noise = np.random.normal(0,1, (batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, valid)

            
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0],100*d_loss[1],g_loss))
            
            
            summary_stats[epoch] = (epoch, d_loss_real[0], d_loss_real[1], 
                                    d_loss_fake[0], d_loss_fake[1], g_loss)
            
            if epoch % sample_interval == 0:
                self.sample_images(epoch, summary_stats)
    
    def sample_images(self, epoch, summary_stats):
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
        
        fig1, axs1 = plt.subplots(3,1)
        axs1[0].plot(summary_stats[0:epoch,0],summary_stats[0:epoch,1],
                     label='Discriminator Loss Real',color = 'b')
        axs1[0].plot(summary_stats[0:epoch,0],summary_stats[0:epoch,3],
                     label='Discriminator Loss Fake',color = 'r')
        axs1[0].set_title('Discrimitator Loss')
        axs1[1].plot(summary_stats[0:epoch,0],summary_stats[0:epoch,2],
                     label='Discriminator Accuracy Real',color = 'b')
        axs1[1].plot(summary_stats[0:epoch,0],summary_stats[0:epoch,4],
                     label='Discriminator Accuracy Fake',color = 'r')
        axs1[1].set_title('Discriminator Accuracy')
        axs1[2].plot(summary_stats[0:epoch,0],summary_stats[0:epoch,5],
                     label='Generator Loss',color = 'g')
        axs1[2].set_title('Generator Loss')
        
        d_loss_patch = mpatches.Patch(color = 'blue', label = 'Real')
        d_acc_patch  = mpatches.Patch(label='Fake',color = 'red')
        g_loss_patch = mpatches.Patch(label='Generator Loss',color = 'green')

        fig1.legend(handles =[d_loss_patch,d_acc_patch, g_loss_patch], 
                    loc='upper right')
        fig1.set_size_inches(18,10)
        fig1.savefig('plots/%d.png' % epoch,dpi=100)
        plt.close()
        
        
if __name__== '__main__':
    gan = PokeGAN_CNN()
    gan.train(epochs=10001,batch_size=100, sample_interval=500)