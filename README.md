# PokeGAN
My attempt to use a GAN to generate Pokémon sprites.

This repo contains several different architectures that I used to experiment on making the GAN. 

[PokeGAN.py](./PokeGAN.py) was my intial attempt using a fully connected neural network. As this project is trying to generate images, this approach didn't work well.

[PokeGAN_CNN.py](./PokeGAN_CNN.py) and [PokeGAN_CNN_version2.py](./PokeGAN_CNN_version2.py) were attempts at making sprites using convolutional layers in the GAN. These attempts worked better. I ran several tests over the course of a couple weeks varying parameters of the network. At best, there were some networks that generate images that almost looked like Pokémon sprites if you squinted.

[PokeGAN_CNN_biggerImages.py](PokeGAN_CNN_biggerImages.py) was trained in a similar way on a different data set. This data set had larger images of Pokémon, rather than sprites. Due to the small size of the training set, I was not able to get useful output from the network.

