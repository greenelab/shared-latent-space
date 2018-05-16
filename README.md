### shared-latent-space
Shared Latent Space VAE's

## main_file.py
This is the file which should be called. It handles loading data and formatting it. It also calls upon shared_vae_class.py to create the mode, train it, and generate data. As work continues, this file will become more general and easier to work with.

## shared_vae_class.py
This file is the main class which hold the model. It contains functions to compile, train, and generate from the model. The model will take in a series of parameters which control size of layers, etc. The model right now is very rigid in structure, but this may change.

## model_objects.py
This file contains the model_parameters class which is fed to the shared_vae_class when it is initialized.

## vae.py
This is a simple variational autoencoder which I implemented before modifying it to be a shared latent space variational autoencoder.
