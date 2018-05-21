
import numpy as np
from numpy import linalg as LA


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Lambda, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping, Callback
from keras.layers import LeakyReLU
from keras import regularizers

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import os


#import pylab


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], encodedSize))
    return z_mean + K.exp(.5 * z_log_sigma) * epsilon


# Loss function comprised of two parts, Cross_entropy, and divergense
def vae_loss(inputs, finalLayer):
    '''cross_entropy = objectives.binary_crossentropy(inputs,finalLayer)
    kl_divergence = -.5*K.mean(1+ z_log_sigma - K.square(z_mean) 
        - K.exp(z_log_sigma), axis=-1)
    return cross_entropy + kl_divergence
    '''

    reconstruction_loss = K.sum(K.square(finalLayer - inputs))
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) -
                            K.square(K.exp(z_log_sigma)), axis=-1)
    total_loss = K.mean(reconstruction_loss + kl_loss)
    return total_loss

'''
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
'''

'''
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import misc
import matplotlib.pyplot as plt
import os
output = []
for filename in sorted(os.listdir('Data/ICVL_Data/Training/depth/')):
    filename = 'Data/ICVL_Data/Training/depth/' + filename
    temp = misc.imread(filename)
    temp = resize(temp, (temp.shape[0] / 4, temp.shape[1] / 4))
    temp = np.array(temp)
    temp = temp.flatten()

    output.append(temp)


x_train = np.array(output).astype('float32') / np.max(output)
print x_train.shape

noise_factor = .5
x_train_noisy = x_train + (noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_train.shape))
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
'''


def draw_hands(Xs, Ys):
    plt.scatter(Xs, Ys)

    plt.gca().invert_yaxis()

    linesStart = np.array([Xs[0], Xs[1]])
    linesEnd = np.array([Ys[0], Ys[1]])
    plt.plot(linesStart, linesEnd)

    linesStart = np.array([Xs[1], Xs[2]])
    linesEnd = np.array([Ys[1], Ys[2]])
    plt.plot(linesStart, linesEnd, 'g')

    linesStart = np.array([Xs[2], Xs[3]])
    linesEnd = np.array([Ys[2], Ys[3]])
    plt.plot(linesStart, linesEnd, 'g')

    linesStart = np.array([Xs[3], Xs[16]])
    linesEnd = np.array([Ys[3], Ys[16]])
    plt.plot(linesStart, linesEnd, 'g')

    linesStart = np.array([Xs[0], Xs[4]])
    linesEnd = np.array([Ys[0], Ys[4]])
    plt.plot(linesStart, linesEnd, 'r')

    linesStart = np.array([Xs[4], Xs[5]])
    linesEnd = np.array([Ys[4], Ys[5]])
    plt.plot(linesStart, linesEnd, 'r')

    linesStart = np.array([Xs[5], Xs[6]])
    linesEnd = np.array([Ys[5], Ys[6]])
    plt.plot(linesStart, linesEnd, 'r')

    linesStart = np.array([Xs[6], Xs[17]])
    linesEnd = np.array([Ys[6], Ys[17]])
    plt.plot(linesStart, linesEnd, 'r')

    linesStart = np.array([Xs[0], Xs[7]])
    linesEnd = np.array([Ys[0], Ys[7]])
    plt.plot(linesStart, linesEnd, 'm')

    linesStart = np.array([Xs[7], Xs[8]])
    linesEnd = np.array([Ys[7], Ys[8]])
    plt.plot(linesStart, linesEnd, 'm')

    linesStart = np.array([Xs[8], Xs[9]])
    linesEnd = np.array([Ys[8], Ys[9]])
    plt.plot(linesStart, linesEnd, 'm')

    linesStart = np.array([Xs[9], Xs[18]])
    linesEnd = np.array([Ys[9], Ys[18]])
    plt.plot(linesStart, linesEnd, 'm')

    linesStart = np.array([Xs[0], Xs[10]])
    linesEnd = np.array([Ys[0], Ys[10]])
    plt.plot(linesStart, linesEnd, 'y')

    linesStart = np.array([Xs[10], Xs[11]])
    linesEnd = np.array([Ys[10], Ys[11]])
    plt.plot(linesStart, linesEnd, 'y')

    linesStart = np.array([Xs[11], Xs[12]])
    linesEnd = np.array([Ys[11], Ys[12]])
    plt.plot(linesStart, linesEnd, 'y')

    linesStart = np.array([Xs[12], Xs[19]])
    linesEnd = np.array([Ys[12], Ys[19]])
    plt.plot(linesStart, linesEnd, 'y')

    linesStart = np.array([Xs[0], Xs[13]])
    linesEnd = np.array([Ys[0], Ys[13]])
    plt.plot(linesStart, linesEnd, 'b')

    linesStart = np.array([Xs[13], Xs[14]])
    linesEnd = np.array([Ys[13], Ys[14]])
    plt.plot(linesStart, linesEnd, 'b')

    linesStart = np.array([Xs[14], Xs[15]])
    linesEnd = np.array([Ys[14], Ys[15]])
    plt.plot(linesStart, linesEnd, 'b')

    linesStart = np.array([Xs[15], Xs[20]])
    linesEnd = np.array([Ys[15], Ys[20]])
    plt.plot(linesStart, linesEnd, 'b')

with open("Data/ICVL_Data/Training/Annotation_Training.csv", "r") as openfileobject:
    temp = [openfileobject.readline().strip().split(',')]
    for line in openfileobject:
        temp.append(line.strip().split(','))

x_train = np.array(temp).astype('float32')


print x_train.shape
print np.max(x_train)
x_train = np.array(x_train).astype('float32')
x_train = x_train + abs(np.min(x_train))
print np.min(x_train)
x_train = x_train / np.max(x_train)
print np.max(x_train)
x_train_noisy = x_train

'''
noise_factor = .2
x_train_noisy = x_train + (noise_factor *
                           np.random.normal(loc=0.0, scale=1.0, size=x_train.shape))
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
'''
'''
import copy

for i in range(len(x_train)):
    x_train[i] = 1 - x_train[i]

for i in range(len(x_test)):
    x_test[i] = 1 - x_test[i]

for i in range(len(x_train_noisy)):
    x_train_noisy[i] = 1 - x_train_noisy[i]

for i in range(len(x_test_noisy)):
    x_test_noisy[i] = 1 - x_test_noisy[i]
'''


firstLayerSize = 84
secondLayerSize = 48
encodedSize = 2


# Define Encoder
encoderInputs = Input(shape=(x_train.shape[1],))  # Formating the input layer

encoderSecondLayer = Dense(firstLayerSize, activation='relu')(encoderInputs)

encoded = Dense(secondLayerSize, activation='relu')(encoderSecondLayer)
z_mean = Dense(encodedSize)(encoded)
z_log_sigma = Dense(encodedSize)(encoded)

z = Lambda(sampling)([z_mean, z_log_sigma])


encoder = Model(encoderInputs, [z_mean, z_log_sigma, z])

# Decoder
decoderInputs = Input(shape=(encodedSize,))
decoderZeroLayer = Dense(secondLayerSize, activation='relu')(decoderInputs)

decoderSecondLayer = Dense(firstLayerSize, activation='relu')(decoderZeroLayer)

decoded = Dense(x_train.shape[1], activation='sigmoid')(decoderSecondLayer)

decoder = Model(decoderInputs, decoded)

# Full Model
outputs = decoder(encoder(encoderInputs)[2])
vae_model = Model(encoderInputs, outputs)  # Create the full model
vae_model.compile(optimizer='adam', loss=vae_loss)  # Compile

callback = EarlyStopping(monitor='loss',
                         patience=3, verbose=0, mode='auto')


vae_model.fit(x_train_noisy, x_train,
              epochs=150,
              batch_size=96,
              shuffle=True,
              callbacks=[callback])

encoded_imgs = encoder.predict(x_train)[2]
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt


predicted = encoder.predict(x_train,
                            batch_size=96)[2]  # Predict the outputs

max_coords = np.amax(predicted, axis=0)
min_coords = np.amin(predicted, axis=0)

rng = np.random.normal(0, 1, (x_train.shape[0], encodedSize))

generatedImgs = decoder.predict(rng)

'''
x_test_encoded = encoder.predict(x_train, batch_size=96)[2]
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
plt.show()
'''
randIndexes = np.random.randint(0, x_train.shape[0], (10,))


n = 10  # how many digits we will display
plt.figure()
for i in range(n):
    Xs = np.array(x_train[randIndexes[i]][0::3])
    Ys = np.array(x_train[randIndexes[i]][1::3])
    ax = plt.subplot(3, n, i + 1)
    draw_hands(Xs, Ys)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display inv reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    Xs = np.array(decoded_imgs[randIndexes[i]][0::3])
    Ys = np.array(decoded_imgs[randIndexes[i]][1::3])
    draw_hands(Xs, Ys)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display right to left transformed cycled through
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    Xs = np.array(generatedImgs[randIndexes[i]][0::3])
    Ys = np.array(generatedImgs[randIndexes[i]][1::3])
    draw_hands(Xs, Ys)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
