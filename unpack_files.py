
import cPickle
import gzip

import gzip
import cPickle
import shutil


with gzip.open('Data/MNIST_Data/Training/MNIST_Training.pkl.gz', 'rb') as f:
    file_content = f.read()

with open('Data/MNIST_Data/Training/MNIST_Training.pkl', 'rb') as f_in:
    f_in.write(file_content)


with gzip.open('Data/MNIST_Data/Testing/MNIST_Testing.pkl.gz', 'rb') as f:
    file_content = f.read()

with open('Data/MNIST_Data/Testing/MNIST_Testing.pkl', 'rb') as f_in:
    f_in.write(file_content)

with gzip.open('Data/ICVL_Data/Training/ICVL_Training.pkl.gz', 'rb') as f:
    file_content = f.read()

with open('Data/ICVL_Data/Training/ICVL_Training.pkl', 'rb') as f_in:
    f_in.write(file_content)


with gzip.open('Data/ICVL_Data/Testing/ICVL_Testing.pkl.gz', 'rb') as f:
    file_content = f.read()

with open('Data/ICVL_Data/Testing/ICVL_Testing.pkl', 'rb') as f_in:
    f_in.write(file_content)
