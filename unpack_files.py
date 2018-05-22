
import cPickle
import gzip

import gzip
import cPickle
import shutil


def unpackFiles(name):
    with gzip.open('Data/' + name + '_Data/Training/' + name + '_Training.pkl.gz', 'rb') as f:
        file_content = f.read()

    with open('Data/' + name + '_Data/Training/' + name + '_Training.pkl', 'wb') as f_in:
        f_in.write(file_content)

    with gzip.open('Data/' + name + '_Data/Testing/' + name + '_Testing.pkl.gz', 'rb') as f:
        file_content = f.read()

    with open('Data/' + name + '_Data/Testing/' + name + '_Testing.pkl', 'wb') as f_in:
        f_in.write(file_content)

