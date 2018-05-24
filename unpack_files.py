"""
shared-latent-space/unpacked_files.py

This function will unpack the data from their compressed form


Author: Chris Williams
Date: 5/23/18
"""

import os
import cPickle
import gzip

import gzip
import cPickle
import shutil


def unpackFiles(name):
    with gzip.open(os.path.join('Data',
                                '{}_Data'.format(name),
                                'Training',
                                '{}_Training.pkl.gz'.format(name)),
                   'rb') as f:
        file_content = f.read()

    with open(os.path.join('Data',
                           '{}_Data'.format(name),
                           'Training',
                           '{}_Training.pkl'.format(name)),
              'wb') as f:
        f_in.write(file_content)

    with gzip.open(os.path.join('Data',
                                '{}_Data'.format(name),
                                'Testing',
                                '{}_Testing.pkl.gz'.format(name)),
                   'rb') as f:
        file_content = f.read()

    with open(os.path.join('Data',
                           '{}_Data'.format(name),
                           'Testing',
                           '{}_Testing.pkl'.format(name)),
              'wb') as f:
        f_in.write(file_content)
