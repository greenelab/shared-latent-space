"""
shared-latent-space/unpacked_files.py

This function will unpack the data from their compressed form


Author: Chris Williams
Date: 5/23/18
"""

import os
import cPickle
import gzip
import shutil


def unpackFiles(name):

    train_file_to_read = os.path.join('Data', '{}_Data'.format(name),
                                      'Training',
                                      '{}_Training.pkl.gz'.format(name),
                                      'rb')
    train_file_to_write = os.path.join('Data', '{}_Data'.format(name),
                                       'Training',
                                       '{}_Training.pkl'.format(name),
                                       'wb')
    test_file_to_read = os.path.join('Data', '{}_Data'.format(name),
                                     'Testing',
                                     '{}_Testing.pkl.gz'.format(name),
                                     'rb')
    test_file_to_write = os.path.join('Data', '{}_Data'.format(name),
                                      'Testing',
                                      '{}_Testing.pkl'.format(name),
                                      'wb')

    # Open compressed file
    with gzip.open(train_file_to_read) as f:
        file_content = f.read()

    # Write decompressed file
    with open(train_file_to_write) as f:
        f.write(file_content)

    # Open compressed file
    with gzip.open(test_file_to_read) as f:
        file_content = f.read()

    # Write decompressed file
    with open(test_file_to_write) as f:
        f.write(file_content)
