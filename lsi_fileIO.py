from __future__ import division
import numpy as np
import random
import csv
import re
import string
import unicodedata
import operator
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import sets
import os
import errno

#-------------------------
# FILE IO
#-------------------------

def print_to_file( name, matrix ):
    directory = 'Results'
    make_sure_path_exists(directory)
    new_path = directory + '/' + name
    f1=open(new_path, 'w+')
    for row in matrix:
        print >>f1, row
    f1.close()

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    
def save_matrix_to_file( name, matrix ):
    save_file( name, matrix )

def load_matrix_from_file( name ):
    return load_file( name )

def save_dict_file( filename, dictionary ):
    save_file( filename, dictionary )

def open_dict_file( filename ):
    return load_file( filename )

def save_words_file( filename, words ):
    save_file( filename, words )

def open_words_file( filename ):
    return load_file( filename )

def save_file( filename, data ):
    f = open( filename, 'wb' ) #b means binary
    pickle.dump( data, f)
    f.close()

def load_file( filename ):
    f = open( filename, 'rb' )
    data = pickle.load( f )
    f.close()
    return data
