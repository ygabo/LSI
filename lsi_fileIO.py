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
  """
  Saves the matrix/data into a file.
  This file will be named according to the name
  parameter, and will be stored in the Results folder.
  The folder is located where this code is run.
  
  Args:
    name (str): will be the name of the file. eg 'BigDoc.txt' 
    matrix (2D list): data structure composed of many rows
  """
  directory = 'Results'
  make_sure_path_exists(directory)
  new_path = directory + '/' + name
  f1=open(new_path, 'w+')
  for row in matrix:
      print >>f1, row
  f1.close()

def make_sure_path_exists(path):
  """
  Makes sure the path given exists.
  It makes it if it doesn't.
  
  Args:
    path (str): name of the path to be created
  """
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
    
def save_matrix_to_file( name, matrix ):
  """
  This will save a matrix/data into 
  a file.
  The file will not be human-readable.
  Useful to preserve state between program executions.
  
  Args:
    name (str): name of file where the matrix is saved
    matrix (2D list): name of matrix we should save to a file
  """
  save_file( name, matrix )

def load_matrix_from_file( name ):
  """
  This will read a file and return the data.
  Limitation is, you should know beforehand that
  the file holds a matrix data.
  
  Args: 
    name (str): name of file to be read
  Returns:
    matrix (2D list/array)
  """
  return load_file( name )

def save_dict_file( filename, dictionary ):
  """
  This will save a dictionary into a file.
   
  Args:
    filename (str): name of file where the dictionary is saved
    dictionary (dict): name of dict we should save to a file
  """
  save_file( filename, dictionary )

def open_dict_file( filename ):
  """
  Load dictionary from a file.
  
  Args:
    filename (str): name of file where the dict is saved
  Returns:
    dictionary
  """
  return load_file( filename )

def save_file( filename, data ):
  """
  Base function everyone calls to save data to a file.
  Uses pickle to dump different kinds of objects to a given file
  
  Args:
    filename (str): name of file where the data object is saved
    data (object): name of object we should save to a file
  """
  f = open( filename, 'wb' ) #b means binary
  pickle.dump( data, f )
  f.close()


def load_file( filename ):
  """
  Base function everyone calls to load data from a file.
  Uses pickle to dump different kinds of objects to a given file
  
  Args:
    filename (str): name of file where the data object is to be read
  Returns:
    data object
  """
  f = open( filename, 'rb' )
  data = pickle.load( f )
  f.close()
  return data
