from __future__ import division
import numpy as np
import random
import csv # csv
import re # regex
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

from gensim import corpora, models, similarities
import gensim
import logging

from sys import argv

# todo, clean up coupling
from lsi_weights import *
from lsi_datacleanup import *
from lsi_fileIO import *
from lsi_matrix import *

def JITT( freq, g, rank, filename, quizname ):
    # read the answer log given by the file name
    # choose the proper quiz name specified 
    # essays are always question 1, so manually specified
    answermatrix = read_answer_log( filename, quizname, [1])
    #print_to_file('rawfile.txt', answermatrix)

    #---------------------------------------------------
    # BIG DOC block
    #---------------------------------------------------

    # handle the pre-processing
    fixed_matrix = clean_up_data(answermatrix)
#    print np.array(fixed_matrix)
    # print to file
    print_to_file('BigDoc.txt', fixed_matrix)
    # construct the initial dictionary
    dictionary = count_all_words(fixed_matrix)    

    #---------------------------------------------------
    # DOC Matrix block
    #---------------------------------------------------

    # get the columns needed for the doc matrix    
    columns = set_dict_values_to_zero(dictionary)
    
    # construct doc matrix
    A_doc_matrix = word_matrix(fixed_matrix, columns)

    #---------------------------------------------------
    # Matrix Preprocessing block
    #---------------------------------------------------

    # preprocess the doc matrix
    # A_prime has words as columns, students as rows
    A_prime = weight_matrix( A_doc_matrix, freq, g )
    print_to_file('A_prime.txt', A_prime)
    
    words = dict_to_array( dictionary )
    # recreate the original matrix from a frequency matrix
    word_array = words[0:,0]
    
    print_to_file('word_array.txt', word_array)
    
    word_freq_A_prime = [ round(np.sum(row)) for row in A_prime.T ]
    zip_freq_to_words = dict(zip(word_array, word_freq_A_prime))

    # create something that wordle understands
    A_prime_wordle = recreate_wordle_matrix(zip_freq_to_words)
    print_to_file('ProcessedMatrix_wordle.txt', A_prime_wordle)

    #---------------------------------------------------
    # Matrix SVD block
    #---------------------------------------------------

    # only get the words that occur > 1
    dict_threshold = set_minimum( zip_freq_to_words, 1 )
    # get the words that do occur > 1, set their occurence to 0
    columns_thresh = set_dict_values_to_zero(dict_threshold)
    # new word matrix
    A_prime_thresh = word_matrix(fixed_matrix, columns_thresh)

    # word array for words that occur > 1
    words_thresh = dict_to_array( dict_threshold )
    # recreate the original matrix from a frequency matrix
    word_array_thresh = words_thresh[0:,0]
    
    # get SVD of the A_prime transpose matrix
    # transpose because, we want the words to be the
    # rows and students as columns
    # it's also weighted
    U,S,VT = np.linalg.svd(A_prime_thresh.T, full_matrices=False)

    # U should have #-of-words rows

    # fix rank first
    (x,_) = A_prime_thresh.shape
    if rank > x:
        rank = x

    U_k = np.array(U[:,:rank])
    S_k = np.array(S[0:rank])
    V_k = np.array(VT[0:rank])
    left = np.dot(U_k, np.eye(rank)*S_k)
    
   # print S_k
   # print V_k

    #----------------------------------------------
    #----------------------------------------------
    Puff = np.array([[ 0,  0,  1,  1,  0,  0,  0,  0,  0,],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  1,],
    [ 0,  1,  0,  0,  0,  0,  0,  1,  0,],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1,],
    [ 1,  0,  0,  0,  0,  1,  0,  0,  0,],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,],
    [ 1,  0,  1,  0,  0,  0,  0,  0,  0,],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1,],
    [ 0,  0,  0,  0,  0,  2,  0,  0,  1,],
    [ 1,  0,  1,  0,  0,  0,  0,  1,  0,],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,]])
    
    Us,Ss,VTs = np.linalg.svd(Puff, full_matrices=False)

    # fix rank first
    (x,_) = A_prime_thresh.shape
    if rank > x:
        rank = x

    U_s = np.array(Us[:,:3])
    S_s = np.array(Ss[0:3])
    V_s = np.array(VTs[0:3])
    lol = np.dot(Us, np.eye(9)*Ss)
    lool= np.dot(lol,VTs)
    
    #print np.dot(np.dot(Us, Ss),VTs)
    #print U_s
    #print Ss
    #print VTs
    #print lool[0]
    #print Puff[0]
    
    Rank_k = np.dot( left, V_k )
    word_freq_Rank_k = [ round(np.sum(row)) for row in Rank_k ]
    Rank_k_dict = dict(zip(word_array_thresh, word_freq_Rank_k))
    Rank_k_wordle = recreate_wordle_matrix(Rank_k_dict)

    filename = 'Rank_' + str(rank) + '_wordle.txt'
    
    print_to_file(filename, Rank_k_wordle)

    top30 = np.array(sort_by_value(Rank_k_dict))[0:31]

    top30words = top30[:,0]
    wordset = set(top30words)

    
    #----------------------------------------------
    # this is without threshold ( all words, even ones that occured once )
    #----------------------------------------------

    print A_prime.shape
    Ul,Sl,VTl = np.linalg.svd(A_prime.T, full_matrices=False)

    # fix rank first
    (x,_) = A_prime_thresh.shape
    if rank > x:
        rank = x

    #print Sl

    ###--------------------------------------------

    # PLOTTING
    #-------------------------------
    
    U_plot = np.array(U[:,1:3])*1000
    V_plot = np.array(VT[1:3])
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    x_axis = np.array(word_array_thresh)
    p, = ax.plot(U_plot.T[0], U_plot.T[1], '.')
    row_anno = 0

    U_anno = U_plot
    for word in word_array_thresh:
        ax.annotate(word, (U_anno[row_anno][0],U_anno[row_anno][1]), size='xx-small')
        row_anno += 1

    temp = 0

    lol =  np.array(zip(range(125),np.array(word_array_thresh)))
    print_to_file('label.txt', lol)
  
    # END #-----------------------------------------------------------

def main(argv):
  log = argv[1] # name of answer log
  quiz = argv[2] # quiz name  
  rank = argv[3] # rank of SVD matrix
  #Lecture_7
  JITT( 0, 0, rank, log, quiz)
  
if __name__=="__main__":
  main(argv)