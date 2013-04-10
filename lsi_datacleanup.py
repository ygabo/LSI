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

#-------------------------
# DATA CLEAN UP
#-------------------------

def clean_up_data( matrix ):
    # clean up the matrix
    clean = [ clean_up( row[:] ) for row in matrix ]
    return clean

def clean_up_data_no_prune( matrix ):
    matrix = matrix[1:]
    # clean up the matrix
    clean = [ clean_up_no_prune( row[:] ) for row in matrix ]
    return clean

# helper
# wrapper for all the preprocessing functions
def clean_up(matrix_row):
    fresh = remove_words(     # Remove words like an, is, the, etc.
            plurals(          # Convert plural words into singular
            remove_dot(       # Remove periods at the end of words
            str.split(        # Split the string into words, space as a separator
            conv_uni(         # Convert from unicode to string
            remove_punc(      # Remove punctuations like ",!?.  
            strip_html(       # Convert html special characters into equivalent form, instead of acii (might be redundant with remove_func on top)
            merge_rows(       # Merge rows just in case they're separated by the csv reader( for now, one row is one feedback of one student)  
              matrix_row )))) # The actual row
            .lower()))))       # Convert everything into lower case

    #print fresh
    return fresh

# helper
# wrapper for all the preprocessing functions
def clean_up_no_prune(matrix_row):
    fresh = remove_dot(
            plurals(          # Convert plural words into singular
            str.split(        # Split the string into words, space as a separator
            conv_uni(         # Convert from unicode to string
            remove_punc(      # Remove punctuations like ",!?.  
            strip_html(       # Convert html special characters into equivalent form, instead of acii (might be redundant with remove_func on top)
            merge_rows(       # Merge rows just in case they're separated by the csv reader( for now, one row is one feedback of one student)  
              matrix_row )))) # The actual row
            .lower())))      # Convert everything into lower case  
    return fresh

# helper
# convert unicode to string
# if string already, return that
def conv_uni(unicode_row):
    if isinstance(unicode_row, str):
        return unicode_row
    clean = unicodedata.normalize('NFKD', unicode_row).encode('ascii','ignore')
    return clean

# helper
# merge rows in the matrix
# e.g. [ 'hello', 'hi' ] will be [ 'hello hi' ]
def merge_rows(matrix_row):
    t = ''
    for i in matrix_row:
        t += i
    return t

def plurals( words ):
    lmtzr = WordNetLemmatizer()
    new_words = [ lmtzr.lemmatize(elem) for elem in words ]
    return new_words

def remove_words( words ):
    # todo, put these in a file
    extra = ['explain', 'equation', 'screencast', 'difficult', 'video', 'understand',
             'understanding', 'doe', 'bit', 'lol']
    dolch220 = ['a', 'all', 'after', 'always', 'about', 'and', 'am', 'again',
                'around', 'better', 'away', 'are', 'an', 'because', 'bring',
                'bat', 'any', 'been', 'carry', 'blue', 'ate', 'as', 'before',
                'can', 'be', 'ask', 'best', 'cut', 'come', 'by', 'both', 'done',
                'down', 'could', 'buy', 'draw', 'find', 'but', 'every', 'call',
                'drink', 'for', 'came', 'funny', 'did', 'from', 'does', 'fall',
                'go', 'do', 'give', "don't", 'far', 'dont', 'help', 'eat', 'going',
                'full', 'here', 'had', 'got', 'I', 'get', 'has', 'grow', 'in',
                'good', 'her', 'found', 'hold', 'is', 'have', 'him', 'gave',
                'it', 'he', 'his', 'goes', 'hurt', 'jump', 'into', 'how', 'if',
                'like', 'just', 'its', 'keep', 'look', 'must', 'know', 'made',
                'kind', 'make', 'new', 'let', 'many', 'laugh', 'me', 'no',
                'live', 'off', 'my', 'now', 'may', 'or', 'long', 'not', 'on',
                'of', 'much', 'our', 'old', 'read', 'myself', 'play', 'out',
                'once', 'right', 'never', 'please', 'open', 'sing', 'only',
                'run', 'pretty', 'over', 'sit', 'own', 'said', 'ran', 'put',
                'sleep', 'pick', 'see', 'ride', 'round', 'tell', 'the', 'saw',
                'some', 'their', 'shall', 'say', 'stop', 'these', 'show',
                'to', 'she', 'take', 'those', 'so', 'thank', 'upon', 'up',
                'soon', 'them', 'us', 'start', 'we', 'that', 'then', 'use',
                'where', 'there', 'think', 'very', 'today', 'they', 'walk',
                'wash', 'together', 'you', 'this', 'were', 'which', 'try',
                'too', 'when', 'why', 'under', 'wa', 'wish', 'want', 'work', 'was',
                'would', 'well', 'write', 'went', 'your', 'what', 'white',
                'hill', 'who', 'will', 'with', 'yes']    
    words_to_remove = stopwords.words('english') + dolch220 + extra
    return filter(lambda x: x not in words_to_remove, words)

def remove_dot( words ):
    new_words=[]
    for word in words:
        if word.endswith("."):
            new_words += [word[:-1]]
        else:
            new_words += [word]
    return new_words

# helper
# remove punctuations for easy analysis
def remove_punc(sentence):
    # pad punctuations with space
    # just in case so 2 words wont stick together
    # e.g. avoid this: "by this!Also" --> "by thisAlso"
    
    sentence = re.sub('([,!?()])', r' \1 ', sentence)
    # get rid of punctuations
    exclude = set(string.punctuation)

    # don't remove these two symbols
    exclude.remove('-')
    exclude.remove('_')
    exclude.remove('.')

    # remove the punctuations
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    return sentence

# helper
# This function is by Fredrik Lundh
# convert html special characters
def strip_html(text):
    def fixup(m):
        text = m.group(0)
        if text[:1] == "<":
            return "" # ignore tags
        if text[:2] == "&#":
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        elif text[:1] == "&":
            import htmlentitydefs
            entity = htmlentitydefs.entitydefs.get(text[1:-1])
            if entity:
                if entity[:2] == "&#":
                    try:
                        return unichr(int(entity[2:-1]))
                    except ValueError:
                        pass
                else:
                    return unicode(entity, "iso-8859-1")
        return text # leave as is
    return re.sub("(?s)<[^>]*>|&#?\w+;", fixup, text)
