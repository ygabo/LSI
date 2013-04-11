from __future__ import division
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from lsi_datacleanup import *
from lsi_fileIO import *
import numpy as np
import string
import operator
import csv
import re
import sets
import os
import errno

def read_answer_log( filename, quizname, numbers ):
    """
    This method reads data from an answer log.
    Each row is a student submission.
    This takes care of multiple submissions and only considers
    the latest submit.
    
    Args:
      filename (str): will be the name of the file. eg 'answer_log'
      quizename (str): name of the quiz to extract. eg 'Lecture_6'
      numbers (int): number that has the essay question in it, for now this is alway 1.
    Returns:
      matrix of the essay answers, each row is a 
    """
    answer_log = csv.reader(open(filename, 'rb'), delimiter='\t', quoting=csv.QUOTE_NONE)
    # convert to a list
    y = list(answer_log)
    # then convert to a numpy matrix
    matrix2 = np.array(y)

    matrix_joined = []
    x = 0
    last, = matrix2.shape
    last = last - 1
    
    while x < last :
        f = matrix2[x]
        matrix_joined = matrix_joined + [f]

        y = x+1
        if y > last:
            break

        while not has_time_stamp( matrix2[y] ):
            tol = matrix_joined[-1]
            lol = matrix2[y]
            matrix_joined[-1] = tol + lol
            y = y + 1
            if y > last:
                break
        # end while loop

        x = y
        if y > last:
            break
    # end while loop

    # only retrieve the test we want
    harvest = [ row for row in matrix_joined if get_test_name_from_line( row ) == quizname]
    # only get the numbers we specified 
    harvest = [ row for row in harvest if get_number_from_line( row ) in [1] ]
    harvest = np.array( harvest)
    harvest = remove_duplicate_submits( harvest, quizname, numbers)

    name_list  = np.array([ get_name_from_line(row) for row in harvest ])
    essay_list = np.array([ get_essay_from_line(row) for row in harvest ])

    dual_list = ( zip( name_list, essay_list ) )

    print_to_file( 'essay_with_names.csv', dual_list)
    print_to_file( 'essay.csv', essay_list)

    harvest = np.array( [ row for row in essay_list if row != '' ] )
    print_to_file( 'noblanks.csv', harvest)

    return harvest

# helper function
def remove_duplicate_submits( matrix, quizname, numbers ):
    """
    This method is used by read_answer_log().
      
    This ensures that the last submitted answer
    from a student is the one we extract
    
    Args:
      matrix (numpy 2D array): matrix that contains the answer log data
      quizname (str): quize name we are currently searching for
      numbers (int): currently set at 1
    Returns:
      matrix without any duplicate submissions
    """
    names = set([])

    # add names to the set
    # the set doesn't add if the name
    # is in it already
    for row in matrix:
        name = get_name_from_line(row)
        names.add(name)
    latest_stamp = 0
    
    # for each name in the set
    # get the latest time stamp
    # then filter out the earlier ones
    while True and len(names) > 0:
        curname = names.pop()
        submatrix = np.array([ row for row in matrix if get_name_from_line(row) == curname])
        # get latest time stamp
        for row in submatrix:
            stamp = get_time_stamp_from_line(row)
            if latest_stamp < stamp:
                latest_stamp = stamp

        # remove duplicates
        matrix = [ row for row in matrix if (curname != get_name_from_line(row)) or (get_time_stamp_from_line(row) == latest_stamp) ]       
        latest_stamp = 0
        if len(names) == 0:
            break
    # abi, TODO
    # simple fix for now
    # remove the guy with this ID
    matrix =  [ row for row in matrix if get_name_from_line(row) != 'ITP1MGS34Q06' ]
    matrix = np.array(matrix)
    return matrix

    
#-------------------------
# COUNTING
#-------------------------#

def count_all_words(matrix):
  """
  This method counts all the words in the matrix.
  
  Args:
    matrix (numpy 2D array): matrix that contains words
  Returns:
    counter object that has the word count for all the words
  """
  c1 = Counter()
  for row in matrix:
    c1 += Counter(row)
  return c1

def count_student_words( words, blankdictionary ):
    """
    This method uses the blank dictionary as a reference.
    Update the values in the blank dictionary with regards to 
    how much they occur in the words matrix.
    
    Args:
      words (numpy 2D array): word matrix
      blankdictionary (dict): contains the words we should be looking for
    Returns:
      dictionary with words from the blankdictionary with frequency updated
    """
    x = dict(blankdictionary)
    for word in words:
        if word in x:
            x[word] = x[word] + 1
    return x

def dict_to_array( dictionary ):
    """
    Converts a dictionary into an arrays
    
    Args:
      dictionary (dict): the dictionary to be converted
    Returns:
     array of words that are sorted by its value
    """
    x = sort_by_key( dictionary )
    x = np.array( x )#, dtype = [('y', '|S11'), ('Value', float)] )
    return x

def recreate_wordle_matrix( dictionary ):
    """
    This method uses the dictionary input to make
    a list of words with a word appearing so many times
    according to its value in the dictionary.
    
    Args:
      dictionary (dict): the dictionary to be read
    Returns:
     array of words
    """
    new_list = []
    for word in dictionary.keys():
        for i in range(int(round(dictionary.get(word)))):
            new_list += [word]
    return new_list

def recreate_wordle_matrix_from_array( array ):
    """
    Converts an array of word:value into an array of words.
    The same as recreate_wordle_matrix() but here we have an
    array.
    
    Args:
      array (numpy array): the dictionary to be converted
    Returns:
     array of words with each word repeating so many times according to the array input
    """
    new_list = []
    for word in c3.keys():
        for i in range(c3.get(word)):
            new_list += [word]
    return new_list

def set_dict_values_to_zero( dictionary ):
    """
    This method sets all the values of a dictionary 
    to 0.
    
    Args:
      dictionary (dict): the dictionary to be reset
    Returns:
     dictionary with words as keys and values set to 0.
    """
    x = dict(dictionary)
    for key in x.keys():
        x[key] = 0
    return x

def set_minimum(dictionary, minimum):
    """
    Filter out elements that do not make the minimum value.
    
    Args:
      dictionary (dict): the dictionary to be checked for values
      minimum (int): minimum value
    Returns:
     dictionary with all its elements having values greater than minimum
    """
    x = dict(dictionary)
    for key in x.keys():
        if x[key] <= minimum:
            del x[key]
    return x

def set_threshold( matrix, dictionary, minimum ):
    """
    TODO: currently not using this
    
    Args:
      array (numpy array): the dictionary to be converted
    Returns:
     array of words with each word repeating so many times according to the array input
    """
    return [ set_minimum(row, dictionary, minimum) for row in matrix ]

def sort_by_key(unsorted):
    """
    This method sorts a dictionary by its key.
    
    Args:
      unsorted (dict): the dictionary to be sorted
    Returns:
     sorted tuple
    """
    sorted_tuple = sorted(unsorted.items(), key=operator.itemgetter(0), reverse=False)
    return sorted_tuple

def sort_by_value(unsorted):
    """
    This method sorts a dictionary by its value.
    
    Args:
      unsorted (dict): the dictionary to be sorted
    Returns:
     sorted tuple
    """
    sorted_tuple = sorted(unsorted.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_tuple

# constuct the word matrix
# only accepts a 2D student comment matrix
def word_matrix( student_comment_matrix, dictionary ):
    """
    This method turns a word matrix into a frequency matrix.
    A matrix of words into numbers that represent frequency.
    
    Args:
      student_comment_matrix (numpy 2D array): student essay matrix
      dictionary (dict): blank dictionary of words that are relevant
    Returns:
     frequency matrix
    """
    z = []
    for student_comment in student_comment_matrix:
        a = count_student_words( student_comment, dictionary )
        # converts the dictionary count into an array, 
        # sorts the words as well
        a = dict_to_array(a)
        a = map(int, a[:,1])
        z = z+[a]
    x = np.array(z)
    return x

def get_essay_from_line( row ):
    """
    This method takes in a row from the answer log and extracts the
    essay submission in the line.
    
    Args:
      row (numpy array): array with different information from the anser log
    Returns:
     answer (str): student answer essay from this line
    """
    # first index of
    # essay answer is 2
    index = 2
    last, = np.array(row).shape
    last  = last - 1 

    answer = str.split(row[index], '[')
    
    index  = index + 1
    while index <= last:
        if len(str.split(row[index], ' ')) < 2:
            break
        answer = answer + [row[index]]
        index  = index + 1     
    answer = ' '.join(answer)
    return answer

def get_name_from_line( row ):
    """
    This method extracts the student ID from the row.
    
    Args:
      row (numpy array): row from the answer log
    Returns:
     string (str): ID that identifies the student
    """
    return str.split(str.split(row[0], ']')[1], '|' )[1]

def get_test_name_from_line( row ):
    """
    This method extracts the test name from the row.
    
    Args:
      row (numpy array): row from the answer log
    Returns:
     string (str): name of the quiz/test in this row
    """
    f = np.array(row)
    return str.split(str.split(row[0], ']')[1], '|' )[2]

def get_time_stamp_from_line( row ):
    """
    This method extracts the time stamp from the row.
    
    Args:
      row (numpy array): row from the answer log
    Returns:
     string (str): timestamp extracted from the row
    """
    return str.split(row[1], ']')

def get_number_from_line( row ):
    """
    This method extracts the number from the row.
    Number is the question number in the quiz.
    
    Args:
      row (numpy array): row from the answer log
    Returns:
     string (str): question number extracted from the row
    """
    return int(str.split(str.split(row[0], ']')[1], '|' )[3])

# Check if this string is a time stamp
# Stamps look like this :
# [Tue Dec 18 15:03:25 2012] |bb_demo_17032|Lecture_1|2|010101
# This just checks if the string has the correct number of elements
def has_time_stamp( string ):
    """
    This method checks if the row has a time stamp.
    
    Args:
      row (numpy array): row from the answer log
    Returns:
     answer (bool): returns True if there is a timestamp, False otherwise
    """
    # check if empty first
    if string == '':
        return False
    if string == None:
        return False
    if string == []:
        return False
    
    # only consider the first element
    # since that is where the time stamp is
    local = string[0]
    local = np.array(local.split())
    
    outer, = local.shape
    if outer != 6: 
        return False
    
    inner, = np.array(local[5].replace('|', ' ').split()).shape
    if inner != 4:
        return False

    # outer has 6 elements
    # inner has 4 elements
    return True
    